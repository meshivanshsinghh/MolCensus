"""Gemini on Vertex AI for BioMiner (scripts/aws/molcensus_vertex.py and the patched runner/mllm.py), with a
fake sign-in and a fake endpoint. No network. Run with: pytest -q

The last test patches a real copy of BioMiner's mllm.py, so it needs a BioMiner clone at commit 17c6161
(BIOMINER_DIR, or tools/BioMiner); it is skipped without one, as in CI.
"""

import http.server
import json
import os
import subprocess
import sys
import threading
from datetime import timedelta
from pathlib import Path

import pytest

google_credentials = pytest.importorskip("google.auth.credentials")
httpx = pytest.importorskip("httpx")
openai = pytest.importorskip("openai")
from google.auth import _helpers  # noqa: E402

AWS = Path(__file__).resolve().parents[1] / "scripts" / "aws"
sys.path.insert(0, str(AWS))
import biominer_gemini  # noqa: E402
import molcensus_vertex  # noqa: E402


class FakeCredentials(google_credentials.Credentials):
    """Hands out tok-1, tok-2, ...; each lives `lifetime`."""

    def __init__(self, lifetime):
        super().__init__()
        self.lifetime, self.refreshes = lifetime, 0

    def refresh(self, request):
        self.refreshes += 1
        self.token, self.expiry = f"tok-{self.refreshes}", _helpers.utcnow() + self.lifetime


def test_token_is_reused_until_it_nears_expiry(monkeypatch):
    fake = FakeCredentials(timedelta(minutes=60))
    monkeypatch.setattr(molcensus_vertex, "_credentials", fake)
    assert molcensus_vertex.credentials().token == "tok-1"
    assert molcensus_vertex.credentials().token == "tok-1"  # 60 min left: no new token
    fake.expiry = _helpers.utcnow() + timedelta(minutes=3)  # inside google-auth's 3 min 45 s margin
    assert molcensus_vertex.credentials().token == "tok-2"


def status_error(status, body=None, headers=None):
    response = httpx.Response(status, request=httpx.Request("POST", "https://x/chat/completions"), headers=headers or {})
    return openai.APIStatusError(f"Error code: {status} - {body}", response=response, body=body)


def google_429(delay=None):
    details = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}] if delay else []
    return [{"error": {"code": 429, "message": "Resource exhausted.", "status": "RESOURCE_EXHAUSTED", "details": details}}]


def test_retry_rules():
    wait = molcensus_vertex.retry_wait
    assert [wait(status_error(503), tries) for tries in range(1, 7)] == [15, 30, 60, 120, 240, None]
    assert wait(status_error(429, google_429()), 1) == 15  # no wait asked: the usual schedule
    assert wait(status_error(429, google_429("40s")), 1) == 40  # at least the wait it asks for
    assert wait(status_error(429, google_429("40s")), 3) == 60
    assert wait(status_error(429, google_429("21732s")), 1) is None  # a spent daily quota: give up at once
    assert wait(status_error(429, None, {"retry-after": "1000"}), 1) is None
    assert wait(status_error(400), 1) is None and wait(status_error(401), 1) is None
    request = httpx.Request("POST", "https://x")
    assert wait(openai.APIConnectionError(request=request), 1) == 15
    assert wait(openai.APITimeoutError(request=request), 2) == 30


def answer(model="google/gemini-3.5-flash-001", content='{"ok": true}'):
    return 200, {"id": "x", "object": "chat.completion", "created": 0, "model": model,
                 "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
                 "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 20}}


@pytest.fixture
def endpoint():
    """A local stand-in for Vertex's /chat/completions: send() queues replies, requests records what came in."""
    replies, requests = [], []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append({"path": self.path, "authorization": self.headers["Authorization"], "body": body})
            status, payload = replies.pop(0)
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/v1", replies.extend, requests
    server.shutdown()


def biominer_clone():
    for candidate in (os.environ.get("BIOMINER_DIR"), Path(__file__).resolve().parents[1] / "tools" / "BioMiner"):
        if candidate and subprocess.run(["git", "-C", str(candidate), "cat-file", "-e", "17c6161"], capture_output=True).returncode == 0:
            return Path(candidate)
    pytest.skip("needs a BioMiner clone at commit 17c6161 (BIOMINER_DIR or tools/BioMiner)")


def test_patched_biominer_client_against_fake_endpoint(tmp_path, monkeypatch, capsys, endpoint):
    clone = biominer_clone()
    fake_clone = tmp_path / "BioMiner"  # the upstream files the patch edits, as at 17c6161
    for name in ["BioMiner/runner/mllm.py"] + [f"scripts/{s}_server.py" for s in ("mineru", "moldet", "ocsr")]:
        (fake_clone / name).parent.mkdir(parents=True, exist_ok=True)
        (fake_clone / name).write_text(subprocess.run(["git", "-C", str(clone), "show", f"17c6161:{name}"],
                                                      capture_output=True, text=True, check=True).stdout)
    (fake_clone / "BioMiner/commons/1d_r_groups").mkdir(parents=True)  # where mllm.py keeps its IUPAC cache
    subprocess.run([sys.executable, str(AWS / "patch_biominer.py"), str(fake_clone)], check=True, capture_output=True)
    monkeypatch.chdir(tmp_path)  # patched_client_code changes into the clone; this changes back afterwards
    mllm = biominer_gemini.patched_client_code(fake_clone)
    assert mllm.molcensus_vertex is molcensus_vertex

    fake = FakeCredentials(timedelta(minutes=3))  # always inside the margin: a new token for every request
    monkeypatch.setattr(molcensus_vertex, "_credentials", fake)
    waits = []
    monkeypatch.setattr(mllm.time, "sleep", waits.append)
    url, send, requests = endpoint
    client = mllm.get_api_client("gemini-3.5-flash", url, "unused")

    # 503, then a 429 asking for 40 s, then an answer: two retries, one usage line, no lost call.
    send([(503, {"error": {"code": 503, "message": "high demand"}}), (429, google_429("40s")), answer()])
    assert mllm.call_api_iamge(client, "Find the molecules.", "gemini-3.5-flash", "iVBORw0K") == '{"ok": true}'
    assert waits == [15, 40] and len(requests) == 3
    assert [r["authorization"] for r in requests] == ["Bearer tok-1", "Bearer tok-2", "Bearer tok-3"]  # never "unused"
    body = requests[0]["body"]
    assert requests[0]["path"] == "/v1/chat/completions" and body["model"] == "google/gemini-3.5-flash"
    assert body["response_format"] == {"type": "json_object"} and body["max_tokens"] == 4096 and body["temperature"] == 0
    image, text = body["messages"][1]["content"]
    assert image["image_url"] == {"url": "data:image/png;base64,iVBORw0K", "detail": "high"} and text["text"] == "Find the molecules."
    out = capsys.readouterr().out
    assert out.count("try failed, retrying in") == 2 and " try failed:" not in out
    usage = [json.loads(line.split(" ", 1)[1]) for line in out.splitlines() if line.startswith("MOLCENSUS_USAGE ")]
    assert usage == [{"model": "google/gemini-3.5-flash-001", "input": 10, "output": 5, "total": 20}]

    # A 429 with a long retry delay (a spent quota), then a 400: each gives up at once, as one lost call.
    for reply in [(429, google_429("21732s")), (400, {"error": {"code": 400, "message": "bad request"}})]:
        waits.clear(), requests.clear()
        send([reply])
        assert mllm.call_api_iamge(client, "Find the molecules.", "gemini-3.5-flash", "iVBORw0K") == "[]"
        assert waits == [] and len(requests) == 1
        assert capsys.readouterr().out.count(" try failed:") == 1

    # The text call goes the same way.
    send([answer(content='{"rows": []}')])
    assert mllm.call_api_text(client, "Read this. ", "gemini-3.5-flash", "Table 1") == '{"rows": []}'
    assert requests[-1]["body"]["messages"][1]["content"] == "Read this. Table 1"
