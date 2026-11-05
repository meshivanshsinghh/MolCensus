"""Gemini on Google Cloud Vertex AI for BioMiner: sign-in, a fresh Google token on every request, retry rules.

patch_biominer.py copies this file into the BioMiner clone (next to example_open_source.py) and makes
BioMiner's runner/mllm.py use it; biominer_gemini.py uses it for its checks. It needs google-auth, httpx
and openai, which BioMiner's environment has.

Sign-in, read from the environment (run_all.sh loads ~/vertex.env; none of it is a secret):

    VERTEX_WIF_CONFIG        On AWS. The one-line JSON from `gcloud iam workload-identity-pools
                             create-cred-config --aws --enable-imdsv2`. google-auth reads the instance role's
                             AWS credentials from the metadata service (IMDSv2), trades them at Google's STS
                             for a federated token, and trades that for a token of the service account named
                             in the JSON. No key is stored anywhere.
    VERTEX_AUTH=impersonate  On the Mac instead. Your gcloud login (application-default credentials)
                             impersonates VERTEX_SERVICE_ACCOUNT, with VERTEX_PROJECT as quota project.

A Google token lasts one hour. credentials() gets a new one when less than google-auth's margin
(3 min 45 s) is left, and every request asks for it, so a run of many hours never sends an expired token.
"""

from __future__ import annotations

import json
import os
import re
import threading

SCOPE = "https://www.googleapis.com/auth/cloud-platform"
MODEL_PREFIX = "google/"  # Vertex's OpenAI-compatible endpoint names Gemini models google/<name>
LONG_429_S = 300  # a 429 that asks for a longer wait than this is a spent quota, not a busy moment: give up

_lock = threading.Lock()
_credentials = None


def base_url(project: str) -> str:
    return f"https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/endpoints/openapi"


def _sign_in():
    import google.auth
    from google.auth import impersonated_credentials

    if os.environ.get("VERTEX_AUTH") == "impersonate":
        source, _ = google.auth.default(scopes=[SCOPE], quota_project_id=os.environ["VERTEX_PROJECT"])
        return impersonated_credentials.Credentials(
            source_credentials=source, target_principal=os.environ["VERTEX_SERVICE_ACCOUNT"],
            target_scopes=[SCOPE], lifetime=3600,
        )
    credentials, _ = google.auth.load_credentials_from_dict(json.loads(os.environ["VERTEX_WIF_CONFIG"]), scopes=[SCOPE])
    return credentials


def credentials():
    """Signed-in credentials with a token that has more than 3 min 45 s left. Safe to call from threads."""
    global _credentials
    from google.auth.transport.requests import Request

    with _lock:
        if _credentials is None:
            _credentials = _sign_in()
        if not _credentials.valid:  # no token yet, or it expires within google-auth's refresh margin
            _credentials.refresh(Request())
        return _credentials


def http_client():
    """openai's default HTTP client, plus a Google token in the Authorization header of every request."""
    import httpx
    import openai

    class GoogleToken(httpx.Auth):
        def auth_flow(self, request):
            request.headers["Authorization"] = "Bearer " + credentials().token
            yield request

    return openai.DefaultHttpxClient(auth=GoogleToken())


def _asked_wait(error) -> float | None:
    """Seconds a 429 asks to wait: Google's RetryInfo retryDelay ("21732s"), else a Retry-After header."""
    found = re.search(r"retryDelay\W+(\d+(?:\.\d+)?)s", f"{getattr(error, 'body', '')} {error}")
    if found:
        return float(found.group(1))
    response = getattr(error, "response", None)
    header = response.headers.get("retry-after", "") if response is not None else ""
    return float(header) if re.fullmatch(r"\d+(\.\d+)?", header.strip()) else None


def retry_wait(error, tries: int) -> float | None:
    """Seconds to wait before the next try, or None to give up, after `tries` failed tries of one call.

    Retried, up to 5 times (6 tries), waiting 15, 30, 60, 120 and 240 s: HTTP 500, 502, 503 (Gemini under
    high demand), 504, network errors and timeouts, and a 429 that asks for no wait or a short one (it gets
    at least the wait it asks for). A 429 asking for more than LONG_429_S (a daily quota asked for about
    6 hours on 2026-11-04) and every other error, a 400 or 401 for example, give up at once.
    """
    import openai

    status = getattr(error, "status_code", None)
    if tries > 5 or not (status in (429, 500, 502, 503, 504) or isinstance(error, openai.APIConnectionError)):
        return None
    wait = 15 * 2 ** (tries - 1)
    if status == 429:
        asked = _asked_wait(error)
        if asked is not None and asked > LONG_429_S:
            return None
        wait = max(wait, asked or 0)
    return wait


def log_usage(response) -> None:
    """One MOLCENSUS_USAGE line per answered call: the model the API says answered, and its token counts.
    biominer_runner.py adds them up per paper in run_info.json."""
    usage = getattr(response, "usage", None)
    print("MOLCENSUS_USAGE", json.dumps({
        "model": getattr(response, "model", None),
        "input": getattr(usage, "prompt_tokens", 0) or 0,
        "output": getattr(usage, "completion_tokens", 0) or 0,
        "total": getattr(usage, "total_tokens", 0) or 0,
    }), flush=True)
