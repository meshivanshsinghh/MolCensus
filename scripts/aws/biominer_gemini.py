"""Point BioMiner at Gemini on Google Cloud Vertex AI, check the sign-in, and send one test call.

Runs on the GPU instance with the BioMiner environment's Python, or on the Mac with the MolCensus venv
(it needs PyYAML, Pillow, openai and google-auth). It reads ~/vertex.env on the instance, copied up from
~/.molcensus/vertex.env on the Mac. That file holds no secret, but it names the Google project, so it
stays out of git:

    VERTEX_PROJECT=...          the Google Cloud project billed for the calls
    VERTEX_SERVICE_ACCOUNT=...  the service account that calls Vertex AI (roles/aiplatform.user)
    VERTEX_WIF_CONFIG='{...}'   the AWS federation config (molcensus_vertex.py explains the sign-in)

It writes BioMiner/config/molcensus_vertex.yaml inside the clone: BioMiner's default config with the
Vertex endpoint, MODEL for text and vision calls, n_jobs 2 and grounding_gpu_num 1. BioMiner handles
n_jobs papers at a time and makes its MLLM calls for one paper one after another, so n_jobs is also the
number of requests in flight; the owner's checker ran 25 calls two at a time on Vertex with no 429. Its
load_prompts() picks the gemini-series-prompt folder for any model name without "gpt" in it. It also
deletes the configs of the old AI Studio route (molcensus_free.yaml, molcensus_paid.yaml), which held
API keys.

--identity  gets a Google token the way BioMiner will (on the instance: through AWS federation), and
            prints whose it is and when it expires. The token itself is never printed.
--check     sends one request through patched BioMiner's own client code (runner/mllm.py, so run
            patch_biominer.py first), shaped exactly like BioMiner's image call: system message, base64
            PNG with detail "high", one of BioMiner's Gemini prompts, temperature 0, response_format
            json_object, max_tokens 4096. It goes through the patched retry rules, so it can wait up to
            about 8 minutes on 503s. It prints the model the API returned, the token counts and whether
            the answer is JSON.

Usage (on the Mac, add VERTEX_AUTH=impersonate in front to sign in with your gcloud login):
    python scripts/aws/biominer_gemini.py --vertex ~/vertex.env --biominer ~/BioMiner --identity --check
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import importlib.util
import io
import json
import os
import sys
from pathlib import Path
from unittest import mock

import yaml

import molcensus_vertex  # next to this script

MODEL = "gemini-3.5-flash"  # sent as google/gemini-3.5-flash by patched BioMiner
N_JOBS = 2
DEFAULT_CONFIG = "BioMiner/config/default_open_source.yaml"
CONFIG = "BioMiner/config/molcensus_vertex.yaml"
OLD_CONFIGS = ["BioMiner/config/molcensus_free.yaml", "BioMiner/config/molcensus_paid.yaml"]
MLLM = "BioMiner/runner/mllm.py"
CHECK_PROMPT = "BioMiner/commons/prompts/gemini-series-prompt/prompt_for_structure_full_updated_v3_with_index.txt"
STUBBED = ["traitlets", "py2opsin", "BioMiner", "BioMiner.commons"]  # imported by mllm.py, unused by its API client


def load_env(path: Path) -> None:
    """NAME=value lines into os.environ (quotes around a value are dropped)."""
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            name, value = line.split("=", 1)
            os.environ[name.strip()] = value.strip().strip("'").strip('"')


def write_config(biominer: Path) -> Path:
    config = yaml.safe_load((biominer / DEFAULT_CONFIG).read_text())
    config["grounding_gpu_num"] = 1  # one GPU: one request at a time to each model server
    config["n_jobs"] = N_JOBS
    config["model"].update(
        base_url=molcensus_vertex.base_url(os.environ["VERTEX_PROJECT"]),
        api_key="unused: patched BioMiner sends a Google token instead",
        text_mllm_type=MODEL, vision_mllm_type=MODEL,
    )
    path = biominer / CONFIG
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"wrote {path} (Vertex AI, model {MODEL}, n_jobs {N_JOBS})")
    for old in OLD_CONFIGS:
        if (biominer / old).exists():
            (biominer / old).unlink()
            print(f"deleted {biominer / old} (old AI Studio route; it held an API key)")
    return path


def identity() -> bool:
    try:
        credentials = molcensus_vertex.credentials()
    except Exception as exc:
        print(f"identity FAILED: {type(exc).__name__}: {exc}")
        return False
    print(f"identity: {type(credentials).__module__}.{type(credentials).__name__} acting as "
          f"{credentials.service_account_email}; token expires {credentials.expiry:%Y-%m-%d %H:%M} UTC (not shown)")
    return True


def patched_client_code(biominer: Path):
    """BioMiner's patched runner/mllm.py, loaded on its own. Only its API client functions are used, so the
    modules it imports for chemistry and parsing are stubbed; that also lets it load outside BioMiner's
    environment, on the Mac."""
    if "MolCensus patch v2" not in (biominer / MLLM).read_text():
        raise SystemExit(f"{biominer / MLLM} is not patched: run scripts/aws/patch_biominer.py {biominer} first")
    stubs = {name: mock.MagicMock() for name in STUBBED if name not in sys.modules}
    sys.modules.update(stubs)
    os.chdir(biominer)  # mllm.py opens its IUPAC cache at a relative path when it is loaded
    try:
        spec = importlib.util.spec_from_file_location("biominer_mllm", biominer / MLLM)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name in stubs:
            sys.modules.pop(name, None)
    return module


class _Tee(io.StringIO):
    def write(self, text):
        sys.__stdout__.write(text)
        return super().write(text)


def check(biominer: Path, config_path: Path) -> bool:
    from PIL import Image

    mllm = patched_client_code(biominer)
    model = yaml.safe_load(config_path.read_text())["model"]
    buffer = io.BytesIO()
    Image.new("RGB", (256, 256), "white").save(buffer, format="png")
    image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    prompt = (biominer / CHECK_PROMPT).read_text().strip().replace("<index></index>", "[]")

    printed = _Tee()
    with contextlib.redirect_stdout(printed):
        client = mllm.get_api_client(model["vision_mllm_type"], model["base_url"], model["api_key"])
        answer = mllm.call_api_iamge(client, prompt, model["vision_mllm_type"], image)
    usage = [json.loads(line.split(" ", 1)[1]) for line in printed.getvalue().splitlines() if line.startswith("MOLCENSUS_USAGE ")]
    if not usage:  # BioMiner returns "[]" for a call it gave up on, which is valid JSON, so look for the usage line
        print("check FAILED: the call was not answered (see the 'try failed' lines above)")
        return False
    try:
        json.loads(answer)
        valid = True
    except ValueError:
        valid = False
    print(f"check: endpoint vertex, model {usage[0]['model']}, tokens input {usage[0]['input']} output "
          f"{usage[0]['output']} total {usage[0]['total']}, valid JSON {valid}: {answer[:200]!r}")
    return valid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vertex", type=Path, required=True, help="vertex.env")
    parser.add_argument("--biominer", type=Path, required=True, help="the BioMiner clone")
    parser.add_argument("--identity", action="store_true", help="get a Google token and say whose it is")
    parser.add_argument("--check", action="store_true", help="send one test call through patched BioMiner")
    args = parser.parse_args()

    load_env(args.vertex.expanduser())
    biominer = args.biominer.expanduser().resolve()
    config_path = write_config(biominer)
    ok = identity() if args.identity else True
    if ok and args.check:
        ok = check(biominer, config_path)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
