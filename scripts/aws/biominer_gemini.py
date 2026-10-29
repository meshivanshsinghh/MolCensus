"""Point BioMiner at Gemini through Gemini's OpenAI-compatible endpoint, and check each key.

Runs on the GPU instance with the BioMiner environment's Python (it needs PyYAML, Pillow and openai,
which that environment has). It reads a keys file that never enters git:

    GEMINI_MODEL=...           model name for both text and vision calls
    GEMINI_API_KEY_FREE=...    free tier: public papers only (BioMiner's examples, BioVista)
    GEMINI_API_KEY_PAID=...    paid tier: the six course papers, only once permission is given

and writes two configs inside the BioMiner clone, next to its default one:

    BioMiner/config/molcensus_free.yaml    n_jobs 1
    BioMiner/config/molcensus_paid.yaml    n_jobs 2

BioMiner handles n_jobs papers at a time and makes its MLLM calls for one paper one after another,
so n_jobs is also the number of requests in flight. Its load_prompts() picks the
gemini-series-prompt folder for any model name without "gpt" in it.

With --check, it sends one request per key shaped exactly like BioMiner's image call (system message,
base64 PNG with detail "high", one of BioMiner's Gemini prompts, temperature 0, response_format
json_object, max_tokens 4096) and prints whether the answer is valid JSON. Run it before any paper.

Usage:
    python scripts/aws/biominer_gemini.py --keys ~/molcensus_keys.env --biominer ~/BioMiner --check
"""

from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import sys
from pathlib import Path

import yaml

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_CONFIG = "BioMiner/config/default_open_source.yaml"
CHECK_PROMPT = "BioMiner/commons/prompts/gemini-series-prompt/prompt_for_structure_full_updated_v3_with_index.txt"
TIERS = {"free": 1, "paid": 2}  # tier -> n_jobs (papers, and so MLLM requests, at a time)


def read_keys(path: Path) -> dict[str, str]:
    keys = {}
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            name, value = line.split("=", 1)
            keys[name.strip()] = value.strip().strip('"').strip("'")
    return keys


def write_configs(keys: dict[str, str], biominer: Path) -> list[Path]:
    base = yaml.safe_load((biominer / DEFAULT_CONFIG).read_text())
    written = []
    for tier, n_jobs in TIERS.items():
        api_key = keys.get(f"GEMINI_API_KEY_{tier.upper()}")
        if not api_key:
            print(f"{tier}: no GEMINI_API_KEY_{tier.upper()} in the keys file; config not written")
            continue
        config = copy.deepcopy(base)
        config["grounding_gpu_num"] = 1  # one GPU: one request at a time to each model server
        config["n_jobs"] = n_jobs
        config["model"].update(
            base_url=BASE_URL, api_key=api_key, text_mllm_type=keys["GEMINI_MODEL"], vision_mllm_type=keys["GEMINI_MODEL"]
        )
        path = biominer / f"BioMiner/config/molcensus_{tier}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        path.chmod(0o600)  # holds an API key
        written.append(path)
        print(f"{tier}: wrote {path} (model {keys['GEMINI_MODEL']}, n_jobs {n_jobs})")
    return written


def check(config_path: Path, prompt_path: Path) -> bool:
    from openai import OpenAI
    from PIL import Image

    model = yaml.safe_load(config_path.read_text())["model"]
    buffer = io.BytesIO()
    Image.new("RGB", (256, 256), "white").save(buffer, format="png")
    image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    prompt = prompt_path.read_text().strip().replace("<index></index>", "[]")

    client = OpenAI(base_url=model["base_url"], api_key=model["api_key"])
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a expert with certian knowledge of bioinfomratics and chemistry."},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}", "detail": "high"}},
                    {"type": "text", "text": prompt},
                ]},
            ],
            model=model["vision_mllm_type"], temperature=0, response_format={"type": "json_object"}, max_tokens=4096,
        )
    except Exception as exc:
        print(f"{config_path.name}: request failed: {type(exc).__name__}: {exc}")
        return False
    choice = response.choices[0]
    content = choice.message.content or ""
    try:
        json.loads(content)
        valid = True
    except ValueError:
        valid = False
    print(f"{config_path.name}: model {response.model}, finish_reason {choice.finish_reason}, "
          f"usage {response.usage}, valid JSON {valid}: {content[:200]!r}")
    return valid and choice.finish_reason == "stop"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--biominer", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="send one test request per key")
    args = parser.parse_args()

    keys = read_keys(args.keys)
    if not keys.get("GEMINI_MODEL"):
        raise SystemExit(f"Set GEMINI_MODEL in {args.keys}")
    written = write_configs(keys, args.biominer)
    if args.check and not all([check(path, args.biominer / CHECK_PROMPT) for path in written]):
        sys.exit(1)


if __name__ == "__main__":
    main()
