"""Local edits to the BioMiner clone on the GPU instance. Never pushed to BioMiner. Safe to rerun.

1. One GPU: its three model servers hard-code devices=[0, 1, 2, 3]; this sets devices=[0].
2. Retries: runner/mllm.py tries each MLLM call once and returns an empty answer "[]" on any error,
   so a rate limit or an overloaded model silently drops that page. The patch retries a call that
   failed with HTTP 429 (rate limit), HTTP 503 (overloaded, which Gemini returns under high demand)
   or a network error or timeout: up to 5 retries after the first call, waiting 15, 30, 60, 120 and
   240 seconds. Any other error (for example a 400) is not retried. Prompts, models and parsing are
   unchanged. A call that is given up on still prints BioMiner's own "<model> <n> try failed:" line,
   and only then, so the MolCensus wrapper's count in run_info.json means "calls lost". A retry
   prints "try failed, retrying:" instead, which the wrapper does not count.
3. Client: the OpenAI client gets timeout=300 s and max_retries=0. The SDK's defaults (600 s, plus
   two quick retries of its own) would otherwise sit on top of the schedule above.

Like the MolScribe patch, it stops with an error if the upstream lines are not exactly as expected
(checked against BioMiner commit 17c6161).

Usage: python scripts/aws/patch_biominer.py ~/BioMiner
"""

from __future__ import annotations

import sys
from pathlib import Path

MARKER = "# MolCensus retry patch"

RETRY_EDITS = [  # (old text, new text, times it must appear)
    ("from openai import OpenAI\n", f"from openai import OpenAI\nimport openai, time  {MARKER}\n", 1),
    (
        "        client = OpenAI(base_url=base_url,\n                        api_key=api_key)\n",
        "        client = OpenAI(base_url=base_url,\n                        api_key=api_key,\n"
        f"                        timeout=300, max_retries=0)  {MARKER}: retries are below\n",
        1,
    ),
    ("    while try_times < 1:\n", f"    while try_times < 6:  {MARKER}\n", 2),
    (
        "            print(f'{mllm} {try_times} try failed:', e)\n            try_times += 1\n",
        "            try_times += 1\n"
        "            transient = getattr(e, 'status_code', None) in (429, 503) or isinstance(e, openai.APIConnectionError)\n"
        f"            if transient and try_times <= 5:  {MARKER}: wait 15, 30, 60, 120, 240 s\n"
        "                print(f'{mllm} {try_times} try failed, retrying:', e)\n"
        "                time.sleep(15 * 2 ** (try_times - 1))\n"
        "            else:\n"
        "                print(f'{mllm} {try_times} try failed:', e)\n"
        "                break\n",
        2,
    ),
]


def patch(path: Path, edits: list[tuple[str, str, int]], marker: str) -> None:
    text = path.read_text()
    if marker in text:
        print(f"{path.name}: already patched")
        return
    for old, new, times in edits:
        if text.count(old) != times:
            raise SystemExit(f"{path}: expected {times} x {old!r}, found {text.count(old)}. BioMiner changed; check the patch.")
        text = text.replace(old, new)
    path.write_text(text)
    print(f"{path.name}: patched")


def main() -> None:
    biominer = Path(sys.argv[1]).expanduser()
    for server in ("mineru_server.py", "moldet_server.py", "ocsr_server.py"):
        patch(biominer / "scripts" / server, [("devices=[0, 1, 2, 3]", "devices=[0]", 1)], "devices=[0]")
    patch(biominer / "BioMiner" / "runner" / "mllm.py", RETRY_EDITS, MARKER)


if __name__ == "__main__":
    main()
