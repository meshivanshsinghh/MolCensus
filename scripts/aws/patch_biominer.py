"""Local edits to the BioMiner clone on the GPU instance. Never pushed to BioMiner. Safe to rerun.

1. One GPU: its three model servers hard-code devices=[0, 1, 2, 3]; this sets devices=[0].
2. Retries: runner/mllm.py tries each MLLM call once and returns an empty answer "[]" on any error,
   so a rate limit silently drops that page. The patch tries each call up to 4 times, waiting 15, 30
   and 60 seconds between tries. Prompts, models and parsing are unchanged. A call that fails on
   every try still prints BioMiner's own "<model> <n> try failed:" line, and only then, so the
   MolCensus wrapper's count in run_info.json means "calls lost after all retries". A retry prints
   "try failed, retrying:" instead, which the wrapper does not count.

Like the MolScribe patch, it stops with an error if the upstream lines are not exactly as expected
(checked against BioMiner commit 17c6161).

Usage: python scripts/aws/patch_biominer.py ~/BioMiner
"""

from __future__ import annotations

import sys
from pathlib import Path

MARKER = "# MolCensus retry patch"

RETRY_EDITS = [  # (old text, new text, times it must appear)
    ("import sqlite3\n", f"import sqlite3\nimport time  {MARKER}\n", 1),
    ("    while try_times < 1:\n", f"    while try_times < 4:  {MARKER}\n", 2),
    (
        "            print(f'{mllm} {try_times} try failed:', e)\n            try_times += 1\n",
        "            try_times += 1\n"
        f"            if try_times < 4:  {MARKER}: wait 15, 30, 60 s and try again\n"
        "                print(f'{mllm} {try_times} try failed, retrying:', e)\n"
        "                time.sleep(15 * 2 ** (try_times - 1))\n"
        "            else:\n"
        "                print(f'{mllm} {try_times} try failed:', e)\n",
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
