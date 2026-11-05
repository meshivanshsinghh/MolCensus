"""Local edits to the BioMiner clone on the GPU instance. Never pushed to BioMiner. Safe to rerun.

1. One GPU: its three model servers hard-code devices=[0, 1, 2, 3]; this sets devices=[0].
2. Vertex AI: copies molcensus_vertex.py (next to this script) to the top of the clone, where BioMiner's
   processes import it (they run there with PYTHONPATH=./). The OpenAI client in runner/mllm.py then
   puts a fresh Google token on every request (the api_key in BioMiner's config is never sent), and
   every call names the model google/<model in the config>, as Vertex wants. The config keeps the plain
   name, so BioMiner's output folders and prompt choice stay as they were.
3. Retries: runner/mllm.py tries each MLLM call once and returns an empty answer "[]" on any error,
   so a rate limit or an overloaded model silently drops that page. The patch retries by the rules in
   molcensus_vertex.retry_wait (503s and short 429s, up to 5 retries waiting 15 to 240 s; a 429 with a
   long retry delay gives up at once). A call that is given up on still prints BioMiner's own
   "<model> <n> try failed:" line, and only then, so the MolCensus wrapper's count in run_info.json
   means "calls lost". A retry prints "try failed, retrying in <n> s:" instead, which is not counted.
4. Client: the OpenAI client gets timeout=300 s and max_retries=0. The SDK's defaults (600 s, plus
   two quick retries of its own) would otherwise sit on top of the schedule above.
5. Usage: each answered call prints one "MOLCENSUS_USAGE {...}" line with the model the API returned
   and the input, output and total token counts. The wrapper adds them up per paper in run_info.json.

Like the MolScribe patch, it stops with an error if the upstream lines are not exactly as expected
(checked against BioMiner commit 17c6161).

Usage: python scripts/aws/patch_biominer.py ~/BioMiner
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

MARKER = "# MolCensus patch v2"

MLLM_EDITS = [  # (old text, new text, times it must appear)
    (
        "from openai import OpenAI\n",
        f"from openai import OpenAI\nimport time  {MARKER}\n"
        f"import molcensus_vertex  {MARKER}: Vertex AI sign-in, retry rules, usage line\n",
        1,
    ),
    (
        "        client = OpenAI(base_url=base_url,\n                        api_key=api_key)\n",
        "        client = OpenAI(base_url=base_url,\n                        api_key=api_key,\n"
        f"                        timeout=300, max_retries=0,  {MARKER}: retries are below\n"
        "                        http_client=molcensus_vertex.http_client())  # a fresh Google token on every request\n",
        1,
    ),
    ("model=mllm, temperature=0", "model=molcensus_vertex.MODEL_PREFIX + mllm, temperature=0", 4),
    ("    while try_times < 1:\n", f"    while try_times < 6:  {MARKER}\n", 2),
    (
        "            print(f'{mllm} {try_times} try failed:', e)\n            try_times += 1\n",
        "            try_times += 1\n"
        f"            wait = molcensus_vertex.retry_wait(e, try_times)  {MARKER}\n"
        "            if wait is None:\n"
        "                print(f'{mllm} {try_times} try failed:', e)\n"
        "                break\n"
        "            print(f'{mllm} {try_times} try failed, retrying in {wait:.0f} s:', e)\n"
        "            time.sleep(wait)\n",
        2,
    ),
    (
        "            output = response.choices[0].message.content\n",
        f"            molcensus_vertex.log_usage(response)  {MARKER}\n"
        "            output = response.choices[0].message.content\n",
        2,
    ),
]


def patch(path: Path, edits: list[tuple[str, str, int]], marker: str) -> None:
    text = path.read_text()
    if marker in text:
        print(f"{path.name}: already patched")
        return
    if "MolCensus patch" in text:
        raise SystemExit(f"{path}: has an older MolCensus patch. Undo it with "
                         f"git -C {path.parent} checkout -- {path.name}, then rerun.")
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
    shutil.copy(Path(__file__).with_name("molcensus_vertex.py"), biominer / "molcensus_vertex.py")
    print("molcensus_vertex.py: copied")
    patch(biominer / "BioMiner" / "runner" / "mllm.py", MLLM_EDITS, MARKER)


if __name__ == "__main__":
    main()
