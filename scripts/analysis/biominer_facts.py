"""Run facts for a BioMiner run downloaded from S3: stage times, Gemini calls, tokens and cost.

Reads only logs and run_info.json files, never a SMILES table, so its output is safe to paste into docs.

    python scripts/analysis/biominer_facts.py outputs/s3/20261105T043509Z

Thinking tokens are total minus input minus output: the Vertex OpenAI-compatible endpoint counts the
model's thinking in total_tokens but not in completion_tokens. Cost is at list price (PRICE_PER_M).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# US dollars per million tokens, Gemini 3.5 Flash list price (thinking is billed as output).
PRICE_PER_M = {"input": 1.50, "output": 9.00}

STAGE_LINE = re.compile(r"^\[(\w+)\] (started (\d\d:\d\d:\d\d)|done after (\d+) min|.*skipped.*|stopped.*)")

# The stage name in run_all.log -> where that stage's BioMiner papers are.
PAPER_FOLDERS = {
    "biominer_course": "outputs/work/biominer_output",
    "biominer_biovista": "outputs/work_biovista/biominer_output",
}


def usage_lines(text: str) -> list[dict]:
    """The MOLCENSUS_USAGE lines patched BioMiner prints, one per answered Gemini call."""
    return [json.loads(line.split(" ", 1)[1]) for line in text.splitlines() if line.startswith("MOLCENSUS_USAGE ")]


def call_counts(text: str) -> dict:
    """Calls answered, retries, and calls lost after every retry, from one BioMiner log."""
    lines = usage_lines(text)
    return {
        "answered": len(lines),
        "retries": text.count("try failed, retrying in"),
        "lost": text.count(" try failed:"),
        "input": sum(int(u.get("input") or 0) for u in lines),
        "output": sum(int(u.get("output") or 0) for u in lines),
        "total": sum(int(u.get("total") or 0) for u in lines),
        "models": sorted({str(u.get("model")) for u in lines}),
    }


def cost(row: dict) -> float:
    """List-price cost in dollars: input, then output plus thinking (= total - input)."""
    return (row["input"] * PRICE_PER_M["input"] + (row["total"] - row["input"]) * PRICE_PER_M["output"]) / 1e6


def stages(run_all_log: str) -> list[dict]:
    """One row per stage in run_all.log: start time and what happened."""
    rows = {}
    for line in run_all_log.splitlines():
        match = STAGE_LINE.match(line)
        if not match:
            continue
        row = rows.setdefault(match.group(1), {"stage": match.group(1), "started": "", "minutes": None, "outcome": ""})
        if match.group(3):
            row["started"] = match.group(3)
        elif match.group(4):
            row["minutes"], row["outcome"] = int(match.group(4)), "done"
        else:
            row["outcome"] = line.split("] ", 1)[1]
    return list(rows.values())


def paper_rows(folder: Path) -> list[dict]:
    """One row per paper folder: run_info.json plus the counts from its BioMiner log."""
    rows = []
    for info_file in sorted(folder.glob("*/run_info.json"), key=lambda p: int(p.parent.name)):
        info = json.loads(info_file.read_text())
        log_file = info_file.parent / "biominer_stdout.log"
        counts = call_counts(log_file.read_text(errors="replace")) if log_file.exists() else {}
        rows.append({
            "paper": info["paper_id"], "minutes": info["runtime_s"] / 60, "error": info.get("error") or "",
            "answered": info.get("mllm_calls_answered") or 0, "retries": counts.get("retries", 0),
            "lost": info.get("api_failures") or 0, "input": info.get("input_tokens") or 0,
            "output": info.get("output_tokens") or 0, "total": info.get("total_tokens") or 0,
            "models": info.get("api_models") or [], "endpoint": info.get("endpoint"),
            "predictions": info.get("n_predictions", {}),
        })
    return rows


def add_up(rows: list[dict], label) -> dict:
    total = {"paper": label, "minutes": sum(r["minutes"] for r in rows), "error": ""}
    for key in ("answered", "retries", "lost", "input", "output", "total"):
        total[key] = sum(r[key] for r in rows)
    total["models"] = sorted({m for r in rows for m in r["models"]})
    return total


def token_table(rows: list[dict]) -> list[str]:
    lines = ["| Paper | Minutes | Calls answered | Retries | Lost | Input | Output | Thinking | Cost (USD) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        thinking = r["total"] - r["input"] - r["output"]
        bold = "**" if r["paper"] == "Total" else ""
        lines.append(f"| {bold}{r['paper']}{bold} | {r['minutes']:.1f} | {r['answered']} | {r['retries']} | {r['lost']} | "
                     f"{r['input']:,} | {r['output']:,} | {thinking:,} | {cost(r):.3f} |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", type=Path, help="the run folder downloaded from S3")
    run = parser.parse_args().run

    print("## Stages (run_all.log)\n")
    print("| Stage | Started (UTC) | Minutes | Outcome |\n|---|---|---|---|")
    for s in stages((run / "runs/run_all.log").read_text()):
        print(f"| {s['stage']} | {s['started'] or '-'} | {s['minutes'] if s['minutes'] is not None else '-'} | {s['outcome']} |")

    smoke_log = run / "runs/biominer_smoke_output.log"
    everything = []
    if smoke_log.exists():
        smoke = {"paper": "smoke (68_6r8r)", "minutes": float("nan"), "error": "", **call_counts(smoke_log.read_text(errors="replace"))}
        print("\n## Smoke test (BioMiner's example paper)\n")
        print("\n".join(token_table([smoke])).replace("| nan |", "| - |"))
        everything.append(smoke)

    for stage, folder in PAPER_FOLDERS.items():
        rows = paper_rows(run / folder)
        if not rows:
            continue
        print(f"\n## {stage}: per paper (run_info.json)\n")
        print("\n".join(token_table(rows + [add_up(rows, "Total")])))
        errors = [f"paper {r['paper']}: {r['error']}" for r in rows if r["error"]]
        print("\nModels the API reported:", ", ".join(add_up(rows, "")["models"]) or "none",
              "| endpoints:", ", ".join(sorted({str(r["endpoint"]) for r in rows})))
        print("Predictions per paper (biominer / biominer_bioactivity):",
              ", ".join(f"{r['paper']}: {r['predictions'].get('biominer')}/{r['predictions'].get('biominer_bioactivity')}" for r in rows))
        print("Errors:", "; ".join(errors) or "none")
        everything += rows

    if everything:
        grand = add_up([{**r, "minutes": 0.0} for r in everything], "Total")
        print(f"\n**Whole run:** {grand['answered']} calls answered, {grand['retries']} retries, {grand['lost']} lost; "
              f"{grand['input']:,} input, {grand['output']:,} output and {grand['total'] - grand['input'] - grand['output']:,} "
              f"thinking tokens; ${cost(grand):.2f} at list price (${PRICE_PER_M['input']:.2f} per million input, "
              f"${PRICE_PER_M['output']:.2f} per million output and thinking).")


if __name__ == "__main__":
    main()
