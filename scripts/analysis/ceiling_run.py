"""Recall-ceiling labels for the 146 course molecules no reader found, in one command (docs/recall_ceiling.md).

    python scripts/analysis/ceiling_run.py              # every step; finished steps are reused from outputs/ceiling/
    python scripts/analysis/ceiling_run.py --no-gemini  # make no new Gemini calls (cached answers are still used)

Steps: layers 1 and 2 (ceiling_evidence.py), judge packets and Gemini (ceiling_judges.py), then the final
labels in results/aws/scored/recall_ceiling.csv and the tables for the doc in outputs/ceiling/tables.md.
Both are gitignored. Run from the repo root with the repo's .venv. Prints counts only, never SMILES.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

import ceiling_evidence as ce
import ceiling_judges as cj

BUCKETS = {
    1: "drawn in full and cropped, but misread (reading loss)",
    2: "drawn in full but not cropped (detection loss)",
    3: "only through an R-group table, placeholder or linker box (table loss)",
    4: "not in the paper's figures",
    5: "unresolved",
}
PAPERS = ["1", "2", "4", "6", "16", "19"]
CSV = ce.AWS / "scored" / "recall_ceiling.csv"


def contested(row, trusted: bool) -> bool:
    """Both trusted judges, independently, give the same answer, and it is not the layer 2 bucket's answer.
    The evidence then conflicts, so the layer 2 label is withdrawn (never replaced by the judges' answer)."""
    return (trusted and pd.notna(row.layer2_bucket) and row.claude == row.gemini and isinstance(row.claude, str)
            and row.claude != cj.LAYER2_ANSWER[int(row.layer2_bucket)])


def label(row, trusted: bool) -> tuple[int, str, str]:
    """(bucket, evidence layer, one-line reason) for one missed molecule."""
    claude, gemini = row.claude if isinstance(row.claude, str) else None, row.gemini if isinstance(row.gemini, str) else None
    if pd.notna(row.layer2_bucket):
        bucket = int(row.layer2_bucket)
        if contested(row, trusted):
            return 5, "none", (f"layer {cj.LAYER2_TEST[bucket]} says bucket {bucket}, but both judges answered {claude} "
                               f"(Claude: {row.claude_reason})")
        reason = {2: f"MolScribe read it exactly in new crop {row.new_crop}",
                  3: f"it is an exact instance of a scaffold or fragment with * atoms: {row.table}",
                  1: f"a pipeline crop reads close to it, and it is that reading's closest ground-truth molecule: {row.close_read}"}
        return bucket, cj.LAYER2_TEST[bucket], reason[bucket]
    bucket = cj.accepted(claude, gemini, trusted)
    if bucket:
        return bucket, "3", f"both judges answered {claude}; Claude: {row.claude_reason}"
    why = ["layer 2 evidence conflicts (" + "; ".join(x for x in (row.new_crop, row.table, row.close_read) if x) + ")"
           if any((row.new_crop, row.table, row.close_read)) else "no layer 2 evidence"]
    if not trusted:
        why.append(f"judges not trusted on the known cases (Claude {claude or '-'}, Gemini {gemini or '-'})")
    elif claude != gemini:
        why.append(f"judges disagree (Claude {claude or 'no answer'}, Gemini {gemini or 'no answer'})")
    else:
        why.append(f"both judges answered {claude or 'nothing'}")
    return 5, "none", "; ".join(why)


def markdown(frame: pd.DataFrame) -> str:
    cells = [[str(c) for c in frame.columns]] + [[str(v) for v in row] for row in frame.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(cells[0]) + " |", "|" + "---|" * len(cells[0])] + ["| " + " | ".join(r) + " |" for r in cells[1:]])


def per_paper(final: pd.DataFrame, column: str, values: list) -> pd.DataFrame:
    table = pd.crosstab(final.paper_id, final[column]).reindex(index=PAPERS, columns=values, fill_value=0)
    table.insert(0, "missed", table.sum(axis=1))
    table.loc["total"] = table.sum()
    return table.reset_index().rename(columns={"paper_id": "paper"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-gemini", action="store_true", help="make no new Gemini calls")
    args = parser.parse_args()

    config = ce.load_config(ce.CONFIG)
    featurizer = ce.Featurizer(config["scoring"]["fingerprint_radius"], config["scoring"]["fingerprint_bits"])
    evidence = ce.main().astype({"paper_id": str})
    for column in ("new_crop", "table", "close_read"):
        evidence[column] = evidence[column].fillna("")
    key = cj.build_packets(evidence, ce.molecules(featurizer), ce.low_threshold_crops(config))
    packets = cj.packet_list()
    print(f"Layer 3: {len(key)} molecules in {len(packets)} packets; {int((key.known != '').sum())} with a known answer")

    if not args.no_gemini:
        for n, packet in enumerate(packets, start=1):  # one call at a time
            cj.ask_gemini(packet, list(key[key.packet == packet].id))
            print(f"  Gemini: {n}/{len(packets)} packets answered", flush=True)

    judged = key.merge(evidence, on=["paper_id", "gt_match_key"], validate="1:1")
    for judge in cj.JUDGE_NAMES:
        judged = judged.merge(cj.load_answers(key, judge), on="id", how="left")
    scores = {judge: cj.accuracy(judged, judge) for judge in cj.JUDGE_NAMES}
    trusted = all(s["accuracy"] >= cj.MIN_ACCURACY for s in scores.values())

    missed = judged[~judged.found].copy()
    labels = [label(row, trusted) for row in missed.itertuples()]
    missed["bucket"], missed["evidence_layer"], missed["reason"] = zip(*labels)
    missed["bucket_name"] = missed.bucket.map(BUCKETS)
    final = missed.sort_values(["paper_id", "gt_name"], key=lambda s: s.map(int) if s.name == "paper_id" else s)
    final[["paper_id", "gt_name", "bucket", "bucket_name", "evidence_layer", "reason", "layer2_bucket", "claude", "gemini"]].to_csv(CSV, index=False)

    # Tables for docs/recall_ceiling.md: counts, accuracies and agreement only.
    both = judged.dropna(subset=cj.JUDGE_NAMES)
    layer3 = both[~both.found & both.layer2_bucket.isna()]
    lines = ["## By bucket", markdown(per_paper(final, "bucket", [1, 2, 3, 4, 5])),
             "## By evidence", markdown(per_paper(final, "evidence_layer", ["2a", "2b", "2c", "3", "none"])),
             "## Judges on the known cases"]
    for judge, s in scores.items():
        parts = ", ".join(f"{src} {c}/{n}" for src, (c, n) in s["by_source"].items())
        lines.append(f"- {judge}: {s['correct']}/{s['n']} = {s['accuracy']:.3f} ({parts})")
    lines.append(f"- trusted (both >= {cj.MIN_ACCURACY}): {trusted}")
    known = judged[judged.known != ""]
    disputed = known[(known.claude == known.gemini) & (known.claude != known.known)]
    lines.append(f"- known cases where both judges give the same other answer: {len(disputed)} "
                 f"({disputed.known_source.value_counts().to_dict()}); without them: "
                 + ", ".join(f"{j} {cj.accuracy(judged.drop(disputed.index), j)['correct']}/{len(known) - len(disputed)}"
                             for j in cj.JUDGE_NAMES))
    tokens = [c["tokens"] for p in packets for c in json.loads((cj.JUDGES / "gemini" / f"{p}.json").read_text())["calls"]
              if (cj.JUDGES / "gemini" / f"{p}.json").exists()]
    lines.append(f"- Gemini: {len(tokens)} calls, {sum(t['input'] for t in tokens)} input and {sum(t['output'] for t in tokens)} output tokens")
    for judge in cj.JUDGE_NAMES:
        known = judged[judged.known != ""]
        lines += [f"### {judge}: known answer (rows) by its answer (columns)",
                  markdown(pd.crosstab(known.known, known[judge].fillna("none")).reset_index())]
    lines += ["## Agreement",
              f"- all molecules both judged: {len(both)}, same answer {int((both.claude == both.gemini).sum())}, "
              f"kappa {cj.cohen_kappa(list(both.claude), list(both.gemini)):.3f}",
              f"- known cases: kappa {cj.cohen_kappa(list(both[both.known != ''].claude), list(both[both.known != ''].gemini)):.3f}",
              f"- layer 3 molecules: {len(layer3)}, same answer {int((layer3.claude == layer3.gemini).sum())}, "
              f"kappa {cj.cohen_kappa(list(layer3.claude), list(layer3.gemini)):.3f}",
              markdown(pd.crosstab(both.claude, both.gemini).reset_index().rename(columns={"claude": "Claude \\ Gemini"}))]
    t = missed.best_close_tanimoto
    lines += ["## Layer 1 and 2 details",
              f"- label kinds of the missed molecules: {missed.label_kind.value_counts().to_dict()}",
              f"- label searched in the PDF text: {int(missed.label_searched.sum())}, found there: {int((missed.text_pages.fillna('') != '').sum())}",
              f"- table search: {missed.table_status.value_counts().to_dict()}",
              f"- best eligible close reading: >= 0.80: {int((t >= 0.8).sum())}, [0.75, 0.80): {int(((t >= 0.75) & (t < 0.8)).sum())}, "
              f"[0.80, 0.85): {int(((t >= 0.8) & (t < 0.85)).sum())}, [0.70, 0.90): {int(((t >= 0.7) & (t < 0.9)).sum())}",
              f"- layer 2 conflicts: {int((missed.layer2_bucket.isna() & ((missed.new_crop != '') | (missed.table != '') | (missed.close_read != ''))).sum())}",
              f"- layer 2 labels withdrawn because both judges contradict them: "
              f"{missed[[contested(r, trusted) for r in missed.itertuples()]].layer2_bucket.value_counts().to_dict()}",
              "- strict rule (no withdrawal), by bucket: "
              f"{missed.layer2_bucket.fillna(missed.bucket.where(missed.evidence_layer == '3')).fillna(5).astype(int).value_counts().sort_index().to_dict()}"]
    (ce.OUT / "tables.md").write_text("\n\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Wrote {CSV} and {ce.OUT / 'tables.md'}")


if __name__ == "__main__":
    main()
