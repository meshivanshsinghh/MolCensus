"""scripts/analysis/biominer_facts.py and biominer_scores.py on small made-up BioMiner outputs. Run with: pytest -q"""

import json
import math
import sys
from pathlib import Path

import pytest

pytest.importorskip("rdkit")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
import biominer_facts  # noqa: E402
import biominer_scores  # noqa: E402

LOG = """\
MOLCENSUS_USAGE {"model": "google/gemini-3.5-flash", "input": 1000, "output": 100, "total": 1500}
gemini-3.5-flash 1 try failed, retrying in 15 s: Error code: 503
MOLCENSUS_USAGE {"model": "google/gemini-3.5-flash", "input": 2000, "output": 50, "total": 2050}
gemini-3.5-flash 1 try failed: Error code: 400
"""


def test_call_counts_and_cost():
    counts = biominer_facts.call_counts(LOG)
    assert {k: counts[k] for k in ("answered", "retries", "lost", "input", "output", "total")} == \
        {"answered": 2, "retries": 1, "lost": 1, "input": 3000, "output": 150, "total": 3550}
    # 3,000 input at $1.50 per million, 550 output plus thinking at $9.00 per million
    assert biominer_facts.cost(counts) == pytest.approx((3000 * 1.5 + 550 * 9.0) / 1e6)


def test_stages():
    log = ("[course_readers] done earlier, skipped\n[biominer_smoke] started 04:35:09, estimate 15 to 45 min\n"
           "[biominer_smoke] done after 3 min\n")
    assert biominer_facts.stages(log) == [
        {"stage": "course_readers", "started": "", "minutes": None, "outcome": "done earlier, skipped"},
        {"stage": "biominer_smoke", "started": "04:35:09", "minutes": 3, "outcome": "done"}]


def test_exact_mcnemar_matches_the_earlier_run_summary():
    assert biominer_scores.exact_mcnemar(3, 16) == pytest.approx(0.0044, abs=5e-5)  # docs/aws_run_2026-11-04.md
    assert biominer_scores.exact_mcnemar(0, 0) == 1.0 and biominer_scores.exact_mcnemar(0, 3) == 0.25


def write_paper(folder: Path, name: str) -> None:
    tables = folder / "extraction_gemini-3.5-flash" / name
    tables.mkdir(parents=True)
    (tables / f"{name}_structure_full_output_x.csv").write_text("ligand,smiles\n1a,CCO\n1b,c1ccccc1\n")
    (tables / f"{name}_structure_part_output_x.csv").write_text("ligand,smiles\n2a,CCN\n")
    (tables / f"{name}_structure_merged_output_x_x.csv").write_text("ligand,smiles\n1a,CCO\n1b,c1ccccc1\n2a,CCN\n")
    (tables / f"{name}_structure_merged_output_x_x_clean_new.csv").write_text("ligand,smiles\nz,CCCC\n")  # evaluator copy
    (tables / f"{name}_merge_x_x.csv").write_text(",protein,ligand,type,value,unit,smiles\n0,P,1a,IC50,1,nM,CCO\n")
    (tables / f"{name}_image_merge_full_0_x.csv").write_text("ligand,smiles\nq,CCCl\n")  # per-image copy: not a table read here
    boxes = [[{"bbox": [0, 0, 1, 1], "smiles": "OCC"}, {"smiles": "CC(=O)O"}, {"smiles": None}, {"smiles": "not a molecule"}]]
    (folder / "biominer_upgrade_ocsr_output.json").write_text(json.dumps(boxes))
    (folder / "run_info.json").write_text("{}")


def test_paper_outputs_and_both_rules(tmp_path):
    write_paper(tmp_path / "7", "7")
    outputs = biominer_scores.run_outputs(tmp_path)[7]
    assert outputs["merged"] == ["CCO", "c1ccccc1", "CCN"] and outputs["part"] == ["CCN"]
    assert outputs["bioactivity"] == ["CCO"] and len(outputs["ocsr"]) == 4
    ocsr = biominer_scores.keys_of(outputs["ocsr"], "smiles")
    assert ocsr == {"CCO", "CC(=O)O"}  # OCC is ethanol written another way; None and junk are dropped
    linked = biominer_scores.keys_of(outputs["merged"], "smiles")
    assert len(ocsr - linked) == 1 and len(linked | ocsr) == 4
    assert len(biominer_scores.keys_of(outputs["merged"], "inchikey")) == 3


def test_score_and_paired():
    truth = {1: {"a", "b", "c", "d"}, 2: {"e"}}
    bm = {1: {"a", "b", "x"}, 2: set()}
    reader = {1: {"a", "c"}, 2: {"e", "y"}}
    per = biominer_scores.score(bm, truth, [1, 2], bootstrap_rounds=200)
    pooled = per.iloc[-1]
    assert (pooled["paper_id"], pooled["tp"], pooled["n_gt"], pooled["n_pred"]) == ("ALL", 2, 5, 3)
    assert pooled["recall"] == pytest.approx(0.4) and pooled["precision"] == pytest.approx(2 / 3)
    assert math.isnan(per.iloc[1]["precision"])  # paper 2: no predictions
    assert biominer_scores.paired(bm, reader, truth, [1, 2])[:2] == (1, 2)  # b only by BioMiner; c and e only by the reader
    assert biominer_scores.union(bm, reader) == {1: {"a", "b", "c", "x"}, 2: {"e", "y"}}


def test_biominer_evaluator_lines():
    log = "Average recall_structure_list: 0.5\nAverage precision_structure_list: 0.25\nAverage recall_chiral_structure_list: -1\n"
    assert biominer_scores.biominer_evaluator(log) == {("all", "recall"): 0.5, ("all", "precision"): 0.25}


def test_whole_only_drops_star_pieces_under_both_rules():
    for rule in ("smiles", "inchikey"):
        keys = {1: biominer_scores.keys_of(["CCO", "*c1ccccc1", "[1*]CC"], rule)}
        assert len(keys[1]) == 3 and len(biominer_scores.whole_only(keys)[1]) == 1
