"""Runners for tools that run as subprocesses (BioMiner, crop readers), tested with fake tools. Run with: pytest -q"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from molcensus.recognizers.biominer_runner import run_biominer  # noqa: E402
from molcensus.recognizers.subprocess_reader import run_crop_reader  # noqa: E402

# Writes the tables BioMiner 17c6161 writes (runner/extractor.py), plus look-alikes that must be ignored.
# Paper 2: the final merge fails after the structure tables exist. Paper 3: the structure table is empty.
FAKE_BIOMINER = r'''
import sys, pathlib
args = dict(a.lstrip("-").split("=", 1) for a in sys.argv[1:])
name = pathlib.Path(args["pdf"]).name.split(".")[0]
out = pathlib.Path(args["output_dir"]) / "extraction_gemini-2.5-flash" / name
out.mkdir(parents=True)
pathlib.Path(args["output_dir"]).parent.joinpath("calls.log").open("a").write(name + "\n")
if name == "1":  # two lost MLLM calls and two answered ones, as patched BioMiner logs them
    print("gemini-x 0 try failed: Error code: 429\ngemini-x 0 try failed: Error code: 429")
    print('MOLCENSUS_USAGE {"model": "google/gemini-x-001", "input": 1000, "output": 200, "total": 1300}')
    print('MOLCENSUS_USAGE {"model": "google/gemini-x-001", "input": 500, "output": 50, "total": 600}')
s = "moldetv2_bbox_molglyph_ocsr__updated_v3_with_index_split4seg0.125_bg_lo"
(out / f"{name}_structure_full_output_{s}.csv").write_text("ligand,smiles\n9z,CCCC\n")
(out / f"{name}_structure_merged_output_{s}_{s}_clean_new.csv").write_text(",ligand,smiles\n0,x,CCCl\n")
(out / f"{name}_page_2_{s}.csv").write_text("ligand,smiles\ny,CCBr\n")
(out / f"{name}_merged_updated_v3_txtmu_updated_v3_imgmuc_output.csv").write_text("protein,ligand,type,value,unit\n")
merged = "" if name == "3" else "ligand,smiles\n4x,CCO\n4x,CCO\n4y,c1ccccc1\n10,\n"
(out / f"{name}_structure_merged_output_{s}_{s}.csv").write_text(merged)
if name == "2":
    print("ValueError: cannot map identifiers", file=sys.stderr)
    sys.exit(1)
(out / f"{name}_merge_{s}_{s}.csv").write_text(
    ",protein,ligand,type,value,unit,smiles\n0,EGFR,4x,IC50,5,nM,CCO\n1,EGFR,4x,Ki,2,nM,CCO\n")
(out / f"{name}_merge_failed_ligand_{s}_{s}.json").write_text("[]")
'''


def test_biominer_reads_structure_and_bioactivity_tables(tmp_path):
    repo = tmp_path / "BioMiner"
    repo.mkdir()
    (repo / "example_open_source.py").write_text(FAKE_BIOMINER)
    (repo / "c.yaml").write_text("model:\n  base_url: https://aiplatform.googleapis.com/v1/projects/p/locations/global/endpoints/openapi\n"
                                 "  text_mllm_type: gemini-x\n  vision_mllm_type: gemini-x\n")
    pdfs = {}
    for paper in (1, 2, 3):
        pdfs[paper] = tmp_path / f"{paper}.pdf"
        pdfs[paper].write_bytes(b"%PDF-1.4")
    settings = {"repo_dir": str(repo), "python_executable": sys.executable, "config_path": "c.yaml", "timeout_s": 60}

    outputs = run_biominer(pdfs, settings, tmp_path / "out")
    structure, bioactivity = outputs["biominer"], outputs["biominer_bioactivity"]

    def pairs(frame, paper):
        rows = frame[frame["paper_id"] == paper]
        return sorted(zip(rows["pred_label"], rows["pred_smiles"]))

    # Only the structure_merged table counts, deduplicated; empty SMILES and look-alike files are ignored.
    assert pairs(structure, 1) == [("4x", "CCO"), ("4y", "c1ccccc1")]
    assert (structure.loc[structure["paper_id"] == 1, "error"] == "").all()
    assert pairs(bioactivity, 1) == [("4x", "CCO")]  # two measurements, one prediction

    # Failed final merge: structure rows are kept and say so; the bioactivity table is an error row.
    assert pairs(structure, 2) == [("4x", "CCO"), ("4y", "c1ccccc1")]
    assert structure.loc[structure["paper_id"] == 2, "error"].str.startswith("partial run: exit code 1").all()
    paper2 = bioactivity[bioactivity["paper_id"] == 2]
    assert len(paper2) == 1 and paper2["pred_smiles"].iloc[0] == "" and "exit code 1" in paper2["error"].iloc[0]

    # An unreadable table is skipped without stopping the run.
    paper3 = structure[structure["paper_id"] == 3]
    assert len(paper3) == 1 and paper3["error"].iloc[0] == "no readable output table"
    assert pairs(bioactivity, 3) == [("4x", "CCO")]

    # Every paper records the model names, the endpoint (not its URL, which names the project), the
    # models the API says answered, and how many MLLM calls failed.
    info = json.loads((tmp_path / "out" / "1" / "run_info.json").read_text())
    assert info["text_model"] == "gemini-x" and info["vision_model"] == "gemini-x" and info["endpoint"] == "vertex"
    assert info["api_models"] == ["google/gemini-x-001"] and "projects/p" not in json.dumps(info)
    assert info["api_failures"] == 2 and info["n_predictions"] == {"biominer": 2, "biominer_bioactivity": 1}
    assert info["input_tokens"] == 1500 and info["output_tokens"] == 250 and info["total_tokens"] == 1900
    assert info["mllm_calls_answered"] == 2
    paper2 = json.loads((tmp_path / "out" / "2" / "run_info.json").read_text())
    assert paper2["api_failures"] == 0 and paper2["input_tokens"] is None and paper2["api_models"] is None  # unknown, not zero

    # A rerun reuses finished papers instead of calling BioMiner again; force reruns them.
    again = run_biominer(pdfs, settings, tmp_path / "out")
    assert (tmp_path / "out" / "calls.log").read_text().split() == ["1", "2", "3"]
    assert again["biominer"].drop(columns="runtime_s").equals(structure.drop(columns="runtime_s"))
    run_biominer({1: pdfs[1]}, settings, tmp_path / "out", force=True)
    assert (tmp_path / "out" / "calls.log").read_text().split() == ["1", "2", "3", "1"]


# Paper 2 hangs, as if BioMiner were stuck in retries when the stage time limit sends SIGTERM.
SLOW_BIOMINER = r'''
import sys, pathlib, time
args = dict(a.lstrip("-").split("=", 1) for a in sys.argv[1:])
name = pathlib.Path(args["pdf"]).name.split(".")[0]
pathlib.Path(args["output_dir"]).parent.joinpath("calls.log").open("a").write(name + "\n")
out = pathlib.Path(args["output_dir"]) / "extraction_gemini-x" / name
out.mkdir(parents=True)
if name == "2" and pathlib.Path(args["output_dir"]).parent.joinpath("hang").exists():
    time.sleep(60)
(out / f"{name}_structure_merged_output_a_b.csv").write_text("ligand,smiles\n1,CCO\n")
'''


def test_biominer_stops_on_sigterm_and_resumes(tmp_path):
    import os
    import signal
    import threading
    import time

    repo = tmp_path / "BioMiner"
    repo.mkdir()
    (repo / "example_open_source.py").write_text(SLOW_BIOMINER)
    pdfs = {p: tmp_path / f"{p}.pdf" for p in (1, 2, 3)}
    for path in pdfs.values():
        path.write_bytes(b"%PDF-1.4")
    settings = {"repo_dir": str(repo), "python_executable": sys.executable, "config_path": "none.yaml", "timeout_s": 120}
    out = tmp_path / "out"
    out.mkdir()
    (out / "hang").touch()

    threading.Timer(3, os.kill, (os.getpid(), signal.SIGTERM)).start()  # the stage time limit
    started = time.monotonic()
    first = run_biominer(pdfs, settings, out)
    assert time.monotonic() - started < 30  # the hanging paper was stopped, not waited for
    assert sorted(first["biominer"]["paper_id"]) == [1]  # the finished paper is kept
    assert (out / "1" / "run_info.json").exists() and not (out / "2" / "run_info.json").exists()

    (out / "hang").unlink()
    second = run_biominer(pdfs, settings, out)
    assert sorted(second["biominer"]["paper_id"]) == [1, 2, 3]
    assert (out / "calls.log").read_text().split() == ["1", "2", "2", "3"]  # paper 1 was not run again


# Writes one good row and one per-crop error, then crashes before the third crop.
FAKE_WORKER = r'''
import argparse, csv, sys
parser = argparse.ArgumentParser()
for name in ("--checkpoint", "--device", "--crops", "--out"):
    parser.add_argument(name)
args = parser.parse_args()
crops = list(csv.DictReader(open(args.crops)))
with open(args.out, "w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["crop_id", "pred_smiles", "confidence", "runtime_s", "error"])
    writer.writerow([crops[0]["crop_id"], "CCO", "", 0.1, ""])
    writer.writerow([crops[1]["crop_id"], "", "", 0.1, "OSError: unreadable image"])
sys.exit(3)
'''


def test_crop_reader_keeps_rows_written_before_a_crash(tmp_path):
    worker = tmp_path / "fake_worker.py"
    worker.write_text(FAKE_WORKER)
    crops = pd.DataFrame({"crop_id": ["a", "b", "c"], "paper_id": [1, 1, 2], "page": [3, 3, 1],
                          "image_path": ["a.png", "b.png", "c.png"]})
    settings = {"repo_dir": str(tmp_path), "python_executable": sys.executable, "device": "cpu"}

    predictions = run_crop_reader("fake", crops, settings, tmp_path / "weights.pt", worker=worker).set_index("crop_id")

    assert predictions.loc["a", "pred_smiles"] == "CCO" and predictions.loc["a", "error"] == ""
    assert predictions.loc["b", "pred_smiles"] == "" and predictions.loc["b", "error"] == "OSError: unreadable image"
    assert predictions.loc["c", "pred_smiles"] == "" and predictions.loc["c", "error"] == "fake worker exit code 3"
    assert (predictions["tool"] == "fake").all() and predictions.loc["c", "paper_id"] == 2


def test_smoke_gate_fails_when_more_than_two_calls_are_lost(tmp_path):
    import shutil
    import subprocess

    import pytest

    if shutil.which("bash") is None:
        pytest.skip("needs bash")
    script = (Path(__file__).resolve().parents[1] / "scripts" / "aws" / "run_all.sh").read_text()
    start = script.index("smoke_check() {")
    function = script[start:script.index("\n}\n", start) + 3]  # the real function, not a copy
    out = tmp_path / "biominer_smoke" / "extraction_gemini-3.5-flash" / "68_6r8r"
    out.mkdir(parents=True)
    (out / "68_6r8r_structure_merged_output_a_b.csv").write_text("ligand,smiles\n1,CCO\n")

    def gate(lost_calls):
        log = tmp_path / "smoke_output.log"
        log.write_text("".join(f"gemini-3.5-flash {i} try failed: Error code: 403\n" for i in range(lost_calls)) + "done\n")
        return subprocess.run(["bash", "-c", f'{function}\nsmoke_check "$1" "$2"', "gate",
                               str(tmp_path / "biominer_smoke"), str(log)], capture_output=True, text=True)

    passed, failed = gate(0), gate(3)
    assert passed.returncode == 0 and "passed" in passed.stdout
    assert failed.returncode == 1 and "FAILED: 3 MLLM calls lost" in failed.stdout
