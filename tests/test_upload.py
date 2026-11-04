"""scripts/aws/upload_results.sh, tested with a fake aws command whose bucket is a local folder. Run with: pytest -q"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "aws" / "upload_results.sh"
STAMP = "20261104T010000Z"

# Answers the four calls upload_results.sh makes. s3://<bucket>/<key> is the folder $FAKE_S3/<bucket>/<key>.
# sync applies --exclude like the real one (relative to the source folder, '*' also matches '/').
# FAKE_AWS_LOSE=<folder name>: sync leaves out one file of that folder but exits 0. FAKE_AWS_FAIL=1: sync exits 1.
FAKE_AWS = r'''
import fnmatch, os, pathlib, shutil, sys
args = sys.argv[1:]
s3 = pathlib.Path(os.environ["FAKE_S3"])
def where(url):
    return s3 / url.removeprefix("s3://")
if args[:2] == ["sts", "get-caller-identity"]:
    print("123456789012")
elif args[:2] == ["s3", "sync"]:
    if os.environ.get("FAKE_AWS_FAIL"):
        sys.exit(1)
    source = pathlib.Path(args[2])
    excludes = [args[i + 1] for i, arg in enumerate(args) if arg == "--exclude"]
    files = sorted(p for p in source.rglob("*") if p.is_file()
                   and not any(fnmatch.fnmatchcase(str(p.relative_to(source)), e) for e in excludes))
    for path in files[1:] if source.name == os.environ.get("FAKE_AWS_LOSE") else files:
        target = where(args[3]) / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, target)
elif args[:2] == ["s3", "cp"]:
    target = where(args[3])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(sys.stdin.buffer.read() if args[2] == "-" else pathlib.Path(args[2]).read_bytes())
elif args[:3] == ["s3", "ls", "--recursive"]:
    files = [p for p in where(args[3]).rglob("*") if p.is_file()]
    for path in files:
        print("2026-11-04 01:00:00", path.stat().st_size, path.relative_to(s3))
    sys.exit(0 if files else 1)
else:
    sys.exit(f"fake aws: unexpected call {args}")
'''

# The instance's home folder: what goes up, plus a MinerU copy of a paper and an earlier upload.log, which stay.
HOME_FILES = {
    "molcensus/results/aws/public/public_metrics.csv": "tool,dataset\n",
    "molcensus/results/aws_biovista/predictions/biominer_predictions.csv": "tool\n",
    "molcensus/outputs/work/biominer_output/1/run_info.json": "{}",
    "molcensus/outputs/work/biominer_output/1/mineru/1/auto/1_origin.pdf": "%PDF-1.7",
    "runs/course_readers.done": "",
    "runs/upload.log": "an earlier attempt\n",
    "setup.log": "=== setup.sh finished\n",
}
UPLOADED = {
    "results/aws/public/public_metrics.csv",
    "results/aws_biovista/predictions/biominer_predictions.csv",
    "outputs/work/biominer_output/1/run_info.json",
    "runs/course_readers.done",
    "setup.log",
}


def run_upload(tmp_path: Path, **fake_env: str):
    home, s3, bin_dir = tmp_path / "home", tmp_path / "s3", tmp_path / "bin"
    for name, text in HOME_FILES.items():
        (home / name).parent.mkdir(parents=True, exist_ok=True)
        (home / name).write_text(text)
    script = home / "molcensus" / "scripts" / "aws" / SCRIPT.name
    script.parent.mkdir(parents=True)
    shutil.copy(SCRIPT, script)
    bin_dir.mkdir()
    (bin_dir / "aws").write_text(f"#!{sys.executable}\n{FAKE_AWS}")
    (bin_dir / "aws").chmod(0o755)
    env = {**os.environ, "HOME": str(home), "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "FAKE_S3": str(s3), **fake_env}
    completed = subprocess.run(["bash", str(script), STAMP], env=env, capture_output=True, text=True, timeout=60)
    run_dir = s3 / "molcensus-results-123456789012" / STAMP
    in_bucket = {str(p.relative_to(run_dir)) for p in run_dir.rglob("*") if p.is_file()} if run_dir.exists() else set()
    return completed, in_bucket, (home / "runs" / "upload.log").read_text()


def test_upload_sends_everything_but_pdfs_and_marks_it_complete(tmp_path):
    completed, in_bucket, log = run_upload(tmp_path)
    assert completed.returncode == 0, log
    assert in_bucket == UPLOADED | {"UPLOAD_COMPLETE"}
    assert log.startswith("an earlier attempt\n")  # appended, not overwritten
    assert log.rstrip().endswith(f"UPLOAD_COMPLETE: {len(UPLOADED)} files in s3://molcensus-results-123456789012/{STAMP}/")


@pytest.mark.parametrize("fault, reason", [
    ({"FAKE_AWS_LOSE": "runs"}, f"{len(UPLOADED)} files sent but {len(UPLOADED) - 1} in"),
    ({"FAKE_AWS_FAIL": "1"}, "aws s3 sync results/aws failed (exit 1)"),
])
def test_failed_upload_writes_no_marker_and_logs_why(tmp_path, fault, reason):
    completed, in_bucket, log = run_upload(tmp_path, **fault)
    assert completed.returncode != 0
    assert "UPLOAD_COMPLETE" not in in_bucket
    assert f"UPLOAD FAILED: {reason}" in log.splitlines()[-1]
