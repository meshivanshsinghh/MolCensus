"""The report page builds from the committed data file, and INCLUDE_COURSE=false leaves out every course result."""

import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "scripts" / "report" / "report_data.json").read_text())


def build(monkeypatch, include_course: str) -> str:
    monkeypatch.setenv("INCLUDE_COURSE", include_course)
    return runpy.run_path(str(ROOT / "scripts" / "report" / "build_site.py"), run_name="test")["page"](DATA)


def test_course_switch(monkeypatch):
    full, public = build(monkeypatch, "true"), build(monkeypatch, "false")
    assert 'id="losses"' in full and 'id="agreement"' in full
    assert 'id="losses"' not in public and 'id="agreement"' not in public and "paper 19" not in public.lower()
    for page in (full, public):
        assert "—" not in page and "<svg" in page and "</html>" in page


def test_data_file_holds_no_smiles_or_names():
    text = json.dumps(DATA)
    assert "smiles\": \"" not in text.lower() and "gt_name" not in text and "pred_smiles" not in text
