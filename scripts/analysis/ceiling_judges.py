"""Recall ceiling, layer 3: two AI judges, both tested blind on known cases first (docs/recall_ceiling.md).

build_packets() puts all 219 molecules, shuffled within each paper under random IDs, into packets of up to
PACKET_SIZE. A packet holds the paper's figure pages, with the pipeline's crops drawn as labelled red boxes,
one picture of its molecules, and the written instructions (outputs/ceiling/packets/). Both judges get the
same packets. The answer key (outputs/ceiling/judges/key.csv) marks the molecules with a known answer:
found by a reader (A), or a layer 2 bucket (1 -> A, 2 -> B, 3 -> C). Neither judge sees it.

The Gemini judge is gemini-3.5-flash on Vertex AI, one call at a time, signed in like testing_key.py
(your gcloud login impersonates the service account; set VERTEX_PROJECT and VERTEX_SERVICE_ACCOUNT in .env
or the shell). Its answers are cached in outputs/ceiling/judges/gemini/. The Claude judge is Claude Code,
which read the same packet files and wrote its answers to outputs/ceiling/judges/claude/.
"""

from __future__ import annotations

import base64
import json
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd

from ceiling_evidence import CONFIG, OUT, ROOT, load_config

PACKETS = OUT / "packets"
JUDGES = OUT / "judges"
JUDGE_NAMES = ["claude", "gemini"]
MODEL = "gemini-3.5-flash"
PACKET_SIZE = 10
PAGE_LONG_SIDE = 1568      # pixels; the largest image Claude Code views without shrinking it
MIN_ACCURACY = 0.9         # a judge below this on the known cases is not trusted
SEED = 0
ANSWERS = "ABCDE"
ANSWER_BUCKET = {"A": 1, "B": 2, "C": 3, "D": 4}
LAYER2_ANSWER = {1: "A", 2: "B", 3: "C"}
LAYER2_TEST = {1: "2c", 2: "2a", 3: "2b"}

INSTRUCTIONS = """You are checking where molecules appear in a published chemistry paper.

The first {n_pages} images are pages of the paper. A banner at the top of each gives its page number.
Red boxes mark the crops an automatic detector cut out; each box is labelled with its crop name, such as pg005_c03.
The last image shows {n_molecules} molecules, each with an ID such as M123. Each is listed below with the name
the paper's compound list gives it.

For each molecule, test these options in order and choose the first one that is true:
A. The molecule is drawn in full, and that complete drawing lies inside one red box. Give the crop name.
B. The molecule is drawn in full on a page, but no red box holds the complete drawing (there is no box, or the
   box cuts part of it off). Give the page.
C. The molecule is never drawn in full. The paper defines it only through a scaffold with R-groups or variable
   positions plus a table, through a placeholder or letter, through a linker box, or through fragments drawn
   apart. Give the page of the scaffold or table.
D. The molecule is not drawn on any of these pages, in full or through a scaffold.
E. You cannot tell (too small to read, several similar candidates, or unsure).

"Drawn in full" means every atom and bond is shown; standard abbreviations (Me, Et, Ph, OMe, Boc, ...) are
allowed. It must be this exact molecule: the same rings, atoms, substituents and positions. A close analogue
does not count. Ignore wedges and stereochemistry. The paper's label usually sits next to the drawing or in a
table row; use it to find the molecule, then check the structure.

Molecules:
{molecule_lines}

Reply with JSON only, one entry per molecule, and do not write SMILES:
{{"answers": [{{"id": "M123", "answer": "A", "crop": "pg005_c03", "page": 5, "reason": "one short sentence"}}]}}
Use null for crop or page when they do not apply."""


# ---- Packets ------------------------------------------------------------------------------------------

def molecule_line(mol_id: str, name: str, searched: bool, text_pages: str) -> str:
    """One line of the molecule list: its name, and where that name is in the PDF text (layer 1)."""
    if not searched:
        where = "a plain number or generic name, so the PDF text was not searched for it"
    elif text_pages:
        where = f"found in the PDF text on page{'s' if ' ' in text_pages else ''} {', '.join(text_pages.split())}"
    else:
        where = "not found in the PDF text (it may be inside a figure image)"
    return f'{mol_id}: named "{name}" ({where})'


def page_image(source: Path, page: int, n_pages: int, crops: pd.DataFrame, out: Path) -> Path:
    """The page with the pipeline's crops as labelled red boxes and a page banner, at most PAGE_LONG_SIDE px."""
    from PIL import Image, ImageDraw, ImageFont

    if out.exists():
        return out
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(image.size) // 75)
    for crop in crops.itertuples():
        draw.rectangle((crop.x1, crop.y1, crop.x2, crop.y2), outline=(220, 0, 0), width=max(image.size) // 500)
        label = crop.crop_id.split("_", 1)[1]  # p19_pg005_c03 -> pg005_c03
        left, top, right, bottom = draw.textbbox((crop.x1, crop.y1), label, font=font, anchor="lb")
        draw.rectangle((left - 4, top - 4, right + 4, bottom + 2), fill=(220, 0, 0))
        draw.text((crop.x1, crop.y1), label, font=font, fill="white", anchor="lb")
    banner = Image.new("RGB", (image.width, image.height // 30), "black")
    ImageDraw.Draw(banner).text((20, banner.height // 2), f"Page {page} of {n_pages}", font=font, fill="white", anchor="lm")
    page_with_banner = Image.new("RGB", (image.width, image.height + banner.height), "white")
    page_with_banner.paste(banner, (0, 0))
    page_with_banner.paste(image, (0, banner.height))
    page_with_banner.thumbnail((PAGE_LONG_SIDE, PAGE_LONG_SIDE))
    page_with_banner.save(out, quality=90)
    return out


def molecule_sheet(smiles: list[str], ids: list[str], out: Path) -> Path:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D

    options = rdMolDraw2D.MolDrawOptions()
    options.legendFontSize, options.legendFraction = 32, 0.12
    sheet = Draw.MolsToGridImage([Chem.MolFromSmiles(s) for s in smiles], molsPerRow=3, subImgSize=(500, 420),
                                 legends=ids, drawOptions=options)
    sheet.save(out)
    return out


def build_packets(evidence: pd.DataFrame, gt: pd.DataFrame, low_crops: pd.DataFrame) -> pd.DataFrame:
    """Write every packet and the answer key. gt holds each molecule's SMILES, for the drawings only.
    Returns the key. Page images already written are kept."""
    import fitz  # PyMuPDF, only to count each PDF's pages

    paths = load_config(CONFIG)["paths"]
    table = evidence.merge(gt[["paper_id", "gt_match_key", "gt_smiles"]], on=["paper_id", "gt_match_key"], validate="1:1")
    pipeline = pd.read_csv(paths["work_dir"] / "crops_index.csv", dtype={"paper_id": str})
    page_files = pd.read_csv(paths["work_dir"] / "pages_index.csv", dtype={"paper_id": str}).set_index(["paper_id", "page"]).image_path
    low_crops = low_crops.astype({"paper_id": str})
    rng = random.Random(SEED)
    ids = iter(f"M{n:03d}" for n in rng.sample(range(100, 1000), len(table)))
    key = []
    for paper in sorted(table.paper_id.unique(), key=int):
        rows = table[table.paper_id == paper].sample(frac=1, random_state=rng.randrange(2**31))
        with fitz.open(paths["pdf_dir"] / f"{paper}.pdf") as document:
            n_pages = len(document)
        text_hits = {int(p) for pages in rows.text_pages.fillna("") for p in str(pages).split()}
        pages = sorted(set(pipeline[pipeline.paper_id == paper].page) | set(low_crops[low_crops.paper_id == paper].page) | text_hits)
        folder = PACKETS / f"paper_{paper}"
        folder.mkdir(parents=True, exist_ok=True)
        images = [page_image(Path(page_files[paper, p]), p, n_pages, pipeline[(pipeline.paper_id == paper) & (pipeline.page == p)],
                             folder / f"page_{p:03d}.jpg") for p in pages]
        for start in range(0, len(rows), PACKET_SIZE):
            chunk = rows.iloc[start:start + PACKET_SIZE].assign(id=[next(ids) for _ in range(min(PACKET_SIZE, len(rows) - start))])
            packet = f"p{paper}_{start // PACKET_SIZE + 1:02d}"
            sheet = molecule_sheet(list(chunk.gt_smiles), list(chunk.id), folder / f"{packet}_molecules.png")
            lines = [molecule_line(r.id, r.gt_name, bool(r.label_searched), str(r.text_pages) if pd.notna(r.text_pages) else "")
                     for r in chunk.itertuples()]
            prompt = INSTRUCTIONS.format(n_pages=len(images), n_molecules=len(chunk), molecule_lines="\n".join(lines))
            (folder / f"{packet}_prompt.txt").write_text(prompt)
            (folder / f"{packet}_images.txt").write_text("\n".join(str(p.relative_to(ROOT)) for p in images + [sheet]) + "\n")
            for r in chunk.itertuples():
                known = "A" if r.found else LAYER2_ANSWER.get(r.layer2_bucket if pd.notna(r.layer2_bucket) else None, "")
                source = "found" if r.found else (LAYER2_TEST[int(r.layer2_bucket)] if known else "")
                key.append({"id": r.id, "packet": packet, "paper_id": paper, "gt_match_key": r.gt_match_key,
                            "known": known, "known_source": source})
    key = pd.DataFrame(key)
    JUDGES.mkdir(parents=True, exist_ok=True)
    key.to_csv(JUDGES / "key.csv", index=False)
    return key


def packet_list() -> list[str]:
    return sorted(p.name.removesuffix("_prompt.txt") for p in PACKETS.glob("paper_*/*_prompt.txt"))


def packet_files(packet: str) -> tuple[str, list[Path]]:
    folder = PACKETS / f"paper_{packet.split('_')[0][1:]}"
    images = [ROOT / line for line in (folder / f"{packet}_images.txt").read_text().split()]
    return (folder / f"{packet}_prompt.txt").read_text(), images


# ---- Answers ------------------------------------------------------------------------------------------

def parse_answers(text: str, ids: list[str]) -> dict[str, dict]:
    """The judge's JSON reply as {id: {answer, crop, page, reason}}; IDs it skipped are missing."""
    try:
        reply = json.loads(text.strip().removeprefix("```json").removesuffix("```"))
    except (ValueError, AttributeError):
        return {}
    answers = {}
    for entry in reply.get("answers", []) if isinstance(reply, dict) else []:
        letter = str(entry.get("answer", "")).strip().upper()[:1]
        if entry.get("id") in ids and letter in ANSWERS:
            answers[entry["id"]] = {"answer": letter, "crop": entry.get("crop"), "page": entry.get("page"),
                                    "reason": " ".join(str(entry.get("reason", "")).split())}
    return answers


def ask_gemini(packet: str, ids: list[str]) -> dict:
    """One Gemini call for one packet, cached. A reply that is not JSON or misses IDs is asked once more."""
    cache = JUDGES / "gemini" / f"{packet}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    sys.path.insert(0, str(ROOT / "scripts" / "aws"))
    import molcensus_vertex
    import openai

    os.environ.setdefault("VERTEX_AUTH", "impersonate")  # on the Mac: gcloud login impersonates the service account
    missing = [v for v in ("VERTEX_PROJECT", "VERTEX_SERVICE_ACCOUNT") if not os.environ.get(v)]
    if missing:
        raise RuntimeError(f"Set {', '.join(missing)} in .env or the shell (see testing_key.py)")
    client = openai.OpenAI(base_url=molcensus_vertex.base_url(os.environ["VERTEX_PROJECT"]), api_key="vertex",
                           http_client=molcensus_vertex.http_client(), max_retries=0, timeout=900)
    prompt, images = packet_files(packet)
    content = [{"type": "text", "text": prompt}] + [
        {"type": "image_url", "image_url": {"url": f"data:image/{'png' if p.suffix == '.png' else 'jpeg'};base64,"
                                                   + base64.b64encode(p.read_bytes()).decode()}} for p in images]
    record = {"packet": packet, "model": MODEL, "calls": []}
    for _ in range(2):
        tries = 0
        while True:
            try:
                response = client.chat.completions.create(model=molcensus_vertex.MODEL_PREFIX + MODEL,
                                                          messages=[{"role": "user", "content": content}],
                                                          response_format={"type": "json_object"})
                break
            except Exception as error:  # retry_wait() says which errors are worth another try
                tries += 1
                wait = molcensus_vertex.retry_wait(error, tries)
                if wait is None:
                    raise
                time.sleep(wait)
        text = response.choices[0].message.content or ""
        usage = response.usage
        record["calls"].append({"answered_by": response.model, "text": text,
                                "tokens": {"input": usage.prompt_tokens, "output": usage.completion_tokens}})
        record["answers"] = parse_answers(text, ids)
        if len(record["answers"]) == len(ids):
            break
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(record, indent=1))
    return record


def load_answers(key: pd.DataFrame, judge: str) -> pd.DataFrame:
    """One judge's answers for every packet that has an answer file, as columns <judge>, <judge>_reason."""
    rows = []
    for packet, ids in key.groupby("packet").id:
        path = JUDGES / judge / f"{packet}.json"
        if path.exists():
            record = json.loads(path.read_text())
            answers = record["answers"] if "calls" in record else parse_answers(json.dumps(record), list(ids))
            rows += [{"id": i, judge: a["answer"], f"{judge}_reason": a["reason"]} for i, a in answers.items() if i in set(ids)]
    return pd.DataFrame(rows, columns=["id", judge, f"{judge}_reason"])


# ---- Scores -------------------------------------------------------------------------------------------

def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa of two raters over the same items: agreement beyond what chance would give."""
    n = len(a)
    if n == 0:
        return float("nan")
    observed = sum(x == y for x, y in zip(a, b)) / n
    expected = sum((a.count(c) / n) * (b.count(c) / n) for c in set(a) | set(b))
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def accuracy(judged: pd.DataFrame, judge: str) -> dict:
    """Share of the known cases a judge got right; a missing answer counts as wrong."""
    known = judged[judged.known != ""]
    right = known[judge] == known.known
    by_source = {s: (int(right[known.known_source == s].sum()), int((known.known_source == s).sum()))
                 for s in sorted(known.known_source.unique())}
    return {"correct": int(right.sum()), "n": len(known), "accuracy": right.mean() if len(known) else float("nan"),
            "by_source": by_source}


def accepted(claude: str | None, gemini: str | None, trusted: bool) -> int | None:
    """Bucket from layer 3: both judges trusted, both gave the same answer, and that answer is a bucket."""
    if trusted and claude == gemini and claude in ANSWER_BUCKET:
        return ANSWER_BUCKET[claude]
    return None
