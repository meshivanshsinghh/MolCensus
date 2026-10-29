"""MolGlyph (Yan et al., 2026) on a list of crops, as BioMiner runs it. Runs in BioMiner's environment, from its repo folder.

Called by subprocess_reader.py, which describes the input and output files. It loads MolGlyph through
BioMiner's own class (BioMiner/MolScribe/molscribe/interface_molglyph.py) and turns each caption into
SMILES the way BioMiner's run_ocsr_global_batch does: Translator.refactor, then numbered R groups
([1*], [2*], ...) with get_new_smiles. Like BioMiner, it returns no SMILES when the caption has <r> or
<c> parts. MolGlyph reports no confidence.
"""

import argparse
import csv
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("--checkpoint", "--device", "--crops", "--out"):
        parser.add_argument(name, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    from BioMiner.commons.molglyph_utils import Translator, get_new_smiles, get_refactor
    from BioMiner.MolScribe.molscribe import MolGlyph

    def to_smiles(caption: str) -> str:
        smiles = get_refactor(Translator.refactor(caption))
        core, extension = caption.split("<sep>")[0], caption.split("<sep>")[1]
        if "<r>" in extension or "<c>" in extension:
            return ""
        return get_new_smiles(smiles, core, extension) or ""

    model = MolGlyph(args.checkpoint, torch.device(args.device))
    print("molglyph: model loaded", flush=True)

    with open(args.crops, newline="") as handle:
        crops = list(csv.DictReader(handle))
    with open(args.out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["crop_id", "pred_smiles", "confidence", "runtime_s", "error"])
        for start in range(0, len(crops), args.batch_size):
            batch = crops[start:start + args.batch_size]
            started = time.perf_counter()
            try:
                captions = model.predict_image_files([crop["image_path"] for crop in batch])
                failures = [""] * len(batch)
            except Exception:  # one unreadable image fails the batch: redo it one image at a time
                captions, failures = [], []
                for crop in batch:
                    try:
                        captions.append(model.predict_image_file(crop["image_path"]))
                        failures.append("")
                    except Exception as exc:
                        captions.append("")
                        failures.append(f"{type(exc).__name__}: {exc}")
            seconds = round((time.perf_counter() - started) / len(batch), 3)
            for crop, caption, failure in zip(batch, captions, failures):
                smiles, error = "", failure
                if not failure:
                    try:
                        smiles = to_smiles(caption)
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                writer.writerow([crop["crop_id"], smiles, "", seconds, error])
            handle.flush()  # rows survive a crash later in the run
            print(f"molglyph: {min(start + args.batch_size, len(crops))}/{len(crops)} crops", flush=True)


if __name__ == "__main__":
    main()
