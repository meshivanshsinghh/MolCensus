"""MolSight (Zhang et al., 2026) on a list of crops. Runs in MolSight's own environment, from its repo folder.

Called by subprocess_reader.py, which describes the input and output files. It follows MolSight's
inference.py: the model built from that script's default arguments (EfficientViT encoder, input 512,
formats char,edges, which match the base checkpoint), greedy decoding, then
molsight.chemistry._postprocess_smiles, which expands abbreviations. MolSight's own evaluation does
the same for checkpoints without a coordinate head. MolSight reports no confidence.
"""

import argparse
import csv
import sys
import time

import cv2
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("--checkpoint", "--device", "--crops", "--out"):
        parser.add_argument(name, required=True)
    args = parser.parse_args()

    sys.argv = sys.argv[:1]  # MolSight's own parse_args reads sys.argv; use its defaults
    from inference import parse_args
    from molsight.chemistry import _postprocess_smiles
    from molsight.dataset import get_transforms
    from molsight.model import MolsightModel
    from molsight.tokenizer import CharTokenizer

    model_args = parse_args()
    model_args.formats = model_args.formats.split(",")
    model = MolsightModel(model_args, CharTokenizer(model_args.vocab_file))
    state = torch.load(args.checkpoint, map_location="cpu")["model"]
    missing, unexpected = model.load_state_dict(
        {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}, strict=False
    )
    print(f"molsight: checkpoint loaded, {len(missing)} missing and {len(unexpected)} unexpected keys", flush=True)
    model.to(args.device).eval()
    transform = get_transforms(model_args, augment=False, rotate=False)

    with open(args.crops, newline="") as handle:
        crops = list(csv.DictReader(handle))
    with open(args.out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["crop_id", "pred_smiles", "confidence", "runtime_s", "error"])
        for position, crop in enumerate(crops, start=1):
            started = time.perf_counter()
            smiles, error = "", ""
            try:
                image = cv2.cvtColor(cv2.imread(crop["image_path"]), cv2.COLOR_BGR2RGB)
                tensor = transform(image=image)["image"].unsqueeze(0).to(args.device)
                with torch.no_grad():
                    kv_cache, hooks = model.install_kv_cache_hooks()
                    try:
                        predictions, _ = model.generate(image=tensor, kv_cache=kv_cache)
                    finally:
                        for hook in hooks:
                            hook.remove()
                smiles = _postprocess_smiles(predictions["smiles"][0])[0]
            except Exception as exc:  # one bad crop must not stop the run
                error = f"{type(exc).__name__}: {exc}"
            writer.writerow([crop["crop_id"], smiles, "", round(time.perf_counter() - started, 3), error])
            handle.flush()  # rows survive a crash later in the run
            if position % 100 == 0:
                print(f"molsight: {position}/{len(crops)} crops", flush=True)


if __name__ == "__main__":
    main()
