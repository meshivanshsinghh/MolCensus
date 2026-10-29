"""Recognizer wrappers. Every tool writes predictions with the same columns so one scorer handles all of them."""

PREDICTION_COLUMNS = [
    "tool",
    "crop_id",       # which crop the prediction came from (or a generated ID for whole-PDF tools)
    "paper_id",
    "page",          # empty when the tool does not report a page
    "pred_smiles",   # predicted SMILES, empty if the tool returned nothing
    "pred_label",    # compound label in the paper (e.g. "9c") when the tool links one
    "confidence",    # tool-reported confidence, empty if the tool has none
    "runtime_s",
    "error",         # error message if this item failed
]

# Readers that take one molecule crop at a time. MolScribe runs in this environment; the others run
# in their own environment through subprocess_reader.py and a <tool>_worker.py script.
CROP_TOOLS = ("molscribe", "molsight", "molglyph")


def run_crop_tool(tool: str, crops, settings: dict, models_dir):
    """Run one crop reader on a crops table (crop_id, paper_id, page, image_path)."""
    from ..weights import fetch_weights

    checkpoint = fetch_weights(
        settings["hf_repo"], settings["checkpoint_file"], models_dir, settings.get("local_checkpoint", "")
    )
    if tool == "molscribe":
        from .molscribe_runner import run_molscribe

        return run_molscribe(crops, checkpoint, settings["device"])
    from .subprocess_reader import run_crop_reader

    return run_crop_reader(tool, crops, settings, checkpoint)
