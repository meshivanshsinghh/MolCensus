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
