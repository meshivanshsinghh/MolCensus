"""Fetch model weights from Hugging Face, or use a copy the user already downloaded."""

from __future__ import annotations

import logging
from pathlib import Path

from .config import resolve_path

log = logging.getLogger(__name__)


def fetch_weights(hf_repo: str, filename: str, models_dir: Path, local_override: str = "") -> Path:
    """Return a local path to the weights file, downloading it into models_dir if needed."""
    if local_override:
        local_path = resolve_path(local_override)
        if local_path is None or not local_path.exists():
            raise FileNotFoundError(f"Configured local weights not found: {local_override}")
        return local_path

    target = models_dir / filename
    if target.exists():
        return target

    from huggingface_hub import hf_hub_download

    models_dir.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s from huggingface.co/%s", filename, hf_repo)
    return Path(hf_hub_download(repo_id=hf_repo, filename=filename, local_dir=str(models_dir)))
