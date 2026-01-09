from pathlib import Path
from typing import Any, Dict, Optional

import torch


def trial_ckpt_path(checkpoints_dir: str, trial_number: int) -> str:
    path = Path(checkpoints_dir) / f"trial_{trial_number}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def save_checkpoint(path: str, checkpoint: Dict[str, Any]) -> None:
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, ckpt_path)


def load_checkpoint(path: str, map_location: Optional[str] = None) -> Dict[str, Any]:
    if map_location is None:
        return torch.load(path)
    return torch.load(path, map_location=map_location)
