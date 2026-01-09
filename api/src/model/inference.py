from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, Optional, Sequence

import torch
from torch import nn


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.regressor(last_hidden)


@dataclass
class _ModelBundle:
    model: LSTMRegressor
    cfg: Dict[str, Any]
    mean: float
    std: float
    ckpt: Dict[str, Any]
    device: str
    checkpoint_path: str


_MODEL_LOCK = Lock()
_MODEL_BUNDLE: Optional[_ModelBundle] = None


def _load_singleton(checkpoint_path: str, device: Optional[str]) -> _ModelBundle:
    global _MODEL_BUNDLE

    with _MODEL_LOCK:
        if _MODEL_BUNDLE is not None:
            if checkpoint_path != _MODEL_BUNDLE.checkpoint_path:
                raise ValueError(
                    "Checkpoint diferente do carregado no singleton. "
                    "Reinicie a API para usar outro modelo."
                )
            if device is not None and device != _MODEL_BUNDLE.device:
                raise ValueError(
                    "Device diferente do carregado no singleton. "
                    "Reinicie a API para usar outro device."
                )
            return _MODEL_BUNDLE

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        cfg = ckpt["config"]
        mean = float(ckpt["feature_mean"])
        std = float(ckpt["feature_std"])
        model_device = device or cfg.get("device", "cpu")

        model = LSTMRegressor(
            input_size=1,
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        ).to(model_device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        _MODEL_BUNDLE = _ModelBundle(
            model=model,
            cfg=cfg,
            mean=mean,
            std=std,
            ckpt=ckpt,
            device=model_device,
            checkpoint_path=checkpoint_path,
        )
        return _MODEL_BUNDLE


def predict(
    values: Sequence[float],
    checkpoint_path: str,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    bundle = _load_singleton(checkpoint_path, device)

    series = torch.tensor(list(values), dtype=torch.float32).flatten()
    mask = torch.isfinite(series)
    cleaned = series[mask]
    if cleaned.numel() == 0:
        raise ValueError("Serie vazia apos remover valores invalidos.")

    seq_len = int(bundle.cfg["sequence_length"])
    if cleaned.numel() < seq_len:
        raise ValueError("Serie curta para a janela configurada no checkpoint.")

    window = cleaned[-seq_len:].unsqueeze(0).unsqueeze(-1)
    safe_std = bundle.std if bundle.std > 1e-12 else 1.0
    window_norm = (window - bundle.mean) / safe_std

    window_norm = window_norm.to(bundle.device)
    with torch.no_grad():
        pred_norm = bundle.model(window_norm).cpu()

    pred = pred_norm * bundle.std + bundle.mean
    return {
        "prediction": float(pred.squeeze().item()),
        "checkpoint_path": bundle.checkpoint_path,
        "config": bundle.cfg,
    }
