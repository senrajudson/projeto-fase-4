# src/main_train.py
from __future__ import annotations

from typing import Any, Dict

from src.config import TrainConfig
from src.loader import load_module

# Carrega módulos (mantém seu padrão atual)
_data = load_module(__file__, "data/1_source_yahoo.py", "data")
_integrity = load_module(__file__, "preprocessing/1_integrity.py", "integrity")
_norm = load_module(__file__, "preprocessing/2_normalization.py", "norm")
_train = load_module(__file__, "training/1_train.py", "train")


def load_data(cfg: TrainConfig):
    return _data.download_price_series(cfg.symbol, cfg.start_date, cfg.end_date, cfg.feature)


def clean_data(series):
    return _integrity.clean_series(series)


def train_and_save(cfg: TrainConfig, series):
    mean, std = _norm.compute_train_stats(series, cfg.train_ratio)
    normalized = _norm.normalize_with_stats(series, mean, std)
    return _train.train_model(
        cfg=cfg,
        normalized_series=normalized,
        feature_mean=mean,
        feature_std=std,
        checkpoint_path=cfg.best_checkpoint_path,
    )


def run_training(cfg: TrainConfig) -> Dict[str, Any]:
    """
    Função que a API vai chamar.
    Retorna um JSON-compatível com métricas e infos do treino.
    """
    _train.set_seed(cfg.seed)

    raw = load_data(cfg)
    cleaned, removed = clean_data(raw)

    metrics, ckpt_path, val_loss_best = train_and_save(cfg, cleaned)

    return {
        "removed_invalid": int(removed),
        "checkpoint_path": str(ckpt_path),
        "val_loss_best": float(val_loss_best),
        "metrics": {
            "MAE": float(metrics["MAE"]),
            "RMSE": float(metrics["RMSE"]),
            "MAPE": float(metrics["MAPE"]),
        },
    }


def main() -> None:
    # Exemplo local (continua funcionando via CLI)
    cfg = TrainConfig(
        symbol="DIS",
        start_date="2018-01-01",
        end_date="2024-07-20",
        feature="Close",
        sequence_length=60,
        batch_size=64,
        learning_rate=1e-3,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        max_epochs=120,
        patience=20,
    )

    result = run_training(cfg)
    print("Treino concluido")
    print(result)


if __name__ == "__main__":
    main()
