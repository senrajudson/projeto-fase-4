from dataclasses import asdict
from pathlib import Path
from typing import List

import optuna
import torch

from config import TrainConfig
from loader import load_module

_data = load_module(__file__, "../data/1_source_yahoo.py", "data")
_integrity = load_module(__file__, "../preprocessing/1_integrity.py", "integrity")
_norm = load_module(__file__, "../preprocessing/2_normalization.py", "norm")
_train = load_module(__file__, "1_train.py", "train")
_ckpt = load_module(__file__, "3_checkpointing.py", "checkpointing")


def _choose_best_from_pareto(pareto_trials: List[optuna.trial.FrozenTrial], strategy: str = "min_mape"):
    if not pareto_trials:
        raise ValueError("Pareto vazio.")

    if strategy == "min_mae":
        return min(pareto_trials, key=lambda t: t.values[0])
    if strategy == "min_rmse":
        return min(pareto_trials, key=lambda t: t.values[1])
    if strategy == "min_mape":
        return min(pareto_trials, key=lambda t: t.values[2])

    if strategy == "weighted":
        w_mae, w_rmse, w_mape = 1.0, 1.0, 1.0
        return min(pareto_trials, key=lambda t: w_mae * t.values[0] + w_rmse * t.values[1] + w_mape * t.values[2])

    raise ValueError(f"Estrategia desconhecida: {strategy}")


def _export_pareto_csv(study: optuna.Study, path: str) -> None:
    rows = []
    for t in study.best_trials:
        rows.append({
            "trial_number": t.number,
            "MAE": t.values[0],
            "RMSE": t.values[1],
            "MAPE": t.values[2],
            "val_loss_best": t.user_attrs.get("val_loss_best"),
            "checkpoint_path": t.user_attrs.get("checkpoint_path"),
            "params": t.params,
        })

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas e necessario para exportar CSV do pareto.") from exc

    df = pd.DataFrame(rows).sort_values(by="MAPE", ascending=True)
    df.to_csv(path, index=False)


def _ensure_optuna_paths(cfg: TrainConfig) -> None:
    storage_path = cfg.storage_path
    if storage_path.startswith("sqlite:///"):
        sqlite_file = Path(storage_path.replace("sqlite:///", "", 1))
        sqlite_file.parent.mkdir(parents=True, exist_ok=True)

    for path in (cfg.pareto_csv_path, cfg.best_checkpoint_path, cfg.checkpoints_dir):
        Path(path).parent.mkdir(parents=True, exist_ok=True)


def build_objective(cfg: TrainConfig, normalized_series: torch.Tensor, mean: float, std: float):
    def objective(trial: optuna.Trial):
        cfg_local = TrainConfig(**asdict(cfg))

        cfg_local.sequence_length = trial.suggest_int("sequence_length", 20, 120, step=10)  # Janela de entrada.
        cfg_local.hidden_size = 256 # trial.suggest_categorical("hidden_size", [32, 64, 128, 256])  # Unidades LSTM.
        cfg_local.num_layers = trial.suggest_int("num_layers", 2, 4)  # Camadas LSTM.
        cfg_local.dropout = 0.3  # Dropout fixo (teste; antes era trial.suggest_float).
        cfg_local.learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)  # Taxa de aprendizado.
        cfg_local.batch_size = 256 #trial.suggest_categorical("batch_size", [32, 64, 128])  # Tamanho do batch.
        cfg_local.max_epochs = 100  # Epocas fixas (teste; antes era trial.suggest_int).
        cfg_local.patience = 15  # Early stopping fixo (teste; antes era trial.suggest_int).

        metrics_test, ckpt_path, val_loss_best = _train.train_model(
            cfg=cfg_local,
            normalized_series=normalized_series,
            feature_mean=mean,
            feature_std=std,
            trial_number=trial.number,
            trial_params=trial.params,
        )

        trial.set_user_attr("checkpoint_path", ckpt_path)
        trial.set_user_attr("val_loss_best", float(val_loss_best))

        return metrics_test["MAE"], metrics_test["RMSE"], metrics_test["MAPE"]

    return objective


def run_optuna_with_series(
    cfg: TrainConfig,
    series: torch.Tensor,
    n_trials: int = 30,
    timeout_sec=None,
    best_strategy: str = "min_mape",
) -> None:
    _train.set_seed(cfg.seed)

    mean, std = _norm.compute_train_stats(series, cfg.train_ratio)
    normalized = _norm.normalize_with_stats(series, mean, std)

    _ensure_optuna_paths(cfg)

    storage = optuna.storages.RDBStorage(url=cfg.storage_path)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(
        study_name=cfg.study_name,
        storage=storage,
        load_if_exists=True,
        directions=["minimize", "minimize", "minimize"],
        pruner=pruner,
    )
    study.set_metric_names(["MAE", "RMSE", "MAPE"])

    objective = build_objective(cfg, normalized, mean, std)
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec)

    _export_pareto_csv(study, cfg.pareto_csv_path)

    pareto = study.best_trials
    best_trial = _choose_best_from_pareto(pareto, strategy=best_strategy)

    ckpt_path = best_trial.user_attrs.get("checkpoint_path")
    if not ckpt_path:
        raise RuntimeError("checkpoint_path nao encontrado no best_trial.user_attrs")

    ckpt = _ckpt.load_checkpoint(ckpt_path, map_location="cpu")
    torch.save(ckpt, cfg.best_checkpoint_path)


def run_optuna(cfg: TrainConfig, n_trials: int = 30, timeout_sec=None, best_strategy: str = "min_mape") -> None:
    raw = _data.download_price_series(cfg.symbol, cfg.start_date, cfg.end_date, cfg.feature)
    cleaned, _ = _integrity.clean_series(raw)
    run_optuna_with_series(
        cfg=cfg,
        series=cleaned,
        n_trials=n_trials,
        timeout_sec=timeout_sec,
        best_strategy=best_strategy,
    )
