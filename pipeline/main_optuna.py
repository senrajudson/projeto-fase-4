from datetime import datetime

from config import TrainConfig
from loader import load_module

_optuna = load_module(__file__, "training/4_optuna_hpo.py", "optuna_hpo")


def load_data(cfg: TrainConfig):
    data = load_module(__file__, "data/1_source_yahoo.py", "data")
    return data.download_price_series(cfg.symbol, cfg.start_date, cfg.end_date, cfg.feature)


def clean_data(series):
    integrity = load_module(__file__, "preprocessing/1_integrity.py", "integrity")
    return integrity.clean_series(series)


def run_optuna_study(cfg: TrainConfig, series, n_trials: int, timeout_sec, best_strategy: str) -> None:
    _optuna.run_optuna_with_series(
        cfg=cfg,
        series=series,
        n_trials=n_trials,
        timeout_sec=timeout_sec,
        best_strategy=best_strategy,
    )


def main() -> None:
    cfg = TrainConfig(optuna_output_dir="main_optuna_result")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    cfg.study_name = f"{cfg.study_name}_{timestamp}"

    n_trials = 40
    timeout_sec = None
    best_strategy = "min_mape"

    raw = load_data(cfg)
    cleaned, removed = clean_data(raw)
    if removed > 0:
        print(f"Aviso: removidos {removed} valores invalidos da serie.")

    run_optuna_study(
        cfg,
        series=cleaned,
        n_trials=n_trials,
        timeout_sec=timeout_sec,
        best_strategy=best_strategy,
    )


if __name__ == "__main__":
    main()
