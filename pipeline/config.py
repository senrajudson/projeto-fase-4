from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class TrainConfig:
    symbol: str = "BTC-USD"
    #start_date: str = "2018-01-01"
    start_date: str = "2021-01-01"
    end_date: str = "2024-07-20"
    feature: str = "Close"

    train_ratio: float = 0.85
    val_ratio_within_train: float = 0.15

    sequence_length: int = 60
    batch_size: int = 64
    learning_rate: float = 1e-3
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2

    max_epochs: int = 120
    patience: int = 20

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    study_name: str = "stock_lstm_optuna"
    optuna_output_dir: str = "artifacts/optuna"
    storage_path: str = ""
    pareto_csv_path: str = ""
    best_checkpoint_path: str = ""
    checkpoints_dir: str = ""

    def __post_init__(self) -> None:
        output_dir = Path(self.optuna_output_dir)
        if not self.storage_path:
            self.storage_path = f"sqlite:///{output_dir / 'optuna_study.db'}"
        if not self.pareto_csv_path:
            self.pareto_csv_path = str(output_dir / "pareto_trials.csv")
        if not self.best_checkpoint_path:
            self.best_checkpoint_path = str(output_dir / "best_pareto_lstm.pt")
        if not self.checkpoints_dir:
            self.checkpoints_dir = str(output_dir / "checkpoints")
