# src.lstm.lstm.py
from typing import Any, Dict
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    config1: str = Field(default="")
    config2: Any = Field(default=None)
    config3: int = Field(default=0)


class LSTMModel:
    def train(self, config: ModelConfig) -> Dict[str, Any]:
        # aqui você faria o treino de verdade
        print("Hello world.")
        # retorne algo serializável (dict/list/str/número)
        return {
            "message": "trained",
            "received": config.model_dump(),
        }
