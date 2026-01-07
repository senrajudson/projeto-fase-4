# src/main.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import anyio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.main_infer import predict_next_from_values  # <- seu código de inferência
from middleware.response_time import ResponseTimeMiddleware

app = FastAPI(title="LSTM model API")
app.add_middleware(ResponseTimeMiddleware)


# ----------------------------
# PREDICT
# ----------------------------
class PredictRequest(BaseModel):
    values: List[float] = Field(..., min_length=1, description="Série de valores (floats)")
    checkpoint_path: str = Field(default="checkpoints/best.pt")
    device: Optional[str] = Field(default=None, description="Ex: 'cpu' ou 'cuda'")


@app.post("/predict")
async def predict_endpoint(req: PredictRequest):
    try:
        ckpt_path = Path(req.checkpoint_path)

        # (opcional) força ficar dentro do projeto, evita caminho arbitrário:
        # ckpt_path = (Path("/app") / req.checkpoint_path).resolve()

        if not ckpt_path.exists():
            raise HTTPException(status_code=404, detail=f"Checkpoint não encontrado: {ckpt_path}")

        pred, ckpt = await anyio.to_thread.run_sync(
            predict_next_from_values,
            req.values,
            str(ckpt_path),
            req.device,
        )

        cfg = ckpt.get("config", {})
        return {
            "ok": True,
            "prediction": pred,
            "checkpoint_path": str(ckpt_path),
            "sequence_length": cfg.get("sequence_length"),
            "symbol": cfg.get("symbol"),
            "feature": cfg.get("feature"),
        }

    except HTTPException:
        raise
    except ValueError as e:
        # erros de validação do seu infer (série curta, vazia, etc.)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import os
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "9010"))
    uvicorn.run("src.main:app", host=host, port=port, reload=True)
