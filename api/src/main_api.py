from __future__ import annotations

import os
import time
from functools import partial
from pathlib import Path
from typing import List, Optional

import anyio
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.model.main_infer import predict_next_from_values


app = FastAPI(title="LSTM model API")


# ----------------------------
# MIDDLEWARE
# ----------------------------
@app.middleware("http")
async def response_time_middleware(request: Request, call_next):
    t0 = time.perf_counter()
    resp = await call_next(request)
    resp.headers["X-Response-Time-ms"] = f"{(time.perf_counter() - t0) * 1000:.2f}"
    return resp


# ----------------------------
# SCHEMAS
# ----------------------------
class PredictRequest(BaseModel):
    values: List[float] = Field(..., min_length=1, description="Série de valores (floats)")
    checkpoint_path: str = Field(default="checkpoints/best.pt")
    device: Optional[str] = Field(default=None, description="Ex: 'cpu' ou 'cuda'")


def ensure_checkpoint(path_str: str) -> Path:
    ckpt_path = Path(path_str)
    if not ckpt_path.exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint não encontrado: {ckpt_path}")
    return ckpt_path


# ----------------------------
# ROUTES
# ----------------------------
@app.post("/predict")
async def predict(req: PredictRequest):
    ckpt_path = ensure_checkpoint(req.checkpoint_path)

    try:
        # roda inferência em thread para não bloquear o event loop
        fn = partial(predict_next_from_values, req.values, str(ckpt_path), req.device)
        pred, ckpt = await anyio.to_thread.run_sync(fn)

        cfg = (ckpt or {}).get("config", {})
        return {
            "ok": True,
            "prediction": pred,
            "checkpoint_path": str(ckpt_path),
            "sequence_length": cfg.get("sequence_length"),
            "symbol": cfg.get("symbol"),
            "feature": cfg.get("feature"),
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------
# ENTRYPOINT
# ----------------------------
def main():
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "9010"))
    reload = os.getenv("API_RELOAD", "true").lower() in ("1", "true", "yes", "y")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
