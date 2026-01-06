# src.main.py
from fastapi import FastAPI, HTTPException
import anyio

from src.lstm.lstm import ModelConfig, LSTMModel  # ajuste o nome do seu Model conforme existe

app = FastAPI()

model = LSTMModel()  # ou crie com deps/paths etc.

@app.post("/train")
async def train_endpoint(config: ModelConfig):
    """
    Espera JSON no formato do ModelConfig.
    Ex: { "epochs": 10, "batch_size": 32, ... }
    """
    try:
        # evita travar o event loop (treino normalmente é pesado)
        result = await anyio.to_thread.run_sync(model.train, config)
        return {"ok": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    import os

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "9010"))
    uvicorn.run("src.main:app", host=host, port=port, reload=True)
