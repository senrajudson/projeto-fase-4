#!/usr/bin/env python3
import argparse
import json
import sys
import urllib.request
import urllib.error


def post_json(url: str, payload: dict, timeout: int) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        return resp.status, body


def main() -> int:
    ap = argparse.ArgumentParser(description="Testa POST /train da sua API FastAPI")
    ap.add_argument("--base-url", default="http://localhost:9010", help="ex: http://localhost:9010")
    ap.add_argument("--timeout", type=int, default=600, help="timeout em segundos (treino pode demorar)")
    ap.add_argument("--json", dest="json_path", default="", help="caminho de um JSON com o payload")
    ap.add_argument("--epochs", type=int, default=1, help="max_epochs (default 1 pra testar rápido)")
    args = ap.parse_args()

    url = args.base_url.rstrip("/") + "/train"

    if args.json_path:
        with open(args.json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        # payload padrão (ajuste os campos para bater com seu TrainRequest/TrainConfig)
        payload = {
            "symbol": "DIS",
            "start_date": "2018-01-01",
            "end_date": "2024-07-20",
            "feature": "Close",
            "sequence_length": 60,
            "batch_size": 64,
            "learning_rate": 1e-3,
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.2,
            "max_epochs": args.epochs,
            "patience": 5,
            "train_ratio": 0.8,
            "seed": 42,
            "best_checkpoint_path": "checkpoints/best.pt",
        }

    print(f"POST {url}")
    try:
        status, body = post_json(url, payload, timeout=args.timeout)
        print(f"HTTP {status}")
        try:
            obj = json.loads(body)
            print(json.dumps(obj, indent=2, ensure_ascii=False))
        except Exception:
            print(body)
        return 0 if 200 <= status < 300 else 2

    except urllib.error.HTTPError as e:
        print(f"HTTPError: {e.code}")
        try:
            print(e.read().decode("utf-8", errors="replace"))
        except Exception:
            pass
        return 2
    except urllib.error.URLError as e:
        print(f"Falha de conexão: {e.reason}")
        print("Dica: verifique se a API está no ar e a porta está exposta (docker ps / docker logs).")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
