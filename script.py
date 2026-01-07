#!/usr/bin/env python3
import argparse
import csv
import json
import os
import urllib.request
import urllib.error


def parse_float(s: str) -> float:
    s = (s or "").strip()
    if not s:
        raise ValueError("vazio")

    # remove espaços
    s = s.replace(" ", "")

    # Heurística pra vírgula/ponto:
    # - "12,34" (sem ponto) => decimal com vírgula
    # - "1,234.56" => milhares com vírgula (remove vírgulas)
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")

    return float(s)


def read_last_values(csv_path: str, column: str, n_last: int) -> list[float]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except Exception:
            dialect = csv.excel

        reader = csv.DictReader(f, dialect=dialect)
        if not reader.fieldnames:
            raise ValueError("CSV sem cabeçalho (fieldnames)")

        # acha coluna case-insensitive
        fields = {name.lower(): name for name in reader.fieldnames}
        col_key = column.lower()
        if col_key not in fields:
            raise ValueError(f"Coluna '{column}' não encontrada. Disponíveis: {reader.fieldnames}")

        real_col = fields[col_key]

        values: list[float] = []
        for row in reader:
            try:
                v = parse_float(row.get(real_col, ""))
                values.append(v)
            except Exception:
                # ignora linhas inválidas
                continue

    if not values:
        raise ValueError("Nenhum valor numérico válido foi lido do CSV.")

    if n_last > 0:
        values = values[-n_last:]

    return values


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
    ap = argparse.ArgumentParser(description="Testa POST /predict lendo valores de um CSV")
    ap.add_argument("--base-url", default="http://localhost:9010")
    ap.add_argument("--timeout", type=int, default=60)

    ap.add_argument("--csv", default="data/BTC-USD_historico_completo.csv",
                    help="caminho do CSV (no host)")
    ap.add_argument("--column", default="Close",
                    help="nome da coluna (ex: Close, Open, Adj Close)")
    ap.add_argument("--n", type=int, default=200,
                    help="quantos últimos valores enviar (>= seq_len do checkpoint)")

    # ATENÇÃO: checkpoint_path é o caminho visto PELA API (dentro do container)
    ap.add_argument("--checkpoint", default="checkpoints/best.pt",
                    help="checkpoint_path para a API (ex: checkpoints/best.pt)")
    ap.add_argument("--device", default="", help="cpu|cuda (opcional)")

    args = ap.parse_args()

    url = args.base_url.rstrip("/") + "/predict"

    values = read_last_values(args.csv, args.column, args.n)

    payload = {
        "values": values,
        "checkpoint_path": args.checkpoint,
    }
    if args.device.strip():
        payload["device"] = args.device.strip()

    print(f"POST {url}")
    print(f"CSV: {args.csv} | column: {args.column} | values enviados: {len(values)}")
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
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
