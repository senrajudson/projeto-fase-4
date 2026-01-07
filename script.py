#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import urllib.request
import urllib.error


def parse_float(s: str) -> float:
    s = (s or "").strip()
    if not s:
        raise ValueError("vazio")

    s = s.replace(" ", "")

    # Trata decimal com vírgula e milhares
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")

    v = float(s)
    if not (v == v):  # NaN
        raise ValueError("nan")
    return v


def resolve_csv_path(csv_arg: str) -> Path:
    """
    Resolve o CSV de forma robusta:
    - Se csv_arg existir como caminho, usa.
    - Se não existir, tenta procurar em ./data relativo ao script.
    - Se vier sem extensão, tenta adicionar .csv
    """
    p = Path(csv_arg)

    # 1) caminho direto (relativo ao cwd)
    if p.exists():
        return p.resolve()

    # 2) se não existe e não tem sufixo, tenta .csv
    if p.suffix.lower() != ".csv":
        p2 = Path(str(p) + ".csv")
        if p2.exists():
            return p2.resolve()

    # 3) tenta dentro de "data/" relativo ao script
    script_dir = Path(__file__).resolve().parent
    candidate = (script_dir / "data" / p.name)
    if candidate.exists():
        return candidate.resolve()

    if candidate.suffix.lower() != ".csv":
        candidate2 = candidate.with_suffix(".csv")
        if candidate2.exists():
            return candidate2.resolve()

    raise FileNotFoundError(
        f"CSV não encontrado. Tente passar o caminho completo em --csv.\n"
        f"Recebido: {csv_arg}\n"
        f"Tentei também: {candidate} e {candidate.with_suffix('.csv')}"
    )


def read_last_values(csv_path: Path, column: str, n_last: int) -> list[float]:
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

        fields = {name.lower(): name for name in reader.fieldnames}
        col_key = column.lower()
        if col_key not in fields:
            raise ValueError(f"Coluna '{column}' não encontrada. Disponíveis: {reader.fieldnames}")

        real_col = fields[col_key]

        values: list[float] = []
        for row in reader:
            try:
                values.append(parse_float(row.get(real_col, "")))
            except Exception:
                continue

    if not values:
        raise ValueError("Nenhum valor numérico válido foi lido do CSV.")

    return values[-n_last:] if n_last > 0 else values


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
    ap = argparse.ArgumentParser(description="Testa POST /predict lendo valores de um CSV na pasta data/")
    ap.add_argument("--base-url", default="http://localhost:9010")
    ap.add_argument("--timeout", type=int, default=120)

    ap.add_argument(
        "--csv",
        default="BTC-USD_historico_completo.csv",
        help="arquivo CSV (ex: BTC-USD_historico_completo.csv) ou caminho (ex: data/arquivo.csv)",
    )
    ap.add_argument("--column", default="Close")
    ap.add_argument("--n", type=int, default=200)

    # checkpoint_path precisa existir DENTRO do container
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--device", default="", help="cpu|cuda (opcional)")

    args = ap.parse_args()

    url = args.base_url.rstrip("/") + "/predict"

    csv_path = resolve_csv_path(args.csv)
    values = read_last_values(csv_path, args.column, args.n)

    payload = {"values": values, "checkpoint_path": args.checkpoint}
    if args.device.strip():
        payload["device"] = args.device.strip()

    print(f"POST {url}")
    print(f"CSV: {csv_path} | column: {args.column} | values enviados: {len(values)}")

    try:
        status, body = post_json(url, payload, timeout=args.timeout)
        print(f"HTTP {status}")
        try:
            print(json.dumps(json.loads(body), indent=2, ensure_ascii=False))
        except Exception:
            print(body)
        return 0 if 200 <= status < 300 else 2

    except urllib.error.HTTPError as e:
        print(f"HTTPError: {e.code}")
        try:
            print(e.read().decode('utf-8', errors='replace'))
        except Exception:
            pass
        return 2
    except urllib.error.URLError as e:
        print(f"Falha de conexão: {e.reason}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
