#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import urllib.request
import urllib.error


# =========================
# CONFIG (variáveis globais)
# =========================
BASE_URL_DEFAULT = "http://localhost:9010"
TIMEOUT_DEFAULT = 120

CSV_DEFAULT = "BTC-USD_historico_completo.csv"
COLUMN_DEFAULT = "Close"
N_DEFAULT = 200
START_INDEX_DEFAULT = 3500



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
    p = Path(csv_arg)

    if p.exists():
        return p.resolve()

    if p.suffix.lower() != ".csv":
        p2 = Path(str(p) + ".csv")
        if p2.exists():
            return p2.resolve()

    script_dir = Path(__file__).resolve().parent  # .../api/src

    candidate = script_dir / "data" / p.name
    if candidate.exists():
        return candidate.resolve()
    if candidate.suffix.lower() != ".csv":
        candidate2 = candidate.with_suffix(".csv")
        if candidate2.exists():
            return candidate2.resolve()

    project_root = script_dir.parents[1]  # .../projeto-fase-4
    candidate = project_root / "data" / p.name
    if candidate.exists():
        return candidate.resolve()
    if candidate.suffix.lower() != ".csv":
        candidate2 = candidate.with_suffix(".csv")
        if candidate2.exists():
            return candidate2.resolve()

    raise FileNotFoundError(
        f"CSV não encontrado.\n"
        f"Recebido: {csv_arg}\n"
        f"Tentei também: {script_dir / 'data' / p.name}\n"
        f"E também: {project_root / 'data' / p.name}"
    )


def read_all_values(csv_path: Path, column: str) -> list[float]:
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
    ap = argparse.ArgumentParser(description="Testa POST /predict usando uma janela a partir de um índice do CSV")

    ap.add_argument("--base-url", default=BASE_URL_DEFAULT)
    ap.add_argument("--timeout", type=int, default=TIMEOUT_DEFAULT)

    ap.add_argument("--csv", default=CSV_DEFAULT)
    ap.add_argument("--column", default=COLUMN_DEFAULT)

    ap.add_argument("--start-index", type=int, default=START_INDEX_DEFAULT,
                    help="índice inicial da janela no array de valores")
    ap.add_argument("--n", type=int, default=N_DEFAULT,
                    help="tamanho da janela enviada ao modelo (seq_len)")

    args = ap.parse_args()

    url = args.base_url.rstrip("/") + "/predict"

    csv_path = resolve_csv_path(args.csv)
    all_values = read_all_values(csv_path, args.column)

    start = args.start_index
    n = args.n

    if start < 0:
        raise SystemExit("Erro: --start-index não pode ser negativo.")

    # Precisamos de n valores pra entrada + 1 valor real pra comparar
    end_input = start + n
    target_idx = end_input

    if target_idx >= len(all_values):
        raise SystemExit(
            f"Erro: janela ultrapassa o tamanho da série.\n"
            f"len(values)={len(all_values)}, start={start}, n={n} => target_idx={target_idx} precisa ser < len."
        )

    window = all_values[start:end_input]
    real_next = all_values[target_idx]

    payload = {"values": window}

    print(f"POST {url}")
    print(f"CSV: {csv_path}")
    print(f"Janela: [{start}:{end_input}] (n={n}) | comparando com índice {target_idx}")

    try:
        status, body = post_json(url, payload, timeout=args.timeout)
        print(f"HTTP {status}")

        obj = json.loads(body)
        if not (200 <= status < 300):
            print(json.dumps(obj, indent=2, ensure_ascii=False))
            return 2

        # tenta achar o valor da predição no formato que sua API retorna
        # (ex: {"ok": true, "prediction": ...} ou {"ok": true, "result": {"prediction": ...}})
        pred = None
        if isinstance(obj, dict):
            if "prediction" in obj:
                pred = obj["prediction"]
            elif "result" in obj and isinstance(obj["result"], dict) and "prediction" in obj["result"]:
                pred = obj["result"]["prediction"]

        if pred is None:
            print("Resposta não contém campo 'prediction'. Resposta completa:")
            print(json.dumps(obj, indent=2, ensure_ascii=False))
            return 2

        pred = float(pred)

        abs_err = abs(pred - real_next)
        pct_err = (abs_err / abs(real_next) * 100.0) if real_next != 0 else None

        print("\n--- Resultado ---")
        print(f"prediction: {pred}")
        print(f"real_next (idx {target_idx}): {real_next}")
        print(f"abs_error: {abs_err}")
        if pct_err is not None:
            print(f"pct_error: {pct_err:.4f}%")
        else:
            print("pct_error: n/a (real_next = 0)")

        return 0

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
