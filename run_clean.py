#!/usr/bin/env python3
"""
run_clean.py — Executa FASE 1 (CLEAN) em paralelo para todos os arquivos raw.

Uso:
    python run_clean.py [--workers N] [--markets BTC15m ETH15m SOL15m XRP15m]

Varredura: data_raw/<MARKET>/*.jsonl
Saída:     data_cache/lean/YYYY-MM-DD.MARKET.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Garante import do src/
sys.path.insert(0, str(Path(__file__).parent))

from src.py.config import (
    DATA_RAW_DIR, LEAN_DIR, META_JSON, MARKET_MAP, get_clean_workers,
)
from src.py.clean import clean_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Padrão de nome de arquivo: pode ser qualquer nome terminado em .jsonl
_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _out_path(raw: Path) -> Path:
    """Deriva caminho de saída Parquet a partir do path raw."""
    date_match = _DATE_RE.search(raw.stem)
    date_str   = date_match.group(1) if date_match else raw.stem
    market_dir = raw.parent.name  # "BTC15m", "ETH15m", ...
    return Path(LEAN_DIR) / f"{date_str}.{market_dir}.parquet"


def _task(args: tuple[str, str]) -> dict:
    raw_path, out_path = args
    try:
        return clean_file(raw_path, out_path)
    except Exception as exc:
        return {"file": raw_path, "error": str(exc), "kept": 0, "dropped": 0}


def main(markets: list[str] | None = None, workers: int = 0) -> None:
    os.makedirs(LEAN_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(META_JSON), exist_ok=True)

    markets = markets or list(MARKET_MAP.keys())
    workers = workers or get_clean_workers()

    tasks: list[tuple[str, str]] = []
    for market in markets:
        raw_dir = Path(DATA_RAW_DIR) / market
        if not raw_dir.exists():
            log.warning("Diretório não encontrado: %s", raw_dir)
            continue
        for raw in sorted(raw_dir.glob("*.jsonl")):
            out = _out_path(raw)
            tasks.append((str(raw), str(out)))

    if not tasks:
        log.error("Nenhum arquivo JSONL encontrado em %s", DATA_RAW_DIR)
        sys.exit(1)

    log.info("CLEAN: %d arquivos, %d workers", len(tasks), workers)

    all_stats: list[dict] = []

    if workers == 1:
        for t in tasks:
            all_stats.append(_task(t))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_task, t): t for t in tasks}
            for fut in as_completed(futures):
                all_stats.append(fut.result())

    # Resumo
    total_kept    = sum(s.get("kept", 0) for s in all_stats)
    total_dropped = sum(s.get("dropped", 0) for s in all_stats)
    errors        = [s for s in all_stats if "error" in s]

    log.info("CLEAN concluído: kept=%d dropped=%d arquivos_com_erro=%d",
             total_kept, total_dropped, len(errors))
    if errors:
        for e in errors:
            log.error("Erro em %s: %s", e["file"], e.get("error"))

    # Salva stats no meta.json (parcial — será atualizado pelo PACK)
    meta: dict = {}
    if os.path.exists(META_JSON):
        with open(META_JSON) as f:
            meta = json.load(f)

    meta["clean_stats"] = {
        "total_kept":    total_kept,
        "total_dropped": total_dropped,
        "files":         len(all_stats),
        "errors":        len(errors),
    }
    with open(META_JSON, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Stats salvas em %s", META_JSON)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLEAN: JSONL raw → Parquet lean")
    parser.add_argument("--workers",  type=int, default=0,
                        help="Número de workers (0=auto)")
    parser.add_argument("--markets",  nargs="*", default=None,
                        help="Mercados a processar (padrão: todos)")
    args = parser.parse_args()
    main(markets=args.markets, workers=args.workers)
