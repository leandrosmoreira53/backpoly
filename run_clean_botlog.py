#!/usr/bin/env python3
"""
run_clean_botlog.py — FASE 1 (variante): bot log JSONL → Parquet lean.

Lê os arquivos JSONL gerados pelo bot de execução (formato diferente
do feed de dados — campos flat, market em minúsculas, ts em segundos)
e converte para o mesmo Parquet lean que o restante do pipeline consome.

Uso:
    python run_clean_botlog.py [--dir PATH] [--workers N]

--dir PATH  : diretório com os JSONL do bot (padrão: data_raw/bot_logs/)
--workers N : paralelismo (0=auto)

Saída: data_cache/lean/YYYY-MM-DD.{MARKET}.parquet
       (mesmo local que run_clean.py — pack.py lê tudo junto)

Formato de entrada esperado (uma linha por evento):
  {
    "ts": 1772237460,           ← timestamp em segundos
    "market": "btc",            ← minúsculo sem sufixo
    "cycle_end_ts": 1772237700, ← fim do ciclo em segundos
    "yes_price": 0.82,          ← prob_up (flat, não nested)
    "no_price":  0.18,          ← prob_down
    "time_to_expiry": 240,      ← opcional (calculado se ausente)
    "action": "SKIP_PRICE_OOR"  ← usado só para err_flag
    ...
  }

Linhas com yes_price/no_price null (ex: NEW_CYCLE) são descartadas.
Múltiplas linhas no mesmo segundo (ORDER_PLACED/TIMEOUT/FAILED) são
deduplicadas — mantém a última por (market, cycle_end_ts, ts_s).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.py.config        import BOT_LOGS_DIR, LEAN_DIR, META_JSON, get_clean_workers
from src.py.clean_botlog  import clean_botlog_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _task(args: tuple[str, str]) -> dict:
    raw_path, out_dir = args
    try:
        return clean_botlog_file(raw_path, out_dir)
    except Exception as exc:
        return {"file": raw_path, "error": str(exc), "kept": 0, "dropped": 0}


def main(src_dir: str | None = None, workers: int = 0) -> None:
    src_dir = Path(src_dir or BOT_LOGS_DIR)
    out_dir = Path(LEAN_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs(os.path.dirname(META_JSON), exist_ok=True)

    workers = workers or get_clean_workers()

    if not src_dir.exists():
        log.error(
            "Diretório não encontrado: %s\n"
            "Crie-o e coloque os JSONL do bot lá, ou use --dir para especificar outro.",
            src_dir,
        )
        sys.exit(1)

    tasks = [
        (str(f), str(out_dir))
        for f in sorted(src_dir.glob("*.jsonl"))
    ]

    if not tasks:
        log.error("Nenhum arquivo *.jsonl encontrado em %s", src_dir)
        sys.exit(1)

    log.info("BOTLOG CLEAN: %d arquivo(s), %d workers  →  %s", len(tasks), workers, out_dir)

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
    total_kept    = sum(s.get("kept",    0) for s in all_stats)
    total_dropped = sum(s.get("dropped", 0) for s in all_stats)
    errors        = [s for s in all_stats if "error" in s]

    log.info(
        "BOTLOG CLEAN concluído: kept=%d  dropped=%d  erros=%d",
        total_kept, total_dropped, len(errors),
    )
    if errors:
        for e in errors:
            log.error("  Erro em %s: %s", e["file"], e.get("error"))

    # Salva stats no meta.json
    meta: dict = {}
    if os.path.exists(META_JSON):
        with open(META_JSON) as f:
            meta = json.load(f)

    meta["clean_botlog_stats"] = {
        "total_kept":    total_kept,
        "total_dropped": total_dropped,
        "files":         len(all_stats),
        "errors":        len(errors),
    }
    with open(META_JSON, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Stats salvas em %s", META_JSON)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BOTLOG CLEAN: log do bot JSONL → Parquet lean"
    )
    parser.add_argument(
        "--dir", default=None,
        help=f"Diretório com JSONL do bot (padrão: {BOT_LOGS_DIR})"
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Número de workers (0=auto)"
    )
    args = parser.parse_args()
    main(src_dir=args.dir, workers=args.workers)
