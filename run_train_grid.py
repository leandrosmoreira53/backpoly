#!/usr/bin/env python3
"""
run_train_grid.py — Executa FASE 4 (Grid Search no treino).

Uso:
    python run_train_grid.py [--workers N] [--seq]

--workers N : número de workers para ProcessPool (0=auto)
--seq       : modo sequencial (útil para debug)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.py.gridsearch import run_grid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid search no conjunto de treino")
    parser.add_argument("--workers", type=int, default=0,
                        help="Workers para ProcessPool (0=auto)")
    parser.add_argument("--seq", action="store_true",
                        help="Modo sequencial (sem paralelismo)")
    args = parser.parse_args()

    workers = 1 if args.seq else args.workers

    results = run_grid(workers=workers)

    print("\n=== Grid Search concluído ===")
    print(f"{'prob':>6}  {'trades':>7}  {'total_pnl':>10}  {'sharpe':>7}  {'max_dd':>8}  {'score':>8}")
    print("-" * 60)
    for r in results:
        score_str = f"{r['score']:8.4f}" if isinstance(r["score"], float) and r["score"] != float("-inf") else "    -inf"
        print(f"{r['prob_entry_min']:6.3f}  {r['n_trades']:7d}  "
              f"{r['total_pnl']:10.2f}  {r['sharpe']:7.4f}  "
              f"{r['max_dd']:8.2f}  {score_str}")
