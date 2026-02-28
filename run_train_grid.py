#!/usr/bin/env python3
"""
run_train_grid.py — FASE 4: Grid Search 3D no conjunto de treino.

Otimiza simultaneamente:
  - prob_entry_min  : limiar mínimo de probabilidade para entrar
  - (t_min, t_max)  : janela de tempo restante (segundos antes do fim)
  - stop_loss_delta : stop loss em pontos de prob (0 = hold to end)

Uso:
    python run_train_grid.py [--workers N] [--seq] [--top N]

--workers N : número de workers ProcessPool (0=auto)
--seq       : modo sequencial (sem paralelismo, útil para debug)
--top N     : mostrar N melhores combinações (padrão: 10)
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


def _score_str(v) -> str:
    if isinstance(v, float) and v == float("-inf"):
        return "    -inf"
    return f"{v:8.4f}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grid search 3D: prob × janela × stop loss"
    )
    parser.add_argument("--workers", type=int, default=0,
                        help="Workers ProcessPool (0=auto)")
    parser.add_argument("--seq", action="store_true",
                        help="Modo sequencial (sem paralelismo)")
    parser.add_argument("--top", type=int, default=10,
                        help="Quantas melhores combinações mostrar")
    args = parser.parse_args()

    workers = 1 if args.seq else args.workers

    results = run_grid(workers=workers)

    # Filtra resultados válidos (score finito)
    import math
    valid   = [r for r in results if math.isfinite(r.get("score", float("-inf")))]
    invalid = len(results) - len(valid)

    print(f"\n=== Grid Search concluído: {len(results)} combinações ===")
    if invalid:
        print(f"  ({invalid} descartadas por trades < mínimo)")

    print(f"\nTop {args.top} combinações:\n")
    header = (f"{'prob':>6}  {'t_min':>5}  {'t_max':>5}  {'stop':>5}  "
              f"{'trades':>7}  {'total_pnl':>10}  {'sharpe':>7}  "
              f"{'max_dd':>8}  {'score':>8}")
    print(header)
    print("-" * len(header))

    for r in valid[:args.top]:
        stop_str = f"{r['stop_loss_delta']:.2f}" if r['stop_loss_delta'] > 0 else "  --"
        print(
            f"{r['prob_entry_min']:6.3f}  "
            f"{r['t_min']:5d}  "
            f"{r['t_max']:5d}  "
            f"{stop_str:>5}  "
            f"{r['n_trades']:7d}  "
            f"{r['total_pnl']:10.2f}  "
            f"{r['sharpe']:7.4f}  "
            f"{r['max_dd']:8.2f}  "
            f"{_score_str(r['score'])}"
        )

    if valid:
        best = valid[0]
        print(f"\n>>> MELHOR COMBINAÇÃO:")
        print(f"    prob_entry_min  = {best['prob_entry_min']}")
        print(f"    t_min / t_max   = {best['t_min']}s / {best['t_max']}s")
        stop_label = f"{best['stop_loss_delta']:.2f}" if best['stop_loss_delta'] > 0 else "sem stop"
        print(f"    stop_loss_delta = {stop_label}")
        print(f"    score (Sharpe−dd) = {best['score']:.4f}")
        print(f"    sharpe   = {best['sharpe']:.4f}")
        print(f"    max_dd   = {best['max_dd']:.2f}")
        print(f"    n_trades = {best['n_trades']}")
        print(f"\n    Salvo em reports/best_params.json")
