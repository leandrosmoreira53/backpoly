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
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.py.gridsearch import run_grid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

_MARKETS = ["btc", "eth", "sol", "xrp"]
_MKT_LABEL = {"btc": "BTC15m", "eth": "ETH15m", "sol": "SOL15m", "xrp": "XRP15m"}


def _score_str(v) -> str:
    if isinstance(v, float) and not math.isfinite(v):
        return "    -inf"
    return f"{v:8.4f}"


def _print_market_table(result: dict) -> None:
    """Imprime tabela de métricas por mercado para uma combinação."""
    print(f"\n    {'Mercado':<8}  {'PnL total':>10}  {'Trades':>7}  {'Win Rate':>9}")
    print(f"    {'-'*8}  {'-'*10}  {'-'*7}  {'-'*9}")
    for m in _MARKETS:
        pnl    = result.get(f"{m}_pnl",    0.0)
        trades = result.get(f"{m}_trades", 0)
        wr     = result.get(f"{m}_wr",     0.0)
        sign   = "+" if pnl >= 0 else ""
        wr_str = f"{wr*100:5.1f}%" if trades > 0 else "   n/a"
        print(f"    {_MKT_LABEL[m]:<8}  {sign}{pnl:>9.2f}  {trades:>7d}  {wr_str:>9}")


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

    valid   = [r for r in results if math.isfinite(r.get("score", float("-inf")))]
    invalid = len(results) - len(valid)

    print(f"\n{'='*72}")
    print(f"  Grid Search 3D concluído: {len(results)} combinações")
    if invalid:
        print(f"  ({invalid} descartadas por trades < mínimo)")
    print(f"{'='*72}")

    # ------------------------------------------------------------------
    # Top N — tabela global resumida
    # ------------------------------------------------------------------
    top_n = valid[:args.top]
    if top_n:
        print(f"\nTop {len(top_n)} combinações (por score):\n")
        hdr = (f"{'prob':>6}  {'t_min':>5}  {'t_max':>5}  {'stop':>5}  "
               f"{'trades':>7}  {'pnl':>9}  {'sharpe':>7}  "
               f"{'max_dd':>8}  {'score':>8}")
        print(hdr)
        print("-" * len(hdr))
        for r in top_n:
            stop_s = f"{r['stop_loss_delta']:.2f}" if r['stop_loss_delta'] > 0 else "  --"
            print(
                f"{r['prob_entry_min']:6.3f}  "
                f"{r['t_min']:5d}  "
                f"{r['t_max']:5d}  "
                f"{stop_s:>5}  "
                f"{r['n_trades']:7d}  "
                f"{r['total_pnl']:9.2f}  "
                f"{r['sharpe']:7.4f}  "
                f"{r['max_dd']:8.2f}  "
                f"{_score_str(r['score'])}"
            )

    # ------------------------------------------------------------------
    # Melhor combinação — detalhe completo com breakdown por mercado
    # ------------------------------------------------------------------
    if valid:
        best = valid[0]
        print(f"\n{'='*72}")
        print(f"  MELHOR COMBINAÇÃO")
        print(f"{'='*72}")
        print(f"\n  prob_entry_min  = {best['prob_entry_min']}")
        print(f"  t_min / t_max   = {best['t_min']}s / {best['t_max']}s")
        stop_lbl = (f"{best['stop_loss_delta']:.2f}" if best['stop_loss_delta'] > 0
                    else "sem stop (hold to end)")
        print(f"  stop_loss_delta = {stop_lbl}")
        print(f"\n  Score global:")
        print(f"    score    = {_score_str(best['score'])}")
        print(f"    sharpe   = {best['sharpe']:.4f}")
        print(f"    max_dd   = {best['max_dd']:.2f}")
        print(f"    n_trades = {best['n_trades']}")
        print(f"    total_pnl= {best['total_pnl']:.2f}")

        if any(f"{m}_trades" in best for m in _MARKETS):
            print(f"\n  Breakdown por mercado (treino):")
            _print_market_table(best)

        print(f"\n  Salvo em reports/best_params.json")
        print(f"{'='*72}\n")
