#!/usr/bin/env python3
"""
run_train_grid_per_market.py — Seleciona melhores parâmetros por mercado.

Lê grid_train.csv (gerado por run_train_grid.py) e para cada mercado
seleciona a combinação com melhor score individual.

Score por mercado = total_pnl * win_rate / sqrt(trades)
  → recompensa PnL alto + win rate alto + penaliza ruído de poucos trades

Salva: reports/best_params_per_market.json

Uso:
    python run_train_grid_per_market.py
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.py.config import GRID_CSV, REPORTS_DIR, MARKET_MAP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MARKETS = [name.lower() for name in sorted(MARKET_MAP, key=MARKET_MAP.__getitem__)]
LABELS  = {name.lower(): name for name in MARKET_MAP}
MIN_TRADES = 30

OUT_JSON = os.path.join(REPORTS_DIR, "best_params_per_market.json")


def _market_score(row: dict, mkt: str) -> float:
    """Score por mercado: PnL total × win_rate / sqrt(trades)."""
    trades = int(row.get(f"{mkt}_trades", 0))
    pnl    = float(row.get(f"{mkt}_pnl",    0.0))
    wr     = float(row.get(f"{mkt}_wr",     0.0))
    if trades < MIN_TRADES or pnl <= 0:
        return float("-inf")
    return pnl * wr / math.sqrt(trades)


def main() -> None:
    if not os.path.exists(GRID_CSV):
        log.error(
            "%s não encontrado.\nExecute run_train_grid.py primeiro.", GRID_CSV
        )
        sys.exit(1)

    with open(GRID_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    log.info("Grid carregado: %d combinações", len(rows))

    best_per_market: dict[str, dict] = {}

    print(f"\n{'='*72}")
    print(f"  MELHORES PARÂMETROS POR MERCADO (treino)")
    print(f"{'='*72}")

    for mkt in MARKETS:
        label = LABELS[mkt]

        scored = [(r, _market_score(r, mkt)) for r in rows]
        scored.sort(key=lambda x: x[1], reverse=True)

        best_row, best_score = scored[0]

        if not math.isfinite(best_score):
            log.warning("%s: nenhuma combinação válida (trades < %d ou PnL <= 0)", label, MIN_TRADES)
            continue

        best_per_market[mkt] = {
            "market":          label,
            "prob_entry_min":  float(best_row["prob_entry_min"]),
            "t_min":           int(best_row["t_min"]),
            "t_max":           int(best_row["t_max"]),
            "stop_loss_delta": float(best_row["stop_loss_delta"]),
            "train_pnl":       float(best_row[f"{mkt}_pnl"]),
            "train_trades":    int(best_row[f"{mkt}_trades"]),
            "train_wr":        round(float(best_row[f"{mkt}_wr"]) * 100, 1),
        }

        # Top 3 desta moeda para referência
        top3 = [r for r, s in scored[:3] if math.isfinite(s)]

        print(f"\n  {label}")
        print(f"  {'-'*68}")
        print(f"  {'prob':>6}  {'t_min':>5}  {'t_max':>5}  {'stop':>5}  {'trades':>7}  {'pnl':>8}  {'win%':>6}")
        for r in top3:
            print(
                f"  {float(r['prob_entry_min']):6.3f}  "
                f"{int(r['t_min']):5d}  "
                f"{int(r['t_max']):5d}  "
                f"{float(r['stop_loss_delta']):5.2f}  "
                f"{int(r[f'{mkt}_trades']):7d}  "
                f"{float(r[f'{mkt}_pnl']):+8.2f}  "
                f"{float(r[f'{mkt}_wr'])*100:5.1f}%"
            )
        print(f"\n  >>> ESCOLHIDO: prob={float(best_row['prob_entry_min']):.3f}  "
              f"t={int(best_row['t_min'])}-{int(best_row['t_max'])}s  "
              f"stop={float(best_row['stop_loss_delta']):.2f}")

    print(f"\n{'='*72}\n")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(best_per_market, f, indent=2)
    log.info("Salvo: %s", OUT_JSON)

    # Resumo final compacto
    print("  Resumo para o bot:")
    print(f"  {'Mercado':<8}  {'prob':>6}  {'t_min':>5}  {'t_max':>5}  {'stop':>5}  {'PnL treino':>10}  {'WinRate':>7}")
    print(f"  {'-'*8}  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*10}  {'-'*7}")
    for mkt in MARKETS:
        if mkt not in best_per_market:
            continue
        p = best_per_market[mkt]
        print(
            f"  {p['market']:<8}  "
            f"{p['prob_entry_min']:6.3f}  "
            f"{p['t_min']:5d}  "
            f"{p['t_max']:5d}  "
            f"{p['stop_loss_delta']:5.2f}  "
            f"{p['train_pnl']:+10.2f}  "
            f"{p['train_wr']:6.1f}%"
        )
    print()


if __name__ == "__main__":
    main()
