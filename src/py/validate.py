"""
validate.py — FASE 5: Validação out-of-sample com os 3 parâmetros fixos.

Carrega best_params.json (prob_entry_min + t_min/t_max + stop_loss_delta),
roda simulação no conjunto OOS, gera métricas por dia e por mercado.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

_MARKET_NAMES = {0: "BTC15m", 1: "ETH15m", 2: "SOL15m", 3: "XRP15m"}


def run_oos() -> dict[str, Any]:
    """
    Executa validação OOS e salva reports/oos_results.csv.
    Retorna dict com métricas completas.
    """
    from src.py.config import (
        MONTH_NPZ, CYCLES_NPZ, BEST_PARAMS_JSON,
        T_MIN, T_MAX, SIZE_SHARES,
        OOS_CSV, REPORTS_DIR, SCORE_MIN_TRADES,
        get_sim_threads,
    )
    from src.py.metrics import (
        compute_score, aggregate_by_day, aggregate_by_market,
    )

    if not os.path.exists(BEST_PARAMS_JSON):
        raise FileNotFoundError(
            f"{BEST_PARAMS_JSON} não encontrado. Execute run_train_grid.py primeiro."
        )

    with open(BEST_PARAMS_JSON) as f:
        best = json.load(f)

    # Lê os 3 parâmetros otimizados (com fallback para defaults se arquivo antigo)
    prob_min  = float(best["prob_entry_min"])
    t_min     = int(best.get("t_min",           T_MIN))
    t_max     = int(best.get("t_max",           T_MAX))
    stop_loss = float(best.get("stop_loss_delta", 0.0))

    log.info(
        "OOS: prob=%.4f  t_min=%d  t_max=%d  stop=%.2f",
        prob_min, t_min, t_max, stop_loss,
    )

    # Carrega com mmap (sem cópia desnecessária de RAM)
    data   = np.load(MONTH_NPZ,  mmap_mode="r")
    cycles = np.load(CYCLES_NPZ, mmap_mode="r")

    time_remaining  = np.ascontiguousarray(data["time_remaining"],    dtype=np.int32)
    prob_up         = np.ascontiguousarray(data["prob_up"],           dtype=np.float32)
    cycle_start_idx = np.ascontiguousarray(cycles["cycle_start_idx"],  dtype=np.int32)
    cycle_end_idx   = np.ascontiguousarray(cycles["cycle_end_idx"],    dtype=np.int32)
    cycle_end_ts    = np.ascontiguousarray(cycles["cycle_end_ts"],     dtype=np.int64)
    cycle_market_id = np.ascontiguousarray(cycles["cycle_market_id"],  dtype=np.int8)
    oos_ids         = np.ascontiguousarray(cycles["oos_cycle_ids"],    dtype=np.int32)

    n_threads = get_sim_threads()

    try:
        from src.cy.sim_core import run_cycles_by_market
    except ImportError:
        log.error("sim_core não compilado. Execute: python setup.py build_ext --inplace")
        raise

    pnl_arr, ent_arr, ep_arr, sth_arr, pnl_mkt, trades_mkt = run_cycles_by_market(
        time_remaining, prob_up,
        cycle_start_idx, cycle_end_idx,
        cycle_market_id,
        oos_ids,
        prob_min, t_min, t_max,
        SIZE_SHARES, stop_loss,
        n_threads,
    )

    nc = len(oos_ids)
    n_stops = int(sth_arr.sum())
    log.info("OOS: %d ciclos simulados, %d stops acionados (%.1f%%)",
             nc, n_stops, 100 * n_stops / max(int(ent_arr.sum()), 1))

    # -------------------------------------------------------------------------
    # Métricas globais
    # -------------------------------------------------------------------------
    global_metrics = compute_score(
        pnl_arr, ent_arr, cycle_end_ts, oos_ids, SCORE_MIN_TRADES
    )
    global_metrics["n_stops"] = n_stops
    log.info(
        "OOS global: n_trades=%d  total_pnl=%.2f  sharpe=%.4f  max_dd=%.2f  n_stops=%d",
        global_metrics["n_trades"], global_metrics["total_pnl"],
        global_metrics["sharpe"], global_metrics["max_dd"], n_stops,
    )

    # -------------------------------------------------------------------------
    # Por dia
    # -------------------------------------------------------------------------
    by_day = aggregate_by_day(pnl_arr, ent_arr, cycle_end_ts, oos_ids)

    # -------------------------------------------------------------------------
    # Por mercado
    # -------------------------------------------------------------------------
    by_market = aggregate_by_market(
        pnl_arr, ent_arr, oos_ids, cycle_market_id, _MARKET_NAMES
    )

    # -------------------------------------------------------------------------
    # Salvar CSV
    # -------------------------------------------------------------------------
    os.makedirs(REPORTS_DIR, exist_ok=True)
    rows_day = [
        {
            "date":          d,
            "daily_pnl":     round(float(by_day["daily_pnl"][i]), 4),
            "daily_trades":  int(by_day["daily_trades"][i]),
        }
        for i, d in enumerate(by_day["dates"])
    ]
    with open(OOS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "daily_pnl", "daily_trades"])
        w.writeheader()
        w.writerows(rows_day)
    log.info("Salvo: %s", OOS_CSV)

    # -------------------------------------------------------------------------
    # Alertas automáticos
    # -------------------------------------------------------------------------
    alerts: list[str] = []

    train_sharpe = float(best.get("sharpe", 0))
    oos_sharpe   = global_metrics["sharpe"]
    if train_sharpe > 0 and oos_sharpe < 0.5 * train_sharpe:
        alerts.append(
            f"OVERFIT PROVÁVEL: Sharpe treino={train_sharpe:.2f}, "
            f"OOS={oos_sharpe:.2f} (< 50% do treino)"
        )

    n_trades_oos = global_metrics["n_trades"]
    if n_trades_oos < SCORE_MIN_TRADES:
        alerts.append(
            f"POUCOS TRADES OOS: {n_trades_oos} trades (mín={SCORE_MIN_TRADES})"
        )

    train_max_dd = float(best.get("max_dd", 0))
    oos_max_dd   = global_metrics["max_dd"]
    if train_max_dd < 0 and oos_max_dd < 2 * train_max_dd:
        alerts.append(
            f"RISCO ELEVADO OOS: max_drawdown treino={train_max_dd:.2f}, "
            f"OOS={oos_max_dd:.2f}"
        )

    stop_pct = 100 * n_stops / max(n_trades_oos, 1)
    if stop_pct > 60:
        alerts.append(
            f"STOP MUITO AGRESSIVO: {stop_pct:.0f}% dos trades acionaram stop. "
            f"Considere aumentar stop_loss_delta."
        )

    for alert in alerts:
        log.warning(alert)

    return {
        "global":      global_metrics,
        "by_day":      rows_day,
        "by_market":   by_market,
        "alerts":      alerts,
        "best_params": best,
    }
