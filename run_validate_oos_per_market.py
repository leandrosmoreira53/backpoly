#!/usr/bin/env python3
"""
run_validate_oos_per_market.py — OOS com parâmetros individuais por mercado.

Usa best_params_per_market.json (gerado por run_train_grid_per_market.py)
e simula cada mercado com seus próprios parâmetros otimizados.

Uso:
    python run_validate_oos_per_market.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from src.py.config import MARKET_MAP as _MM
MARKET_IDS = {name.lower(): mid for name, mid in _MM.items()}
LABELS     = {name.lower(): name for name in _MM}


def _max_drawdown(pnl_arr: np.ndarray) -> float:
    if len(pnl_arr) == 0:
        return 0.0
    cum  = np.cumsum(pnl_arr)
    peak = np.maximum.accumulate(cum)
    return float((cum - peak).min())


def main() -> None:
    from src.py.config import (
        MONTH_NPZ, CYCLES_NPZ, REPORTS_DIR, SIZE_SHARES, get_sim_threads,
    )

    per_mkt_json = os.path.join(REPORTS_DIR, "best_params_per_market.json")
    if not os.path.exists(per_mkt_json):
        log.error(
            "%s não encontrado.\nExecute run_train_grid_per_market.py primeiro.",
            per_mkt_json,
        )
        sys.exit(1)

    with open(per_mkt_json) as f:
        params_per_market: dict = json.load(f)

    data   = np.load(MONTH_NPZ,  mmap_mode="r")
    cycles = np.load(CYCLES_NPZ, mmap_mode="r")

    time_remaining  = np.ascontiguousarray(data["time_remaining"],    dtype=np.int16)
    prob_up         = np.ascontiguousarray(data["prob_up"],           dtype=np.float32)
    cycle_start_idx = np.ascontiguousarray(cycles["cycle_start_idx"], dtype=np.int32)
    cycle_end_idx   = np.ascontiguousarray(cycles["cycle_end_idx"],   dtype=np.int32)
    cycle_market_id = np.ascontiguousarray(cycles["cycle_market_id"], dtype=np.int8)
    oos_ids_all     = np.ascontiguousarray(cycles["oos_cycle_ids"],   dtype=np.int32)

    n_threads = get_sim_threads()

    try:
        from src.cy.sim_core import run_cycles_by_market
    except ImportError:
        log.error("sim_core não compilado. Execute: python setup.py build_ext --inplace")
        raise

    print(f"\n{'='*72}")
    print(f"  OOS POR MERCADO — parâmetros individuais")
    print(f"{'='*72}")
    print(
        f"\n  {'Mercado':<8}  {'prob':>6}  {'t_min':>5}  {'t_max':>5}  "
        f"{'stop':>5}  {'trades':>7}  {'pnl':>8}  {'max_dd':>8}  {'stops%':>6}"
    )
    print(
        f"  {'-'*8}  {'-'*6}  {'-'*5}  {'-'*5}  "
        f"{'-'*5}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*6}"
    )

    total_trades  = 0
    total_pnl     = 0.0
    global_max_dd = 0.0
    results_by_market: dict[str, dict] = {}

    for mkt, mid in MARKET_IDS.items():
        label = LABELS[mkt]

        if mkt not in params_per_market:
            log.warning("%s: sem parâmetros definidos, ignorado.", label)
            continue

        p         = params_per_market[mkt]
        prob_min  = float(p["prob_entry_min"])
        t_min     = int(p["t_min"])
        t_max     = int(p["t_max"])
        stop_loss = float(p["stop_loss_delta"])

        # Filtra ciclos OOS apenas deste mercado
        mkt_mask    = cycle_market_id[oos_ids_all] == mid
        oos_ids_mkt = np.ascontiguousarray(oos_ids_all[mkt_mask], dtype=np.int32)

        if len(oos_ids_mkt) == 0:
            log.warning("%s: 0 ciclos OOS, ignorado.", label)
            continue

        pnl_arr, ent_arr, _ep, sth_arr, _pnl_mkt, _trades_mkt = run_cycles_by_market(
            time_remaining, prob_up,
            cycle_start_idx, cycle_end_idx,
            cycle_market_id,
            oos_ids_mkt,
            prob_min, t_min, t_max,
            SIZE_SHARES, stop_loss,
            n_threads,
        )

        n_trades   = int(ent_arr.sum())
        n_stops    = int(sth_arr.sum())
        pnl_total  = float(pnl_arr.sum())
        max_dd     = _max_drawdown(pnl_arr)
        stop_pct   = 100.0 * n_stops / max(n_trades, 1)

        results_by_market[mkt] = {
            "market":          label,
            "prob_entry_min":  prob_min,
            "t_min":           t_min,
            "t_max":           t_max,
            "stop_loss_delta": stop_loss,
            "n_trades":        n_trades,
            "total_pnl":       round(pnl_total, 4),
            "max_dd":          round(max_dd, 4),
            "n_stops":         n_stops,
            "stop_pct":        round(stop_pct, 1),
        }

        total_trades  += n_trades
        total_pnl     += pnl_total
        global_max_dd  = min(global_max_dd, max_dd)

        print(
            f"  {label:<8}  {prob_min:6.3f}  {t_min:5d}  {t_max:5d}  "
            f"{stop_loss:5.2f}  {n_trades:7d}  {pnl_total:+8.2f}  "
            f"{max_dd:8.2f}  {stop_pct:5.1f}%"
        )

    print(
        f"\n  {'TOTAL':<8}  {'':6}  {'':5}  {'':5}  {'':5}  "
        f"{total_trades:7d}  {total_pnl:+8.2f}  {global_max_dd:8.2f}"
    )
    print(f"\n{'='*72}\n")

    # Alerta de stop agressivo
    for mkt, r in results_by_market.items():
        if r["stop_pct"] > 60:
            log.warning(
                "%s: %.0f%% dos trades acionaram stop — considere aumentar stop_loss_delta.",
                r["market"], r["stop_pct"],
            )

    # Salvar resultado
    out_json = os.path.join(REPORTS_DIR, "oos_per_market.json")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(
            {
                "by_market": results_by_market,
                "global": {
                    "total_trades": total_trades,
                    "total_pnl":    round(total_pnl, 4),
                    "max_dd":       round(global_max_dd, 4),
                },
            },
            f, indent=2,
        )
    log.info("Salvo: %s", out_json)

    print("  Parâmetros finais para o bot:")
    print(f"  {'Mercado':<8}  {'prob':>6}  {'t_min':>5}  {'t_max':>5}  {'stop':>5}")
    print(f"  {'-'*8}  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*5}")
    for mkt in MARKET_IDS:
        if mkt not in results_by_market:
            continue
        r = results_by_market[mkt]
        print(
            f"  {r['market']:<8}  "
            f"{r['prob_entry_min']:6.3f}  "
            f"{r['t_min']:5d}  "
            f"{r['t_max']:5d}  "
            f"{r['stop_loss_delta']:5.2f}"
        )
    print()


if __name__ == "__main__":
    main()
