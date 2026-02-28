"""
gridsearch.py — FASE 4: Grid search 3D paralelo.

Parâmetros otimizados simultaneamente:
  - prob_entry_min  (limiar de entrada)
  - (t_min, t_max)  (janela de tempo restante)
  - stop_loss_delta (stop loss em pontos de prob, 0 = sem stop)

Total: 16 × 5 × 5 = 400 combinações
Com ProcessPoolExecutor + Cython prange → < 3s
"""
from __future__ import annotations

import csv
import itertools
import json
import logging
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker (sub-processo independente por combinação de parâmetros)
# ---------------------------------------------------------------------------

def _build_mkt_names() -> dict[int, str]:
    from src.py.config import MARKET_MAP
    return {mid: name.lower() for name, mid in MARKET_MAP.items()}

_MKT_NAMES: dict[int, str] = {}  # preenchido lazily em _eval_params


def _eval_params(args: tuple) -> dict:
    (prob, t_min, t_max, stop_loss,
     month_npz, cycles_npz, size_shares, n_threads, min_trades) = args

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    data   = np.load(month_npz,  mmap_mode="r")
    cycles = np.load(cycles_npz, mmap_mode="r")

    time_remaining  = np.ascontiguousarray(data["time_remaining"],    dtype=np.int32)
    prob_up         = np.ascontiguousarray(data["prob_up"],           dtype=np.float32)
    cycle_start_idx = np.ascontiguousarray(cycles["cycle_start_idx"], dtype=np.int32)
    cycle_end_idx   = np.ascontiguousarray(cycles["cycle_end_idx"],   dtype=np.int32)
    cycle_end_ts    = np.ascontiguousarray(cycles["cycle_end_ts"],    dtype=np.int64)
    cycle_market_id = np.ascontiguousarray(cycles["cycle_market_id"], dtype=np.int8)
    train_ids       = np.ascontiguousarray(cycles["train_cycle_ids"], dtype=np.int32)

    from src.cy.sim_core import run_cycles_by_market
    from src.py.metrics   import compute_score
    from src.py.config    import MARKET_MAP

    mkt_names  = {mid: name.lower() for name, mid in MARKET_MAP.items()}
    n_markets  = len(MARKET_MAP)

    pnl_arr, ent_arr, _, _, pnl_mkt, trades_mkt = run_cycles_by_market(
        time_remaining, prob_up,
        cycle_start_idx, cycle_end_idx,
        cycle_market_id,
        train_ids,
        float(prob), int(t_min), int(t_max),
        float(size_shares), float(stop_loss),
        n_threads,
        n_markets,
    )

    metrics = compute_score(pnl_arr, ent_arr, cycle_end_ts, train_ids, min_trades)
    metrics.update({
        "prob_entry_min":  round(float(prob),      4),
        "t_min":           int(t_min),
        "t_max":           int(t_max),
        "stop_loss_delta": round(float(stop_loss), 4),
    })

    # --- métricas por mercado ---
    mkt_ids = np.array([cycle_market_id[cid] for cid in train_ids], dtype=np.int8)
    for mid, name in mkt_names.items():
        mask   = mkt_ids == mid
        pnl_m  = pnl_arr[mask]
        n_t    = int(trades_mkt[mid])
        wins   = int((pnl_m > 0).sum()) if n_t > 0 else 0
        win_rt = round(wins / n_t, 4)   if n_t > 0 else 0.0
        metrics[f"{name}_pnl"]    = round(float(pnl_mkt[mid]), 4)
        metrics[f"{name}_trades"] = n_t
        metrics[f"{name}_wr"]     = win_rt

    return metrics


# ---------------------------------------------------------------------------
# Grid search principal
# ---------------------------------------------------------------------------

def run_grid(
    prob_grid:      list[float]            | None = None,
    t_win_grid:     list[tuple[int, int]]  | None = None,
    stop_loss_grid: list[float]            | None = None,
    workers: int = 0,
) -> list[dict]:
    """
    Grid search 3D: prob × janela × stop loss.
    Salva reports/grid_train.csv e reports/best_params.json.
    Retorna lista de resultados ordenada por score (desc).
    """
    from src.py.config import (
        MONTH_NPZ, CYCLES_NPZ,
        PROB_GRID, T_WIN_GRID, STOP_LOSS_GRID,
        SIZE_SHARES, SCORE_MIN_TRADES,
        GRID_CSV, BEST_PARAMS_JSON, REPORTS_DIR,
        get_grid_workers, get_sim_threads,
    )

    prob_grid      = prob_grid      or PROB_GRID
    t_win_grid     = t_win_grid     or T_WIN_GRID
    stop_loss_grid = stop_loss_grid or STOP_LOSS_GRID
    workers        = workers        or get_grid_workers()
    n_threads      = get_sim_threads()

    if not os.path.exists(MONTH_NPZ):
        raise FileNotFoundError(
            f"{MONTH_NPZ} não encontrado. Execute run_pack.py primeiro."
        )

    combos = list(itertools.product(prob_grid, t_win_grid, stop_loss_grid))
    log.info(
        "Grid search 3D: %d probs × %d janelas × %d stops = %d combinações, %d workers",
        len(prob_grid), len(t_win_grid), len(stop_loss_grid), len(combos), workers,
    )

    task_args = [
        (prob, t_min, t_max, stop,
         MONTH_NPZ, CYCLES_NPZ, SIZE_SHARES, n_threads, SCORE_MIN_TRADES)
        for prob, (t_min, t_max), stop in combos
    ]

    results: list[dict] = []

    if workers == 1:
        for args in task_args:
            results.append(_eval_params(args))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_eval_params, a): a for a in task_args}
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as exc:
                    a = futures[fut]
                    log.error("Erro em prob=%.3f t=%d-%d stop=%.2f: %s",
                              a[0], a[1], a[2], a[3], exc)

    # Ordena por score desc para o relatório
    results.sort(key=lambda r: r.get("score", float("-inf")), reverse=True)

    # Salva CSV (ordenado por prob/t_min/t_max/stop para facilitar análise)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    csv_rows = sorted(results, key=lambda r: (
        r["prob_entry_min"], r["t_min"], r["t_max"], r["stop_loss_delta"]
    ))
    from src.py.config import MARKET_MAP as _MM
    _mkt_cols: list[str] = []
    for _name in sorted(_MM.keys(), key=lambda n: _MM[n]):
        _n = _name.lower()
        _mkt_cols += [f"{_n}_pnl", f"{_n}_trades", f"{_n}_wr"]

    fieldnames = [
        "prob_entry_min", "t_min", "t_max", "stop_loss_delta",
        "n_trades", "total_pnl", "sharpe", "max_dd", "score",
    ] + _mkt_cols
    with open(GRID_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(csv_rows)
    log.info("Grid salvo: %s  (%d linhas)", GRID_CSV, len(csv_rows))

    # Melhor parâmetro (score máximo finito)
    valid = [r for r in results if math.isfinite(r.get("score", float("-inf")))]
    if not valid:
        log.warning(
            "Nenhuma combinação válida (todos abaixo de min_trades=%d). "
            "Reduza SCORE_MIN_TRADES em config.py.", SCORE_MIN_TRADES
        )
        return results

    best = valid[0]  # já ordenado por score desc
    with open(BEST_PARAMS_JSON, "w") as f:
        json.dump(best, f, indent=2)

    log.info(
        "Melhor: prob=%.3f  t_min=%d  t_max=%d  stop=%.2f  "
        "score=%.4f  sharpe=%.4f  n_trades=%d",
        best["prob_entry_min"], best["t_min"], best["t_max"],
        best["stop_loss_delta"], best["score"],
        best["sharpe"], best["n_trades"],
    )
    return results
