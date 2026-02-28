"""
gridsearch.py — FASE 4: Grid search paralelo para prob_entry_min.

Otimizações:
  - Arrays carregados via mmap_mode='r' (sem cópia em RAM)
  - ProcessPoolExecutor: cada worker avalia 1 prob_entry_min
  - Cython sim_core: loop quente paralelo (prange)
  - cycle_ids do treino passados como array int32 (sem re-leitura)
"""
from __future__ import annotations

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
# Worker (executado em sub-processo)
# ---------------------------------------------------------------------------

def _eval_prob(args: tuple) -> dict:
    """
    Worker: avalia um único valor de prob_entry_min.
    Carregado em subprocess separado para paralelismo real.
    """
    (prob, month_npz, cycles_npz,
     t_min, t_max, size_shares, n_threads, min_trades) = args

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    data   = np.load(month_npz,  mmap_mode="r")
    cycles = np.load(cycles_npz, mmap_mode="r")

    # Contíguos necessários para memoryviews Cython
    time_remaining   = np.ascontiguousarray(data["time_remaining"],   dtype=np.int16)
    prob_up          = np.ascontiguousarray(data["prob_up"],          dtype=np.float32)
    cycle_start_idx  = np.ascontiguousarray(cycles["cycle_start_idx"], dtype=np.int32)
    cycle_end_idx    = np.ascontiguousarray(cycles["cycle_end_idx"],   dtype=np.int32)
    cycle_end_ts     = np.ascontiguousarray(cycles["cycle_end_ts"],    dtype=np.int64)
    train_ids        = np.ascontiguousarray(cycles["train_cycle_ids"], dtype=np.int32)

    from src.cy.sim_core import run_cycles
    from src.py.metrics   import compute_score

    pnl_arr, ent_arr, _ = run_cycles(
        time_remaining, prob_up,
        cycle_start_idx, cycle_end_idx,
        train_ids,
        float(prob), t_min, t_max, float(size_shares),
        n_threads,
    )

    metrics = compute_score(pnl_arr, ent_arr, cycle_end_ts, train_ids, min_trades)
    metrics["prob_entry_min"] = round(float(prob), 4)
    return metrics


# ---------------------------------------------------------------------------
# Grid search principal
# ---------------------------------------------------------------------------

def run_grid(
    prob_grid: list[float] | None = None,
    workers:   int = 0,
) -> list[dict]:
    """
    Executa grid search sobre PROB_GRID e retorna resultados ordenados por score.
    Salva reports/grid_train.csv e reports/best_params.json.
    """
    import csv

    from src.py.config import (
        MONTH_NPZ, CYCLES_NPZ, PROB_GRID,
        T_MIN, T_MAX, SIZE_SHARES, SCORE_MIN_TRADES,
        GRID_CSV, BEST_PARAMS_JSON, REPORTS_DIR,
        get_grid_workers, get_sim_threads,
    )

    prob_grid = prob_grid or PROB_GRID
    workers   = workers or get_grid_workers()
    n_threads = get_sim_threads()

    if not os.path.exists(MONTH_NPZ):
        raise FileNotFoundError(f"{MONTH_NPZ} não encontrado. Execute run_pack.py primeiro.")

    log.info("Grid search: %d valores de prob_entry_min, %d workers", len(prob_grid), workers)

    task_args = [
        (p, MONTH_NPZ, CYCLES_NPZ,
         T_MIN, T_MAX, SIZE_SHARES, n_threads, SCORE_MIN_TRADES)
        for p in prob_grid
    ]

    results: list[dict] = []

    if workers == 1:
        # Sequencial (útil para debug)
        for args in task_args:
            results.append(_eval_prob(args))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_eval_prob, args): args[0] for args in task_args}
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as exc:
                    prob = futures[fut]
                    log.error("Erro ao avaliar prob=%.3f: %s", prob, exc)

    # Ordena por prob_entry_min para o CSV
    results.sort(key=lambda r: r["prob_entry_min"])

    # Salva CSV
    os.makedirs(REPORTS_DIR, exist_ok=True)
    fieldnames = ["prob_entry_min", "n_trades", "total_pnl", "sharpe", "max_dd", "score"]
    with open(GRID_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    log.info("Grid salvo: %s", GRID_CSV)

    # Melhor parâmetro (score máximo, excluindo -inf)
    valid = [r for r in results if math.isfinite(r.get("score", float("-inf")))]
    if not valid:
        log.warning("Nenhum parâmetro válido encontrado (todos abaixo de min_trades=%d)",
                    SCORE_MIN_TRADES)
        return results

    best = max(valid, key=lambda r: r["score"])
    with open(BEST_PARAMS_JSON, "w") as f:
        json.dump(best, f, indent=2)
    log.info("Melhor parâmetro: prob_entry_min=%.3f (score=%.4f, n_trades=%d)",
             best["prob_entry_min"], best["score"], best["n_trades"])

    return results
