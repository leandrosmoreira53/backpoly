#!/usr/bin/env python3
"""
run_evaluate_all_markets.py — Avalia TODOS os mercados e timeframes.

Para cada combinação (ativo × timeframe):
  1. Roda grid search com janelas T_WIN proporcionais ao ciclo
  2. Seleciona os melhores parâmetros (treino)
  3. Valida no período OOS
  4. Produz tabela de ranking com recomendação TRADE / SKIP

Saída:
  reports/all_markets_ranking.csv   — dados completos
  reports/all_markets_ranking.md    — tabela formatada para leitura

Uso:
    python run_evaluate_all_markets.py [--workers N] [--min-trades N]

Exemplos:
    python run_evaluate_all_markets.py
    python run_evaluate_all_markets.py --workers 8 --min-trades 20
"""
from __future__ import annotations

import argparse
import csv
import itertools
import logging
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Critérios de recomendação TRADE
# ---------------------------------------------------------------------------
TRADE_MIN_OOS_PNL    = 0.0   # PnL OOS > 0
TRADE_MIN_OOS_SHARPE = 0.5   # Sharpe OOS >= 0.5
TRADE_MIN_OOS_TRADES = 10    # ao menos 10 trades OOS
TRADE_MIN_TRAIN_WR   = 0.55  # win rate treino >= 55%
TRADE_MAX_OVERFIT    = 0.5   # Sharpe OOS >= 50% do Sharpe treino


# ---------------------------------------------------------------------------
# Worker: avalia um mercado com o T_WIN adequado ao seu timeframe
# ---------------------------------------------------------------------------
def _eval_one_market(args: tuple) -> dict:
    (market_name, market_id, month_npz, cycles_npz,
     prob_grid, t_win_grid, stop_grid,
     size_shares, min_trades) = args

    sys.path.insert(0, str(Path(__file__).parent))

    data   = np.load(month_npz,  mmap_mode="r")
    cycles = np.load(cycles_npz, mmap_mode="r")

    time_remaining  = np.ascontiguousarray(data["time_remaining"],    dtype=np.int32)
    prob_up         = np.ascontiguousarray(data["prob_up"],           dtype=np.float32)
    cycle_start_idx = np.ascontiguousarray(cycles["cycle_start_idx"], dtype=np.int32)
    cycle_end_idx   = np.ascontiguousarray(cycles["cycle_end_idx"],   dtype=np.int32)
    cycle_market_id = np.ascontiguousarray(cycles["cycle_market_id"], dtype=np.int8)
    cycle_end_ts    = np.ascontiguousarray(cycles["cycle_end_ts"],    dtype=np.int64)
    train_ids_all   = np.ascontiguousarray(cycles["train_cycle_ids"], dtype=np.int32)
    oos_ids_all     = np.ascontiguousarray(cycles["oos_cycle_ids"],   dtype=np.int32)

    # Filtra ciclos deste mercado
    train_mask = cycle_market_id[train_ids_all] == market_id
    oos_mask   = cycle_market_id[oos_ids_all]   == market_id
    train_ids  = np.ascontiguousarray(train_ids_all[train_mask], dtype=np.int32)
    oos_ids    = np.ascontiguousarray(oos_ids_all[oos_mask],     dtype=np.int32)

    base = {
        "market":         market_name,
        "n_train_cycles": int(len(train_ids)),
        "n_oos_cycles":   int(len(oos_ids)),
    }

    if len(train_ids) < min_trades:
        return {**base, "status": "SEM_DADOS",
                "prob": None, "t_min": None, "t_max": None, "stop": None,
                "train_trades": 0, "train_pnl": 0.0, "train_sharpe": 0.0,
                "train_max_dd": 0.0, "train_wr": 0.0,
                "oos_trades": 0, "oos_pnl": 0.0, "oos_sharpe": 0.0,
                "oos_max_dd": 0.0, "oos_wr": 0.0, "recomendacao": "SKIP"}

    from src.cy.sim_core import run_cycles
    from src.py.metrics   import compute_score

    # ---- Grid search no treino ----
    best: dict = {}
    best_score = float("-inf")

    for prob, (t_mn, t_mx), stop in itertools.product(prob_grid, t_win_grid, stop_grid):
        pnl_arr, ent_arr, _, _ = run_cycles(
            time_remaining, prob_up,
            cycle_start_idx, cycle_end_idx,
            train_ids,
            float(prob), int(t_mn), int(t_mx),
            float(size_shares), float(stop),
            1,
        )
        m = compute_score(pnl_arr, ent_arr, cycle_end_ts, train_ids, min_trades)
        sc = m.get("score", float("-inf"))
        if math.isfinite(sc) and sc > best_score:
            best_score = sc
            n_t  = int(ent_arr.sum())
            wins = int((pnl_arr[ent_arr == 1] > 0).sum()) if n_t > 0 else 0
            best = {
                "prob":          round(float(prob), 3),
                "t_min":         int(t_mn),
                "t_max":         int(t_mx),
                "stop":          round(float(stop), 2),
                "train_trades":  n_t,
                "train_pnl":     round(float(pnl_arr.sum()), 2),
                "train_sharpe":  round(m.get("sharpe", 0.0), 4),
                "train_max_dd":  round(m.get("max_dd", 0.0), 2),
                "train_wr":      round(wins / n_t, 4) if n_t > 0 else 0.0,
            }

    if not best:
        return {**base, "status": "SEM_COMBO_VALIDO",
                "prob": None, "t_min": None, "t_max": None, "stop": None,
                "train_trades": 0, "train_pnl": 0.0, "train_sharpe": 0.0,
                "train_max_dd": 0.0, "train_wr": 0.0,
                "oos_trades": 0, "oos_pnl": 0.0, "oos_sharpe": 0.0,
                "oos_max_dd": 0.0, "oos_wr": 0.0, "recomendacao": "SKIP"}

    # ---- Validação OOS ----
    oos_result = {"oos_trades": 0, "oos_pnl": 0.0, "oos_sharpe": 0.0,
                  "oos_max_dd": 0.0, "oos_wr": 0.0}

    if len(oos_ids) > 0:
        pnl_o, ent_o, _, _ = run_cycles(
            time_remaining, prob_up,
            cycle_start_idx, cycle_end_idx,
            oos_ids,
            float(best["prob"]), int(best["t_min"]), int(best["t_max"]),
            float(size_shares), float(best["stop"]),
            1,
        )
        mo   = compute_score(pnl_o, ent_o, cycle_end_ts, oos_ids, min_trades=1)
        n_to = int(ent_o.sum())
        wins_o = int((pnl_o[ent_o == 1] > 0).sum()) if n_to > 0 else 0
        oos_result = {
            "oos_trades":  n_to,
            "oos_pnl":     round(float(pnl_o.sum()), 2),
            "oos_sharpe":  round(mo.get("sharpe", 0.0), 4),
            "oos_max_dd":  round(mo.get("max_dd", 0.0), 2),
            "oos_wr":      round(wins_o / n_to, 4) if n_to > 0 else 0.0,
        }

    # ---- Recomendação ----
    rec = _recomendacao(best, oos_result)

    return {**base, "status": "OK", **best, **oos_result, "recomendacao": rec}


def _recomendacao(train: dict, oos: dict) -> str:
    """Decide TRADE ou SKIP baseado nos critérios quant."""
    if oos["oos_pnl"] <= TRADE_MIN_OOS_PNL:
        return "SKIP"
    if oos["oos_sharpe"] < TRADE_MIN_OOS_SHARPE:
        return "SKIP"
    if oos["oos_trades"] < TRADE_MIN_OOS_TRADES:
        return "SKIP"
    if train.get("train_wr", 0) < TRADE_MIN_TRAIN_WR:
        return "SKIP"
    train_sh = train.get("train_sharpe", 0)
    if train_sh > 0 and oos["oos_sharpe"] < TRADE_MAX_OVERFIT * train_sh:
        return "SKIP (overfit)"
    return "TRADE"


# ---------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------

def _fmt_pnl(v: float | None) -> str:
    if v is None:
        return "  —  "
    return f"{v:+.2f}"

def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "  —  "
    return f"{v*100:.1f}%"

def _fmt_val(v, fmt=".3f") -> str:
    if v is None:
        return "  —  "
    return format(v, fmt)


def _save_reports(results: list[dict], reports_dir: str) -> None:
    os.makedirs(reports_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(reports_dir, "all_markets_ranking.csv")
    fieldnames = [
        "market", "recomendacao",
        "prob", "t_min", "t_max", "stop",
        "train_trades", "train_pnl", "train_sharpe", "train_max_dd", "train_wr",
        "oos_trades",   "oos_pnl",   "oos_sharpe",   "oos_max_dd",   "oos_wr",
        "n_train_cycles", "n_oos_cycles", "status",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    log.info("Salvo: %s", csv_path)

    # Markdown
    md_path = os.path.join(reports_dir, "all_markets_ranking.md")
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"# Ranking de Mercados — Avaliação Multi-Timeframe",
        f"",
        f"Gerado em: {ts}",
        f"",
        "---",
        "",
        "## Parâmetros finais por moeda (melhores do treino → validados no OOS)",
        "",
        f"| {'Moeda':<10} | {'prob':>6} | {'t_min':>6} | {'t_max':>6} | {'stop':>5} "
        f"| {'PnL OOS':>9} | {'Max DD':>8} | {'WR Treino':>10} | {'Sharpe OOS':>11} "
        f"| {'Trades OOS':>11} | {'Status':>16} |",
        f"|{'-'*12}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*7}"
        f"|{'-'*11}|{'-'*10}|{'-'*12}|{'-'*13}|{'-'*13}|{'-'*18}|",
    ]

    for r in results:
        lines.append(
            f"| {r['market']:<10} "
            f"| {_fmt_val(r['prob']):>6} "
            f"| {_fmt_val(r['t_min'], 'd') if r['t_min'] is not None else '—':>6} "
            f"| {_fmt_val(r['t_max'], 'd') if r['t_max'] is not None else '—':>6} "
            f"| {_fmt_val(r['stop']):>5} "
            f"| {_fmt_pnl(r['oos_pnl']):>9} "
            f"| {_fmt_pnl(r['oos_max_dd']):>8} "
            f"| {_fmt_pct(r['train_wr']):>10} "
            f"| {_fmt_val(r['oos_sharpe'], '.2f') if r['oos_sharpe'] else '—':>11} "
            f"| {(r['oos_trades'] if r['oos_trades'] else '—'):>11} "
            f"| **{r['recomendacao']}** |"
        )

    # Separar TRADE de SKIP
    trades = [r for r in results if r["recomendacao"] == "TRADE"]
    skips  = [r for r in results if r["recomendacao"] != "TRADE"]

    lines += [
        "",
        "---",
        "",
        f"## Resumo",
        "",
        f"- **TRADE**: {len(trades)} mercado(s)",
        f"- **SKIP**:  {len(skips)} mercado(s)",
        "",
    ]

    if trades:
        lines += [
            "### Mercados recomendados para operar",
            "",
            f"| {'Moeda':<10} | {'prob':>6} | {'t_min':>6}s | {'t_max':>6}s | {'stop':>5} | {'PnL OOS':>9} | {'Sharpe OOS':>11} | {'WR Treino':>10} |",
            f"|{'-'*12}|{'-'*8}|{'-'*9}|{'-'*9}|{'-'*7}|{'-'*11}|{'-'*13}|{'-'*12}|",
        ]
        for r in trades:
            lines.append(
                f"| {r['market']:<10} "
                f"| {_fmt_val(r['prob']):>6} "
                f"| {r['t_min']:>7}s "
                f"| {r['t_max']:>7}s "
                f"| {_fmt_val(r['stop']):>5} "
                f"| {_fmt_pnl(r['oos_pnl']):>9} "
                f"| {_fmt_val(r['oos_sharpe'], '.2f'):>11} "
                f"| {_fmt_pct(r['train_wr']):>10} |"
            )

    lines += [
        "",
        "---",
        "",
        "## Métricas de treino completas",
        "",
        f"| {'Moeda':<10} | {'Trades':>7} | {'PnL':>8} | {'Sharpe':>7} | {'Max DD':>8} | {'WR':>6} |",
        f"|{'-'*12}|{'-'*9}|{'-'*10}|{'-'*9}|{'-'*10}|{'-'*8}|",
    ]
    for r in results:
        lines.append(
            f"| {r['market']:<10} "
            f"| {r['train_trades']:>7} "
            f"| {_fmt_pnl(r['train_pnl']):>8} "
            f"| {_fmt_val(r['train_sharpe'], '.2f'):>7} "
            f"| {_fmt_pnl(r['train_max_dd']):>8} "
            f"| {_fmt_pct(r['train_wr']):>6} |"
        )

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    log.info("Salvo: %s", md_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Avalia todos os mercados e timeframes — produz ranking TRADE/SKIP"
    )
    parser.add_argument("--workers",    type=int, default=0,
                        help="Workers paralelos (0=auto)")
    parser.add_argument("--min-trades", type=int, default=30,
                        help="Mínimo de trades treino para validar (padrão: 30)")
    args = parser.parse_args()

    from src.py.config import (
        MARKET_MAP, MONTH_NPZ, CYCLES_NPZ, REPORTS_DIR,
        PROB_GRID, STOP_LOSS_GRID, SIZE_SHARES,
        get_t_win_grid, get_grid_workers,
    )

    if not os.path.exists(MONTH_NPZ):
        log.error(
            "%s não encontrado.\n"
            "Execute primeiro:\n"
            "  1. python run_clean.py\n"
            "  2. python run_pack.py",
            MONTH_NPZ,
        )
        sys.exit(1)

    workers    = args.workers or get_grid_workers()
    min_trades = args.min_trades

    log.info(
        "Avaliando %d mercados com %d workers (min_trades=%d)...",
        len(MARKET_MAP), workers, min_trades,
    )

    task_args = [
        (
            market_name,
            market_id,
            MONTH_NPZ,
            CYCLES_NPZ,
            PROB_GRID,
            get_t_win_grid(market_name),   # T_WIN proporcional ao timeframe
            STOP_LOSS_GRID,
            SIZE_SHARES,
            min_trades,
        )
        for market_name, market_id in sorted(MARKET_MAP.items(), key=lambda x: x[1])
    ]

    results: list[dict] = []

    if workers == 1:
        for a in task_args:
            results.append(_eval_one_market(a))
            log.info("  %-10s  status=%-20s  oos_pnl=%s  rec=%s",
                     a[0],
                     results[-1]["status"],
                     _fmt_pnl(results[-1].get("oos_pnl")),
                     results[-1]["recomendacao"])
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_eval_one_market, a): a[0] for a in task_args}
            for fut in as_completed(futures):
                mkt = futures[fut]
                try:
                    r = fut.result()
                    results.append(r)
                    log.info("  %-10s  status=%-20s  oos_pnl=%s  rec=%s",
                             mkt, r["status"],
                             _fmt_pnl(r.get("oos_pnl")),
                             r["recomendacao"])
                except Exception as exc:
                    log.error("Erro em %s: %s", mkt, exc)

    # Ordena: TRADE primeiro, depois por oos_pnl desc
    results.sort(key=lambda r: (
        0 if r["recomendacao"] == "TRADE" else 1,
        -(r.get("oos_pnl") or 0.0),
    ))

    _save_reports(results, REPORTS_DIR)

    # ---------------------------------------------------------------------------
    # Tabela no terminal (igual ao screenshot)
    # ---------------------------------------------------------------------------
    print(f"\n{'='*90}")
    print(f"  RANKING DE MERCADOS — AVALIAÇÃO MULTI-TIMEFRAME")
    print(f"{'='*90}")
    print(
        f"\n  {'Moeda':<10}  {'prob':>6}  {'t_min':>6}  {'t_max':>6}  {'stop':>5}"
        f"  {'PnL OOS':>9}  {'Max DD':>8}  {'WR treino':>10}  {'Sharpe OOS':>11}  {'Status':>16}"
    )
    print(
        f"  {'-'*10}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*5}"
        f"  {'-'*9}  {'-'*8}  {'-'*10}  {'-'*11}  {'-'*16}"
    )
    for r in results:
        print(
            f"  {r['market']:<10}"
            f"  {_fmt_val(r['prob']):>6}"
            f"  {_fmt_val(r['t_min'], 'd') + 's' if r['t_min'] is not None else '  —  ':>6}"
            f"  {_fmt_val(r['t_max'], 'd') + 's' if r['t_max'] is not None else '  —  ':>6}"
            f"  {_fmt_val(r['stop']):>5}"
            f"  {_fmt_pnl(r['oos_pnl']):>9}"
            f"  {_fmt_pnl(r['oos_max_dd']):>8}"
            f"  {_fmt_pct(r['train_wr']):>10}"
            f"  {_fmt_val(r['oos_sharpe'], '.2f') if r.get('oos_sharpe') else '    —    ':>11}"
            f"  {r['recomendacao']:>16}"
        )

    trades = [r for r in results if r["recomendacao"] == "TRADE"]
    print(f"\n{'='*90}")
    print(f"  TRADE: {len(trades)}  |  SKIP: {len(results) - len(trades)}")
    print(f"{'='*90}\n")

    print(f"  Relatórios salvos em: {REPORTS_DIR}/")
    print(f"    → all_markets_ranking.csv")
    print(f"    → all_markets_ranking.md\n")


if __name__ == "__main__":
    main()
