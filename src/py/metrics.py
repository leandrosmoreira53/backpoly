"""
metrics.py — Funções de score e métricas para o backtest.

Todas as funções trabalham com arrays NumPy para máxima velocidade.
"""
from __future__ import annotations

import math
import numpy as np


def compute_max_drawdown(cumulative_pnl: np.ndarray) -> float:
    """
    Calcula o max drawdown dado o PnL cumulativo.
    Retorna valor negativo (perda), 0.0 se sem drawdown.
    """
    if len(cumulative_pnl) == 0:
        return 0.0
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdowns   = cumulative_pnl - running_max
    return float(drawdowns.min())


def compute_sharpe(daily_pnl: np.ndarray, annualize: bool = True) -> float:
    """
    Sharpe ratio de PnL diário.
    annualize=True → multiplica por sqrt(252).
    """
    if len(daily_pnl) < 2:
        return 0.0
    std = daily_pnl.std()
    if std < 1e-9:
        return 0.0
    sr = daily_pnl.mean() / std
    return float(sr * math.sqrt(252) if annualize else sr)


def aggregate_by_day(
    pnl_arr: np.ndarray,
    entered_arr: np.ndarray,
    cycle_end_ts: np.ndarray,
    cycle_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Agrega PnL e trades por dia (UTC).
    Recebe apenas os ciclos do subset (cycle_ids) para evitar indexação errada.

    Retorna dict com arrays indexados pelos dias únicos.
    """
    from datetime import datetime, timezone

    nc = len(cycle_ids)
    dates = np.array([
        datetime.fromtimestamp(int(cycle_end_ts[cid]), tz=timezone.utc)
               .strftime("%Y-%m-%d")
        for cid in cycle_ids
    ])
    unique_dates = sorted(set(dates))
    date_to_idx  = {d: i for i, d in enumerate(unique_dates)}
    nd = len(unique_dates)

    daily_pnl    = np.zeros(nd, dtype=np.float64)
    daily_trades = np.zeros(nd, dtype=np.int32)

    for ci in range(nc):
        di = date_to_idx[dates[ci]]
        daily_pnl[di]    += pnl_arr[ci]
        daily_trades[di] += int(entered_arr[ci])

    return {
        "dates":        np.array(unique_dates),
        "daily_pnl":    daily_pnl,
        "daily_trades": daily_trades,
    }


def aggregate_by_market(
    pnl_arr: np.ndarray,
    entered_arr: np.ndarray,
    cycle_ids: np.ndarray,
    cycle_market_id: np.ndarray,
    market_names: dict[int, str],
) -> list[dict]:
    """
    Agrega PnL e trades por mercado.
    """
    rows = []
    for mid, name in market_names.items():
        mask  = np.array([cycle_market_id[cid] == mid for cid in cycle_ids], dtype=bool)
        pnl_m = pnl_arr[mask]
        ent_m = entered_arr[mask]

        n_trades   = int(ent_m.sum())
        total_pnl  = float(pnl_m.sum())
        cumsum     = np.cumsum(pnl_m)
        max_dd     = compute_max_drawdown(cumsum) if len(cumsum) > 0 else 0.0

        rows.append({
            "market":    name,
            "n_cycles":  int(mask.sum()),
            "n_trades":  n_trades,
            "total_pnl": round(total_pnl, 4),
            "max_dd":    round(max_dd, 4),
        })
    return rows


def compute_score(
    pnl_arr: np.ndarray,
    entered_arr: np.ndarray,
    cycle_end_ts: np.ndarray,
    cycle_ids: np.ndarray,
    min_trades: int = 30,
) -> dict:
    """
    Calcula score para seleção de parâmetro no grid search.

    Score = Sharpe diário − 0.3 × |max_drawdown|
    Retorna -inf se n_trades < min_trades.
    """
    n_trades = int(entered_arr.sum())
    total_pnl = float(pnl_arr.sum())
    cum       = np.cumsum(pnl_arr.astype(np.float64))
    max_dd    = compute_max_drawdown(cum)

    if n_trades < min_trades:
        score = float("-inf")
        sharpe = 0.0
    else:
        by_day   = aggregate_by_day(pnl_arr, entered_arr, cycle_end_ts, cycle_ids)
        sharpe   = compute_sharpe(by_day["daily_pnl"])
        score    = sharpe - 0.3 * abs(max_dd)

    return {
        "n_trades":  n_trades,
        "total_pnl": round(total_pnl, 4),
        "sharpe":    round(sharpe, 4),
        "max_dd":    round(max_dd, 4),
        "score":     round(score, 4) if math.isfinite(score) else score,
    }
