"""
report.py — FASE 6: Gera reports/summary.md com resultados completos.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any


def generate_report(oos_results: dict[str, Any]) -> None:
    """
    Gera reports/summary.md a partir dos resultados OOS.
    """
    from src.py.config import (
        SUMMARY_MD, REPORTS_DIR, GRID_CSV, BEST_PARAMS_JSON,
        TRAIN_DAYS, OOS_DAYS,
    )

    os.makedirs(REPORTS_DIR, exist_ok=True)

    best   = oos_results.get("best_params", {})
    global_m = oos_results.get("global", {})
    by_day   = oos_results.get("by_day", [])
    by_mkt   = oos_results.get("by_market", [])
    alerts   = oos_results.get("alerts", [])

    # Carrega métricas do treino do best_params.json (se disponível)
    train_sharpe = best.get("sharpe", "N/A")
    train_pnl    = best.get("total_pnl", "N/A")
    train_dd     = best.get("max_dd", "N/A")
    train_trades = best.get("n_trades", "N/A")

    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = [
        f"# Backtest Report — Polymarket 15m",
        f"",
        f"Gerado em: {now}",
        f"",
        f"## Parâmetros selecionados",
        f"",
        f"| Parâmetro       | Valor                        |",
        f"|-----------------|------------------------------|",
        f"| prob_entry_min  | `{best.get('prob_entry_min', 'N/A')}`  |",
        f"| Janela de entrada | T_MIN={best.get('T_MIN', 60)}s .. T_MAX={best.get('T_MAX', 240)}s |",
        f"| Período treino  | {TRAIN_DAYS} dias              |",
        f"| Período OOS     | {OOS_DAYS} dias               |",
        f"",
        f"## Treino vs OOS",
        f"",
        f"| Métrica     | Treino           | OOS              |",
        f"|-------------|------------------|------------------|",
        f"| n_trades    | {train_trades}   | {global_m.get('n_trades', '?')} |",
        f"| total_pnl   | {_fmt(train_pnl)} | {_fmt(global_m.get('total_pnl', 0))} |",
        f"| Sharpe      | {_fmt(train_sharpe, 4)} | {_fmt(global_m.get('sharpe', 0), 4)} |",
        f"| max_dd      | {_fmt(train_dd, 2)} | {_fmt(global_m.get('max_dd', 0), 2)} |",
        f"",
        f"## OOS por mercado",
        f"",
        f"| Mercado | Ciclos | Trades | Total PnL | Max DD |",
        f"|---------|--------|--------|-----------|--------|",
    ]

    for m in by_mkt:
        lines.append(
            f"| {m['market']:7s} | {m['n_cycles']:6d} | {m['n_trades']:6d} "
            f"| {m['total_pnl']:9.2f} | {m['max_dd']:6.2f} |"
        )

    lines += [
        f"",
        f"## OOS por dia",
        f"",
        f"| Data       | PnL diário | Trades |",
        f"|------------|------------|--------|",
    ]
    for d in by_day:
        lines.append(
            f"| {d['date']} | {d['daily_pnl']:10.2f} | {d['daily_trades']:6d} |"
        )

    if alerts:
        lines += [
            f"",
            f"## ⚠ Alertas",
            f"",
        ]
        for a in alerts:
            lines.append(f"- {a}")
    else:
        lines += [
            f"",
            f"## Alertas",
            f"",
            f"Nenhum alerta. Resultados parecem estáveis.",
        ]

    lines += [
        f"",
        f"---",
        f"",
        f"Arquivos gerados:",
        f"- `{GRID_CSV}` — todos os parâmetros testados no treino",
        f"- `{BEST_PARAMS_JSON}` — parâmetro selecionado",
        f"- `reports/oos_results.csv` — PnL diário OOS",
        f"- `reports/summary.md` — este relatório",
    ]

    with open(SUMMARY_MD, "w") as f:
        f.write("\n".join(lines) + "\n")


def _fmt(val: Any, decimals: int = 2) -> str:
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)
