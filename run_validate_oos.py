#!/usr/bin/env python3
"""
run_validate_oos.py — Executa FASE 5 (Validação Out-of-Sample).

Uso:
    python run_validate_oos.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.py.validate import run_oos
from src.py.report import generate_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

if __name__ == "__main__":
    results = run_oos()

    print("\n=== OOS concluído ===")
    g = results["global"]
    print(f"  n_trades:  {g['n_trades']}")
    print(f"  total_pnl: {g['total_pnl']:.2f}")
    print(f"  sharpe:    {g['sharpe']:.4f}")
    print(f"  max_dd:    {g['max_dd']:.2f}")

    if results["alerts"]:
        print("\n  Alertas:")
        for a in results["alerts"]:
            print(f"    {a}")

    print("\n=== Por mercado ===")
    for m in results["by_market"]:
        print(f"  {m['market']:8s}: trades={m['n_trades']:4d}  pnl={m['total_pnl']:8.2f}  "
              f"max_dd={m['max_dd']:8.2f}")

    # Gera o relatório final em Markdown
    generate_report(results)
    print("\nRelatório gerado em reports/summary.md")
