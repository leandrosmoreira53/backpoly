#!/usr/bin/env python3
"""
run_pack.py — Executa FASE 2 (PACK): Parquet lean → month.npz + cycles.npz.

Uso:
    python run_pack.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.py.pack import pack

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

if __name__ == "__main__":
    stats = pack()
    print("\n=== PACK concluído ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
