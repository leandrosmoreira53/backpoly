"""
clean.py — FASE 1: converte JSONL raw → Parquet lean.

Otimizações vs plano original:
  - orjson (3-5x mais rápido que json stdlib)
  - Saída em Parquet/snappy (10x menor, 10x mais rápido de ler no PACK)
  - Dedup por (market_id, cycle_end_ts, ts_s) — mantém a ÚLTIMA obs
  - Pré-computa best_prob, best_side, entry_eligible (nunca no loop do sim)
  - Chamado em paralelo por run_clean.py (um worker por arquivo)
"""
from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import orjson as _json_lib
    _LOADS = _json_lib.loads
except ImportError:
    import json as _json_lib  # type: ignore
    _LOADS = json.loads
    logging.warning("orjson não encontrado; usando json stdlib (mais lento). "
                    "Instale com: pip install orjson")

from src.py.config import (
    MARKET_MAP, CYCLE_LEN_MAP, T_WIN_MAP, get_timeframe,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema Parquet de saída
# ---------------------------------------------------------------------------
LEAN_SCHEMA = pa.schema([
    pa.field("ts_s",           pa.int64()),
    pa.field("market_id",      pa.int8()),
    pa.field("cycle_end_ts",   pa.int64()),
    pa.field("time_remaining", pa.int32()),
    pa.field("prob_up",        pa.float32()),
    pa.field("prob_down",      pa.float32()),
    pa.field("best_prob",      pa.float32()),
    pa.field("best_side",      pa.int8()),
    pa.field("entry_eligible", pa.int8()),
    pa.field("overround",      pa.float32()),
    pa.field("err_flag",       pa.int8()),
])


def _get_probs(rec: dict[str, Any]) -> tuple[float, float] | None:
    """Extrai (prob_up, prob_down) com fallback. Retorna None se inválido."""
    derived = rec.get("derived") or {}
    pu = derived.get("prob_up")
    pd_ = derived.get("prob_down")

    if pu is None or pd_ is None:
        yes = rec.get("yes") or {}
        no_ = rec.get("no") or {}
        pu  = yes.get("mid")
        pd_ = no_.get("mid")

    if pu is None or pd_ is None:
        return None
    try:
        pu, pd_ = float(pu), float(pd_)
    except (TypeError, ValueError):
        return None

    if not (0.001 <= pu <= 0.999) or not (0.001 <= pd_ <= 0.999):
        return None
    return pu, pd_


def _get_overround(rec: dict[str, Any]) -> float:
    try:
        return float((rec.get("derived") or {}).get("overround") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _get_latency(rec: dict[str, Any]) -> float:
    try:
        return float((rec.get("fetch") or {}).get("latency_ms") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def clean_file(raw_path: str | Path, out_path: str | Path, debug: bool = False) -> dict:
    """
    Limpa um arquivo JSONL raw e escreve Parquet lean.

    Retorna dict com stats: kept, dropped, latencies, drop_reasons.
    Se debug=True, loga os primeiros 3 exemplos de cada motivo de rejeição.
    """
    raw_path = Path(raw_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: dict[tuple, dict] = {}  # (market_id, cycle_end_ts, ts_s) → row
    dropped = 0
    latencies: list[float] = []
    drop_reasons: dict[str, int] = {}
    drop_examples: dict[str, list] = {}  # reason → [exemplos de campos]

    def _drop(reason: str, info: str = "") -> None:
        nonlocal dropped
        dropped += 1
        drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
        if debug and drop_reasons[reason] <= 3:
            drop_examples.setdefault(reason, []).append(info)

    with open(raw_path, "rb") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = _LOADS(line)
            except Exception:
                _drop("json_parse_error")
                continue

            # market
            market_str = rec.get("market")
            if market_str not in MARKET_MAP:
                _drop("market_unknown", f"market={market_str!r}")
                continue
            market_id = MARKET_MAP[market_str]

            # timestamps
            ts_ms = rec.get("ts_ms")
            window_start = rec.get("window_start")
            if ts_ms is None or window_start is None:
                _drop("missing_ts_or_window", f"ts_ms={ts_ms!r} window_start={window_start!r} keys={list(rec.keys())}")
                continue
            try:
                ts_ms = int(ts_ms)
                window_start = int(window_start)
            except (TypeError, ValueError):
                _drop("ts_parse_error", f"ts_ms={ts_ms!r} window_start={window_start!r}")
                continue

            ts_s = ts_ms // 1000
            cycle_len     = CYCLE_LEN_MAP.get(market_str, 900)
            cycle_end_ts  = window_start + cycle_len
            time_remaining = cycle_end_ts - ts_s

            if time_remaining < 0:
                # Observação APÓS resolução do ciclo — descarta
                _drop("time_remaining_negative",
                      f"tr={time_remaining} market={market_str}")
                continue
            # Obs. pré-ciclo (tr > cycle_len) são mantidas; o simulador
            # nunca entra em trade nelas pois t_max < cycle_len < tr.

            # probabilities
            probs = _get_probs(rec)
            if probs is None:
                _drop("prob_missing_or_invalid",
                      f"derived={rec.get('derived')} yes={rec.get('yes')} no={rec.get('no')}")
                continue
            prob_up, prob_down = probs

            # derived features (pré-computa 1x aqui, nunca no sim)
            best_prob = max(prob_up, prob_down)
            best_side = 0 if prob_up >= prob_down else 1
            # entry_eligible usa a janela mais ampla do timeframe
            tf = get_timeframe(market_str)
            wins = T_WIN_MAP[tf]
            elig_min = min(w[0] for w in wins)
            elig_max = max(w[1] for w in wins)
            entry_eligible = 1 if elig_min <= time_remaining <= elig_max else 0

            overround = _get_overround(rec)
            lat = _get_latency(rec)
            if lat > 0:
                latencies.append(lat)

            err_flag = 1 if rec.get("err") else 0

            key = (market_id, cycle_end_ts, ts_s)
            rows[key] = {
                "ts_s":           ts_s,
                "market_id":      market_id,
                "cycle_end_ts":   cycle_end_ts,
                "time_remaining": time_remaining,
                "prob_up":        prob_up,
                "prob_down":      prob_down,
                "best_prob":      best_prob,
                "best_side":      best_side,
                "entry_eligible": entry_eligible,
                "overround":      overround,
                "err_flag":       err_flag,
            }

    if not rows:
        log.warning("Nenhuma linha válida em %s", raw_path)
        if debug and drop_examples:
            for reason, examples in drop_examples.items():
                log.warning("  [%s] x%d — ex: %s", reason,
                            drop_reasons[reason], examples[0])
        elif drop_reasons:
            log.warning("  Motivos: %s", drop_reasons)
        return {"file": str(raw_path), "kept": 0, "dropped": dropped,
                "drop_pct": 100.0, "latency_p50": 0.0, "latency_p99": 0.0,
                "drop_reasons": drop_reasons}

    # Construir arrays a partir do dict dedup (já deduplicado por key)
    vals = list(rows.values())
    n = len(vals)

    arrays = {
        "ts_s":           np.array([r["ts_s"]           for r in vals], dtype=np.int64),
        "market_id":      np.array([r["market_id"]      for r in vals], dtype=np.int8),
        "cycle_end_ts":   np.array([r["cycle_end_ts"]   for r in vals], dtype=np.int64),
        "time_remaining": np.array([r["time_remaining"] for r in vals], dtype=np.int32),
        "prob_up":        np.array([r["prob_up"]        for r in vals], dtype=np.float32),
        "prob_down":      np.array([r["prob_down"]      for r in vals], dtype=np.float32),
        "best_prob":      np.array([r["best_prob"]      for r in vals], dtype=np.float32),
        "best_side":      np.array([r["best_side"]      for r in vals], dtype=np.int8),
        "entry_eligible": np.array([r["entry_eligible"] for r in vals], dtype=np.int8),
        "overround":      np.array([r["overround"]      for r in vals], dtype=np.float32),
        "err_flag":       np.array([r["err_flag"]       for r in vals], dtype=np.int8),
    }

    table = pa.table(arrays, schema=LEAN_SCHEMA)
    pq.write_table(table, str(out_path), compression="snappy")

    total = n + dropped
    drop_pct = 100.0 * dropped / max(total, 1)
    lat_arr = np.array(latencies, dtype=np.float32) if latencies else np.array([0.0])
    stats = {
        "file":       str(raw_path),
        "kept":       n,
        "dropped":    dropped,
        "drop_pct":   round(drop_pct, 2),
        "latency_p50": float(np.percentile(lat_arr, 50)),
        "latency_p99": float(np.percentile(lat_arr, 99)),
        "drop_reasons": drop_reasons,
    }
    log.info("CLEAN %s → kept=%d dropped=%d (%.1f%%)",
             raw_path.name, n, dropped, drop_pct)
    return stats
