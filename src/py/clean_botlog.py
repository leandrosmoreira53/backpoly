"""
clean_botlog.py — Adaptador para o log de execução do bot.

O bot grava um JSONL único por dia com todos os 4 mercados misturados.
Esse módulo normaliza o formato do bot para o mesmo Parquet lean
que clean.py produz — assim o resto do pipeline (pack, gridsearch,
validate) funciona sem nenhuma alteração.

Diferenças em relação ao clean.py:

  Campo bot           →  Campo pipeline
  ─────────────────────────────────────
  ts    (segundos)    →  ts_s           (sem divisão)
  market "btc"        →  market_id=0    (via _BOT_TO_MARKET)
  cycle_end_ts        →  cycle_end_ts   (direto; window_start = cycle_end_ts - 900)
  yes_price / no_price →  prob_up / prob_down  (campos flat, sem nested)
  time_to_expiry /
  t_remaining         →  time_remaining (fallback: cycle_end_ts - ts_s)
  yes_price + no_price →  overround
  action == ORDER_FAILED → err_flag=1

Linhas descartadas:
  - yes_price ou no_price ausentes/null  (ex: NEW_CYCLE)
  - market não mapeado
  - time_remaining fora de [0, 900]
  - prob fora de [0.001, 0.999]

Deduplicação:
  Chave (market_id, cycle_end_ts, ts_s) — mantém a ÚLTIMA linha
  (mesmo comportamento que clean.py; elimina múltiplas linhas de
  PLACING_ORDER / ORDER_PLACED / TIMEOUT no mesmo segundo).
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import orjson as _json_lib
    _LOADS = _json_lib.loads
except ImportError:
    _LOADS = json.loads  # type: ignore

from src.py.config import MARKET_MAP, CYCLE_LEN_S, T_MIN, T_MAX
from src.py.clean  import LEAN_SCHEMA

log = logging.getLogger(__name__)

# Mapeamento: nome curto do bot → chave do MARKET_MAP
_BOT_TO_MARKET: dict[str, str] = {
    "btc": "BTC15m",
    "eth": "ETH15m",
    "sol": "SOL15m",
    "xrp": "XRP15m",
}

# action → err_flag=1
_ERR_ACTIONS = {"ORDER_FAILED", "TIMEOUT_CANCEL"}


def _parse_probs(rec: dict[str, Any]) -> tuple[float, float] | None:
    """Lê yes_price / no_price do formato flat do bot."""
    pu  = rec.get("yes_price")
    pd_ = rec.get("no_price")
    if pu is None or pd_ is None:
        return None
    try:
        pu, pd_ = float(pu), float(pd_)
    except (TypeError, ValueError):
        return None
    if not (0.001 <= pu <= 0.999) or not (0.001 <= pd_ <= 0.999):
        return None
    return pu, pd_


def clean_botlog_file(raw_path: str | Path, out_dir: str | Path) -> dict:
    """
    Processa um arquivo JSONL do bot (todos os mercados, um por linha).

    Gera um arquivo Parquet por mercado por data em out_dir:
        <out_dir>/YYYY-MM-DD.BTC15m.parquet
        <out_dir>/YYYY-MM-DD.ETH15m.parquet
        ...

    Retorna dict com estatísticas globais.
    """
    raw_path = Path(raw_path)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # rows[market_name][date_str][(market_id, cycle_end_ts, ts_s)] = row_dict
    # Usamos dois níveis de agrupamento para separar data e mercado.
    rows: dict[str, dict[str, dict[tuple, dict]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    dropped = 0

    with open(raw_path, "rb") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = _LOADS(line)
            except Exception:
                dropped += 1
                continue

            # market
            mkt_raw = rec.get("market", "")
            mkt_full = _BOT_TO_MARKET.get(str(mkt_raw).lower())
            if mkt_full is None or mkt_full not in MARKET_MAP:
                dropped += 1
                continue
            market_id = MARKET_MAP[mkt_full]

            # timestamps
            ts_s = rec.get("ts")
            cycle_end_ts = rec.get("cycle_end_ts")
            if ts_s is None or cycle_end_ts is None:
                dropped += 1
                continue
            try:
                ts_s         = int(ts_s)
                cycle_end_ts = int(cycle_end_ts)
            except (TypeError, ValueError):
                dropped += 1
                continue

            # time_remaining: usa campo explícito se disponível, senão calcula
            tr_raw = rec.get("time_to_expiry") or rec.get("t_remaining")
            if tr_raw is not None:
                try:
                    time_remaining = int(tr_raw)
                except (TypeError, ValueError):
                    time_remaining = cycle_end_ts - ts_s
            else:
                time_remaining = cycle_end_ts - ts_s

            if not (0 <= time_remaining <= CYCLE_LEN_S):
                dropped += 1
                continue

            # probabilidades (campos flat do bot)
            probs = _parse_probs(rec)
            if probs is None:
                dropped += 1
                continue
            prob_up, prob_down = probs

            # features derivadas
            best_prob      = max(prob_up, prob_down)
            best_side      = 0 if prob_up >= prob_down else 1
            entry_eligible = 1 if T_MIN <= time_remaining <= T_MAX else 0
            overround      = round(prob_up + prob_down, 4)
            err_flag       = 1 if rec.get("action") in _ERR_ACTIONS else 0

            # data do tick para nomear o arquivo parquet
            date_str = datetime.fromtimestamp(ts_s, tz=timezone.utc).strftime("%Y-%m-%d")

            key = (market_id, cycle_end_ts, ts_s)
            rows[mkt_full][date_str][key] = {
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

    # -------------------------------------------------------------------------
    # Escreve um parquet por (mercado, data)
    # -------------------------------------------------------------------------
    total_kept = 0
    written_files: list[str] = []

    for mkt_full, date_map in rows.items():
        for date_str, row_map in date_map.items():
            if not row_map:
                continue

            vals = list(row_map.values())
            arrays = {
                "ts_s":           np.array([r["ts_s"]           for r in vals], dtype=np.int64),
                "market_id":      np.array([r["market_id"]      for r in vals], dtype=np.int8),
                "cycle_end_ts":   np.array([r["cycle_end_ts"]   for r in vals], dtype=np.int64),
                "time_remaining": np.array([r["time_remaining"] for r in vals], dtype=np.int16),
                "prob_up":        np.array([r["prob_up"]        for r in vals], dtype=np.float32),
                "prob_down":      np.array([r["prob_down"]      for r in vals], dtype=np.float32),
                "best_prob":      np.array([r["best_prob"]      for r in vals], dtype=np.float32),
                "best_side":      np.array([r["best_side"]      for r in vals], dtype=np.int8),
                "entry_eligible": np.array([r["entry_eligible"] for r in vals], dtype=np.int8),
                "overround":      np.array([r["overround"]      for r in vals], dtype=np.float32),
                "err_flag":       np.array([r["err_flag"]       for r in vals], dtype=np.int8),
            }

            table    = pa.table(arrays, schema=LEAN_SCHEMA)
            out_file = out_dir / f"{date_str}.{mkt_full}.parquet"
            pq.write_table(table, str(out_file), compression="snappy")

            total_kept     += len(vals)
            written_files.append(str(out_file))
            log.info("BOTLOG %s → %s  (%d obs)", raw_path.name, out_file.name, len(vals))

    total   = total_kept + dropped
    drop_pct = 100.0 * dropped / max(total, 1)

    log.info(
        "BOTLOG %s: kept=%d dropped=%d (%.1f%%)  arquivos=%d",
        raw_path.name, total_kept, dropped, drop_pct, len(written_files),
    )
    return {
        "file":         str(raw_path),
        "kept":         total_kept,
        "dropped":      dropped,
        "drop_pct":     round(drop_pct, 2),
        "written":      written_files,
        "latency_p50":  0.0,
        "latency_p99":  0.0,
    }
