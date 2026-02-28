"""
pack.py — FASE 2: Parquet lean → month.npz + cycles.npz.

Otimizações:
  - Polars para sort/groupby (10-100x vs Pandas)
  - Conversão Arrow → NumPy zero-copy onde possível
  - np.ascontiguousarray garante layout C (cache-friendly para Cython)
  - Offsets por ciclo pré-calculados (O(1) acesso por ciclo, sem groupby no sim)
  - Todos os dtypes mínimos (float32, int16, int8) → metade RAM → mais cache hits
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import polars as pl
    _USE_POLARS = True
except ImportError:
    _USE_POLARS = False
    import pyarrow.parquet as pq
    import pyarrow as pa
    logging.warning("Polars não encontrado; usando PyArrow (mais lento). "
                    "Instale com: pip install polars")

from src.py.config import (
    LEAN_DIR, MONTH_NPZ, CYCLES_NPZ, META_JSON, MARKET_MAP,
    MARKET_ID_TO_NAME, TRAIN_DAYS, OOS_DAYS,
)

log = logging.getLogger(__name__)


def _ts_to_date(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _compute_cycle_ids_split(
    cycle_end_ts: np.ndarray,
    train_days: int,
    oos_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Divide cycle_ids em treino e OOS por data (não por linha).
    Retorna (train_cycle_ids, oos_cycle_ids) como arrays int32.
    """
    # datas únicas ordenadas
    unique_dates = sorted({_ts_to_date(int(ts)) for ts in cycle_end_ts})
    total_days = len(unique_dates)

    if total_days < train_days + oos_days:
        log.warning(
            "Apenas %d dias disponíveis (treino=%d oos=%d). "
            "Ajuste TRAIN_DAYS/OOS_DAYS.", total_days, train_days, oos_days
        )
        oos_days   = max(1, total_days - train_days)
        train_days = total_days - oos_days

    train_dates = set(unique_dates[:train_days])
    oos_dates   = set(unique_dates[train_days: train_days + oos_days])

    cycle_dates = np.array([_ts_to_date(int(ts)) for ts in cycle_end_ts])
    all_ids = np.arange(len(cycle_end_ts), dtype=np.int32)

    train_ids = all_ids[np.isin(cycle_dates, list(train_dates))]
    oos_ids   = all_ids[np.isin(cycle_dates, list(oos_dates))]

    return train_ids, oos_ids


def pack(lean_dir: str | None = None) -> dict[str, Any]:
    """
    Lê todos os Parquet lean, consolida, ordena e gera month.npz + cycles.npz.
    Retorna stats do pack.
    """
    lean_dir = lean_dir or LEAN_DIR
    parquet_files = sorted(Path(lean_dir).glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"Nenhum arquivo .parquet encontrado em {lean_dir}. "
            "Execute run_clean.py primeiro."
        )

    log.info("PACK: lendo %d arquivos Parquet...", len(parquet_files))

    # ------------------------------------------------------------------
    # Leitura e sort com Polars (ou fallback PyArrow)
    # ------------------------------------------------------------------
    if _USE_POLARS:
        df = pl.concat([
            pl.scan_parquet(str(f)) for f in parquet_files
        ]).collect()

        # Sort: (market_id, cycle_end_ts, ts_s)
        df = df.sort(["market_id", "cycle_end_ts", "ts_s"])

        # cycle_id sequencial por (market_id, cycle_end_ts)
        df = df.with_columns(
            pl.struct(["market_id", "cycle_end_ts"])
              .rank("dense")
              .cast(pl.Int32)
              .sub(1)
              .alias("cycle_id")
        )

        # Extrair arrays (to_numpy é zero-copy para tipos compatíveis em Polars)
        def col(name: str) -> np.ndarray:
            return df[name].to_numpy()

    else:
        import pyarrow.parquet as pq
        tables = [pq.read_table(str(f)) for f in parquet_files]
        import pyarrow as pa
        tbl = pa.concat_tables(tables)
        # Ordenar via numpy (lento, mas funcional)
        import pandas as pd
        df_pd = tbl.to_pandas()
        df_pd.sort_values(["market_id", "cycle_end_ts", "ts_s"], inplace=True)
        df_pd["cycle_id"] = (
            df_pd.groupby(["market_id", "cycle_end_ts"], sort=False)
                 .ngroup()
                 .astype(np.int32)
        )
        df = df_pd

        def col(name: str) -> np.ndarray:
            return df[name].to_numpy()

    log.info("PACK: %d linhas carregadas e ordenadas.", len(df))

    # ------------------------------------------------------------------
    # Arrays do month.npz (contíguos, dtype mínimo)
    # ------------------------------------------------------------------
    N = len(df)

    ts_s           = np.ascontiguousarray(col("ts_s"),           dtype=np.int64)
    market_id      = np.ascontiguousarray(col("market_id"),      dtype=np.int8)
    cycle_id       = np.ascontiguousarray(col("cycle_id"),       dtype=np.int32)
    time_remaining = np.ascontiguousarray(col("time_remaining"), dtype=np.int16)
    prob_up        = np.ascontiguousarray(col("prob_up"),        dtype=np.float32)
    prob_down      = np.ascontiguousarray(col("prob_down"),      dtype=np.float32)
    best_prob      = np.ascontiguousarray(col("best_prob"),      dtype=np.float32)
    best_side      = np.ascontiguousarray(col("best_side"),      dtype=np.int8)
    entry_eligible = np.ascontiguousarray(col("entry_eligible"), dtype=np.int8)
    err_flag       = np.ascontiguousarray(col("err_flag"),       dtype=np.int8)

    # ------------------------------------------------------------------
    # Arrays do cycles.npz — offsets por ciclo
    # ------------------------------------------------------------------
    C = int(cycle_id.max()) + 1  # total de ciclos

    cycle_start_idx  = np.zeros(C, dtype=np.int32)
    cycle_end_idx    = np.zeros(C, dtype=np.int32)
    cycle_market_id  = np.zeros(C, dtype=np.int8)
    cycle_end_ts_arr = np.zeros(C, dtype=np.int64)

    # Primeiro passe: preencher offsets (cycle_id já está ordenado por construção)
    prev_cid = -1
    for i in range(N):
        cid = cycle_id[i]
        if cid != prev_cid:
            cycle_start_idx[cid] = i
            cycle_market_id[cid] = market_id[i]
            if _USE_POLARS:
                cycle_end_ts_arr[cid] = int(df["cycle_end_ts"][i])
            else:
                cycle_end_ts_arr[cid] = int(col("cycle_end_ts")[i])
            prev_cid = cid
        cycle_end_idx[cid] = i + 1  # exclusive end (atualizado a cada linha)

    # Correção: cycle_end_ts via array cycle_end_ts de todas as linhas
    # (mais eficiente que acesso linha a linha acima)
    if _USE_POLARS:
        cycle_end_ts_col = np.ascontiguousarray(
            df["cycle_end_ts"].to_numpy(), dtype=np.int64
        )
    else:
        cycle_end_ts_col = np.ascontiguousarray(col("cycle_end_ts"), dtype=np.int64)

    # Reestrutura cycle_end_ts_arr usando primeiro valor de cada grupo
    np.minimum.at(cycle_end_ts_arr, cycle_id, cycle_end_ts_col)
    # Isso não funciona diretamente — usar indexação com start_idx
    cycle_end_ts_arr = cycle_end_ts_col[cycle_start_idx]

    # ------------------------------------------------------------------
    # Split treino / OOS
    # ------------------------------------------------------------------
    from src.py.config import TRAIN_DAYS, OOS_DAYS
    train_ids, oos_ids = _compute_cycle_ids_split(
        cycle_end_ts_arr, TRAIN_DAYS, OOS_DAYS
    )

    log.info("PACK: %d ciclos totais, %d treino, %d OOS",
             C, len(train_ids), len(oos_ids))

    # ------------------------------------------------------------------
    # Salvar arquivos
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(MONTH_NPZ), exist_ok=True)

    np.savez_compressed(
        MONTH_NPZ,
        ts_s=ts_s,
        market_id=market_id,
        cycle_id=cycle_id,
        time_remaining=time_remaining,
        prob_up=prob_up,
        prob_down=prob_down,
        best_prob=best_prob,
        best_side=best_side,
        entry_eligible=entry_eligible,
        err_flag=err_flag,
    )
    log.info("Salvo: %s  (N=%d)", MONTH_NPZ, N)

    np.savez_compressed(
        CYCLES_NPZ,
        cycle_start_idx=cycle_start_idx,
        cycle_end_idx=cycle_end_idx,
        cycle_market_id=cycle_market_id,
        cycle_end_ts=cycle_end_ts_arr,
        train_cycle_ids=train_ids,
        oos_cycle_ids=oos_ids,
        n_cycles=np.array([C], dtype=np.int32),
    )
    log.info("Salvo: %s  (C=%d)", CYCLES_NPZ, C)

    # ------------------------------------------------------------------
    # Atualizar meta.json
    # ------------------------------------------------------------------
    unique_dates = sorted({_ts_to_date(int(ts)) for ts in cycle_end_ts_arr})
    stats: dict[str, Any] = {
        "n_rows":         int(N),
        "n_cycles":       int(C),
        "n_train_cycles": int(len(train_ids)),
        "n_oos_cycles":   int(len(oos_ids)),
        "date_min":       unique_dates[0] if unique_dates else None,
        "date_max":       unique_dates[-1] if unique_dates else None,
        "n_dates":        len(unique_dates),
        "markets":        MARKET_MAP,
        "by_market": {
            MARKET_ID_TO_NAME.get(mid, str(mid)): int((market_id == mid).sum())
            for mid in range(len(MARKET_MAP))
        },
    }

    meta: dict = {}
    if os.path.exists(META_JSON):
        with open(META_JSON) as f:
            meta = json.load(f)
    meta["pack_stats"] = stats
    with open(META_JSON, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("meta.json atualizado.")

    return stats
