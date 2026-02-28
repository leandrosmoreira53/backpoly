# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False
"""
sim_core.pyx — Simulador Polymarket 15m em Cython + OpenMP.

Otimizações máximas:
  - Typed memoryviews (float32_t[::1], int16_t[::1]) → zero overhead de acesso
  - prange + OpenMP → paralelismo real em múltiplos cores
  - nogil no loop principal → GIL liberado durante simulação
  - -O3 -march=native -ffast-math → SIMD automático pelo gcc/clang
  - Arrays de saída pré-alocados (np.empty) → sem alloc no loop
  - Zero criação de objetos Python no loop quente

Compilação:
    python setup.py build_ext --inplace

Uso:
    from src.cy.sim_core import run_cycles, run_cycles_by_market
"""

import numpy as np
cimport numpy as cnp
from cython.parallel import prange

# Tipos declarados em sim_core.pxd (incluído automaticamente pelo Cython):
#   i8=int8, i16=int16, i32=int32, i64=int64, f32=float32

# Inicializa API NumPy C (necessário para memoryviews)
cnp.import_array()


# ---------------------------------------------------------------------------
# Função principal: simula um subconjunto de ciclos
# ---------------------------------------------------------------------------
def run_cycles(
    i16[::1]  time_remaining,     # [N] segundos restantes por linha
    f32[::1]  prob_up,            # [N] probabilidade UP por linha
    i32[::1]  cycle_start_idx,    # [C] offset início de cada ciclo
    i32[::1]  cycle_end_idx,      # [C] offset fim (exclusive) de cada ciclo
    i32[::1]  cycle_ids,          # [nc] quais ciclos avaliar (treino ou OOS)
    float     prob_entry_min,
    int       t_min,
    int       t_max,
    float     size_shares,
    int       n_threads = 1,
):
    """
    Executa backtest nos ciclos especificados por cycle_ids.

    Retorna:
        pnl_out   : float32[nc]  — PnL por ciclo
        entered   : int8[nc]     — 1 se houve trade, 0 caso contrário
        entry_prob: float32[nc]  — probabilidade de entrada (0 se sem trade)
    """
    cdef:
        int nc = cycle_ids.shape[0]
        int ci, cid, i, start, end
        i16 tr
        f32 p, ep, last_p
        bint entered_flag

        # Arrays de saída pré-alocados
        cnp.ndarray[f32, ndim=1] pnl_nd  = np.empty(nc, dtype=np.float32)
        cnp.ndarray[i8,  ndim=1] ent_nd  = np.empty(nc, dtype=np.int8)
        cnp.ndarray[f32, ndim=1] ep_nd   = np.empty(nc, dtype=np.float32)

        # Memoryviews tipadas sobre os arrays de saída
        f32[::1] pnl_mv = pnl_nd
        i8[::1]  ent_mv = ent_nd
        f32[::1] ep_mv  = ep_nd

    with nogil:
        for ci in prange(nc, schedule='static', num_threads=n_threads):
            cid   = cycle_ids[ci]
            start = cycle_start_idx[cid]
            end   = cycle_end_idx[cid]

            entered_flag = False
            ep           = 0.0
            last_p       = 0.5  # padrão neutro se ciclo sem obs final

            for i in range(start, end):
                tr = time_remaining[i]
                p  = prob_up[i]

                # Dados ordenados por ts_s ASC → time_remaining DESC.
                # Última iteração = observação mais próxima da resolução.
                # Atualiza sempre: ao sair do loop, last_p = última obs.
                last_p = p

                # Tenta entrar se ainda não entrou e condições ok
                if (not entered_flag
                        and t_min <= tr <= t_max
                        and p >= prob_entry_min):
                    entered_flag = True
                    ep = p

            # Calcula PnL do ciclo
            if entered_flag:
                ent_mv[ci]  = 1
                ep_mv[ci]   = ep
                if last_p > 0.5:
                    # Outcome UP → ganho: (1 - entry_price) × size
                    pnl_mv[ci] = (1.0 - ep) * size_shares
                else:
                    # Outcome DOWN → perda: entry_price × size
                    pnl_mv[ci] = -ep * size_shares
            else:
                ent_mv[ci]  = 0
                ep_mv[ci]   = 0.0
                pnl_mv[ci]  = 0.0

    return pnl_nd, ent_nd, ep_nd


# ---------------------------------------------------------------------------
# Variante: retorna PnL por mercado (4 valores) além do total
# ---------------------------------------------------------------------------
def run_cycles_by_market(
    i16[::1]  time_remaining,
    f32[::1]  prob_up,
    i32[::1]  cycle_start_idx,
    i32[::1]  cycle_end_idx,
    i8[::1]   cycle_market_id,   # [C] mercado de cada ciclo
    i32[::1]  cycle_ids,
    float     prob_entry_min,
    int       t_min,
    int       t_max,
    float     size_shares,
    int       n_threads = 1,
    int       n_markets = 4,
):
    """
    Como run_cycles, mas também acumula PnL por mercado.

    Retorna:
        pnl_out         : float32[nc]
        entered         : int8[nc]
        entry_prob      : float32[nc]
        pnl_per_market  : float64[n_markets]
        trades_per_market: int32[n_markets]
    """
    cdef:
        int nc = cycle_ids.shape[0]
        int ci, cid, i, start, end, mid
        i16 tr
        f32 p, ep, last_p
        bint entered_flag

        cnp.ndarray[f32,  ndim=1] pnl_nd  = np.empty(nc, dtype=np.float32)
        cnp.ndarray[i8,   ndim=1] ent_nd  = np.empty(nc, dtype=np.int8)
        cnp.ndarray[f32,  ndim=1] ep_nd   = np.empty(nc, dtype=np.float32)
        # Por mercado: acumulado no Python após o prange (evita race condition)
        i8[::1]   mkt_mv  = cycle_market_id
        f32[::1]  pnl_mv  = pnl_nd
        i8[::1]   ent_mv  = ent_nd
        f32[::1]  ep_mv   = ep_nd

    with nogil:
        for ci in prange(nc, schedule='static', num_threads=n_threads):
            cid   = cycle_ids[ci]
            start = cycle_start_idx[cid]
            end   = cycle_end_idx[cid]

            entered_flag = False
            ep           = 0.0
            last_p       = 0.5

            for i in range(start, end):
                tr = time_remaining[i]
                p  = prob_up[i]
                last_p = p  # última iteração = obs mais próxima da resolução
                if (not entered_flag
                        and t_min <= tr <= t_max
                        and p >= prob_entry_min):
                    entered_flag = True
                    ep = p

            if entered_flag:
                ent_mv[ci] = 1
                ep_mv[ci]  = ep
                if last_p > 0.5:
                    pnl_mv[ci] = (1.0 - ep) * size_shares
                else:
                    pnl_mv[ci] = -ep * size_shares
            else:
                ent_mv[ci] = 0
                ep_mv[ci]  = 0.0
                pnl_mv[ci] = 0.0

    # Acumula por mercado em Python (seguro, fora do prange)
    pnl_per_market    = np.zeros(n_markets, dtype=np.float64)
    trades_per_market = np.zeros(n_markets, dtype=np.int32)
    for ci in range(nc):
        cid = cycle_ids[ci]
        mid = cycle_market_id[cid]
        pnl_per_market[mid]    += pnl_nd[ci]
        trades_per_market[mid] += ent_nd[ci]

    return pnl_nd, ent_nd, ep_nd, pnl_per_market, trades_per_market


# ---------------------------------------------------------------------------
# Variante com best_side: opera pelo lado com maior prob (UP ou DOWN)
# ---------------------------------------------------------------------------
def run_cycles_best_side(
    i16[::1]  time_remaining,
    f32[::1]  best_prob,          # max(prob_up, prob_down) pré-calculado no PACK
    i8[::1]   best_side,          # 0=UP, 1=DOWN
    f32[::1]  prob_up,            # para determinar outcome
    i32[::1]  cycle_start_idx,
    i32[::1]  cycle_end_idx,
    i32[::1]  cycle_ids,
    float     prob_entry_min,
    int       t_min,
    int       t_max,
    float     size_shares,
    int       n_threads = 1,
):
    """
    Variante que opera pelo lado com maior probabilidade (best_side).
    best_prob e best_side são pré-calculados no PACK (zero overhead no sim).

    PnL:
        side=UP,   outcome=UP   → +( 1 - entry_prob) × size
        side=UP,   outcome=DOWN → -entry_prob × size
        side=DOWN, outcome=DOWN → +( 1 - entry_prob) × size
        side=DOWN, outcome=UP   → -entry_prob × size
    """
    cdef:
        int nc = cycle_ids.shape[0]
        int ci, cid, i, start, end
        i16 tr
        f32 p, ep, last_pu
        i8 side, entry_side
        bint entered_flag
        bint outcome_matches

        cnp.ndarray[f32, ndim=1] pnl_nd = np.empty(nc, dtype=np.float32)
        cnp.ndarray[i8,  ndim=1] ent_nd = np.empty(nc, dtype=np.int8)
        cnp.ndarray[f32, ndim=1] ep_nd  = np.empty(nc, dtype=np.float32)
        cnp.ndarray[i8,  ndim=1] sd_nd  = np.empty(nc, dtype=np.int8)

        f32[::1] pnl_mv = pnl_nd
        i8[::1]  ent_mv = ent_nd
        f32[::1] ep_mv  = ep_nd
        i8[::1]  sd_mv  = sd_nd

    with nogil:
        for ci in prange(nc, schedule='static', num_threads=n_threads):
            cid   = cycle_ids[ci]
            start = cycle_start_idx[cid]
            end   = cycle_end_idx[cid]

            entered_flag = False
            ep           = 0.0
            last_pu      = 0.5
            entry_side   = 0

            for i in range(start, end):
                tr   = time_remaining[i]
                p    = best_prob[i]
                side = best_side[i]
                last_pu = prob_up[i]  # última iteração = obs mais próxima da resolução
                if (not entered_flag
                        and t_min <= tr <= t_max
                        and p >= prob_entry_min):
                    entered_flag = True
                    ep         = p
                    entry_side = side

            if entered_flag:
                ent_mv[ci] = 1
                ep_mv[ci]  = ep
                sd_mv[ci]  = entry_side
                # outcome_up = last_pu > 0.5
                if entry_side == 0:   # operamos UP
                    outcome_matches = last_pu > 0.5
                else:                 # operamos DOWN
                    outcome_matches = last_pu <= 0.5
                if outcome_matches:
                    pnl_mv[ci] = (1.0 - ep) * size_shares
                else:
                    pnl_mv[ci] = -ep * size_shares
            else:
                ent_mv[ci] = 0
                ep_mv[ci]  = 0.0
                sd_mv[ci]  = -1
                pnl_mv[ci] = 0.0

    return pnl_nd, ent_nd, ep_nd, sd_nd
