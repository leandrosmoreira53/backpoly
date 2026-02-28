# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False
"""
sim_core.pyx — Simulador Polymarket 15m em Cython + OpenMP.

Parâmetros otimizáveis pelo grid search:
  - prob_entry_min  : prob mínima para entrar no trade
  - t_min / t_max   : janela de tempo restante (segundos) para entrar
  - stop_loss_delta : quanto a prob pode cair antes de sair (0 = hold to end)

Modelo de PnL:
  Sem stop (stop_loss_delta=0):
    outcome UP   → +(1 − entry_prob) × size
    outcome DOWN → −entry_prob × size

  Com stop (stop_loss_delta > 0):
    Se prob_up cair (entry_prob − stop_loss_delta) APÓS a entrada → sai:
    PnL = −stop_loss_delta × size   (perda limitada)
    Caso contrário → PnL igual ao sem stop.

Compilação:
    python setup.py build_ext --inplace
"""

import numpy as np
cimport numpy as cnp
from cython.parallel import prange

# Tipos declarados em sim_core.pxd (incluído automaticamente pelo Cython):
#   i8=int8, i16=int16, i32=int32, i64=int64, f32=float32

cnp.import_array()


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------
def run_cycles(
    i16[::1]  time_remaining,
    f32[::1]  prob_up,
    i32[::1]  cycle_start_idx,
    i32[::1]  cycle_end_idx,
    i32[::1]  cycle_ids,
    float     prob_entry_min,
    int       t_min,
    int       t_max,
    float     size_shares,
    float     stop_loss_delta = 0.0,
    int       n_threads       = 1,
):
    """
    Backtest de um subconjunto de ciclos com stop loss opcional.

    Args:
        time_remaining   : int16[N]  — segundos restantes por linha
        prob_up          : float32[N]
        cycle_start_idx  : int32[C]  — offset início de cada ciclo
        cycle_end_idx    : int32[C]  — offset fim (exclusive)
        cycle_ids        : int32[nc] — ciclos a avaliar (treino ou OOS)
        prob_entry_min   : limiar de entrada
        t_min, t_max     : janela de tempo restante para entrada
        size_shares      : tamanho da posição
        stop_loss_delta  : 0.0 = hold to end;
                           ex: 0.10 → sai se prob cair 10pp após entrada
        n_threads        : threads OpenMP

    Retorna (float32[nc], int8[nc], float32[nc], int8[nc]):
        pnl_out       — PnL por ciclo
        entered       — 1 se houve trade
        entry_prob    — prob de entrada (0 se sem trade)
        stop_hit_out  — 1 se stop foi acionado
    """
    cdef:
        int nc    = cycle_ids.shape[0]
        int ci, cid, i, start, end
        i16 tr
        f32 p, ep, last_p, stop_level
        bint entered_flag, stop_hit

        cnp.ndarray[f32, ndim=1] pnl_nd  = np.empty(nc, dtype=np.float32)
        cnp.ndarray[i8,  ndim=1] ent_nd  = np.empty(nc, dtype=np.int8)
        cnp.ndarray[f32, ndim=1] ep_nd   = np.empty(nc, dtype=np.float32)
        cnp.ndarray[i8,  ndim=1] sth_nd  = np.empty(nc, dtype=np.int8)

        f32[::1] pnl_mv = pnl_nd
        i8[::1]  ent_mv = ent_nd
        f32[::1] ep_mv  = ep_nd
        i8[::1]  sth_mv = sth_nd

    with nogil:
        for ci in prange(nc, schedule='static', num_threads=n_threads):
            cid         = cycle_ids[ci]
            start       = cycle_start_idx[cid]
            end         = cycle_end_idx[cid]
            entered_flag = False
            stop_hit     = False
            ep           = 0.0
            last_p       = 0.5
            stop_level   = 0.0

            for i in range(start, end):
                tr = time_remaining[i]
                p  = prob_up[i]
                last_p = p  # última iteração = obs mais próxima da resolução

                if not entered_flag:
                    if t_min <= tr <= t_max and p >= prob_entry_min:
                        entered_flag = True
                        ep          = p
                        stop_level  = ep - stop_loss_delta
                elif stop_loss_delta > 0.0 and not stop_hit:
                    # Verifica stop: prob caiu abaixo do nível de stop?
                    if p <= stop_level:
                        stop_hit = True

            # ---- PnL do ciclo ----
            if entered_flag:
                ent_mv[ci] = 1
                ep_mv[ci]  = ep
                if stop_hit:
                    # Saiu no stop: perda limitada = delta × size
                    pnl_mv[ci] = -stop_loss_delta * size_shares
                    sth_mv[ci] = 1
                elif last_p > 0.5:
                    # Outcome UP → lucro
                    pnl_mv[ci] = (1.0 - ep) * size_shares
                    sth_mv[ci] = 0
                else:
                    # Outcome DOWN → perda total
                    pnl_mv[ci] = -ep * size_shares
                    sth_mv[ci] = 0
            else:
                ent_mv[ci]  = 0
                ep_mv[ci]   = 0.0
                pnl_mv[ci]  = 0.0
                sth_mv[ci]  = 0

    return pnl_nd, ent_nd, ep_nd, sth_nd


# ---------------------------------------------------------------------------
# Variante: PnL por mercado
# ---------------------------------------------------------------------------
def run_cycles_by_market(
    i16[::1]  time_remaining,
    f32[::1]  prob_up,
    i32[::1]  cycle_start_idx,
    i32[::1]  cycle_end_idx,
    i8[::1]   cycle_market_id,
    i32[::1]  cycle_ids,
    float     prob_entry_min,
    int       t_min,
    int       t_max,
    float     size_shares,
    float     stop_loss_delta = 0.0,
    int       n_threads       = 1,
    int       n_markets       = 4,
):
    """
    Como run_cycles, mas também acumula PnL e trades por mercado.

    Retorna (pnl[nc], entered[nc], entry_prob[nc], stop_hit[nc],
             pnl_per_market[n_markets], trades_per_market[n_markets])
    """
    cdef:
        int nc    = cycle_ids.shape[0]
        int ci, cid, i, start, end, mid
        i16 tr
        f32 p, ep, last_p, stop_level
        bint entered_flag, stop_hit

        cnp.ndarray[f32, ndim=1] pnl_nd  = np.empty(nc, dtype=np.float32)
        cnp.ndarray[i8,  ndim=1] ent_nd  = np.empty(nc, dtype=np.int8)
        cnp.ndarray[f32, ndim=1] ep_nd   = np.empty(nc, dtype=np.float32)
        cnp.ndarray[i8,  ndim=1] sth_nd  = np.empty(nc, dtype=np.int8)

        f32[::1] pnl_mv = pnl_nd
        i8[::1]  ent_mv = ent_nd
        f32[::1] ep_mv  = ep_nd
        i8[::1]  sth_mv = sth_nd

    with nogil:
        for ci in prange(nc, schedule='static', num_threads=n_threads):
            cid          = cycle_ids[ci]
            start        = cycle_start_idx[cid]
            end          = cycle_end_idx[cid]
            entered_flag = False
            stop_hit     = False
            ep           = 0.0
            last_p       = 0.5
            stop_level   = 0.0

            for i in range(start, end):
                tr = time_remaining[i]
                p  = prob_up[i]
                last_p = p

                if not entered_flag:
                    if t_min <= tr <= t_max and p >= prob_entry_min:
                        entered_flag = True
                        ep          = p
                        stop_level  = ep - stop_loss_delta
                elif stop_loss_delta > 0.0 and not stop_hit:
                    if p <= stop_level:
                        stop_hit = True

            if entered_flag:
                ent_mv[ci] = 1
                ep_mv[ci]  = ep
                sth_mv[ci] = 1 if stop_hit else 0
                if stop_hit:
                    pnl_mv[ci] = -stop_loss_delta * size_shares
                elif last_p > 0.5:
                    pnl_mv[ci] = (1.0 - ep) * size_shares
                else:
                    pnl_mv[ci] = -ep * size_shares
            else:
                ent_mv[ci]  = 0
                ep_mv[ci]   = 0.0
                pnl_mv[ci]  = 0.0
                sth_mv[ci]  = 0

    # Acumula por mercado fora do prange (sem race condition)
    pnl_per_market    = np.zeros(n_markets, dtype=np.float64)
    trades_per_market = np.zeros(n_markets, dtype=np.int32)
    for ci in range(nc):
        cid = cycle_ids[ci]
        mid = cycle_market_id[cid]
        pnl_per_market[mid]    += pnl_nd[ci]
        trades_per_market[mid] += ent_nd[ci]

    return pnl_nd, ent_nd, ep_nd, sth_nd, pnl_per_market, trades_per_market


# ---------------------------------------------------------------------------
# Variante: opera pelo lado com maior prob (UP ou DOWN)
# ---------------------------------------------------------------------------
def run_cycles_best_side(
    i16[::1]  time_remaining,
    f32[::1]  best_prob,
    i8[::1]   best_side,
    f32[::1]  prob_up,
    i32[::1]  cycle_start_idx,
    i32[::1]  cycle_end_idx,
    i32[::1]  cycle_ids,
    float     prob_entry_min,
    int       t_min,
    int       t_max,
    float     size_shares,
    float     stop_loss_delta = 0.0,
    int       n_threads       = 1,
):
    """
    Opera pelo lado com maior probabilidade (best_side pré-calculado no PACK).

    PnL:
      side=UP,   outcome=UP   → +(1 − entry_prob) × size
      side=UP,   outcome=DOWN → −entry_prob × size  (ou stop)
      side=DOWN, outcome=DOWN → +(1 − entry_prob) × size
      side=DOWN, outcome=UP   → −entry_prob × size  (ou stop)
    """
    cdef:
        int nc    = cycle_ids.shape[0]
        int ci, cid, i, start, end
        i16 tr
        f32 p, ep, last_pu, stop_level
        i8  side, entry_side
        bint entered_flag, stop_hit, outcome_matches

        cnp.ndarray[f32, ndim=1] pnl_nd = np.empty(nc, dtype=np.float32)
        cnp.ndarray[i8,  ndim=1] ent_nd = np.empty(nc, dtype=np.int8)
        cnp.ndarray[f32, ndim=1] ep_nd  = np.empty(nc, dtype=np.float32)
        cnp.ndarray[i8,  ndim=1] sd_nd  = np.empty(nc, dtype=np.int8)
        cnp.ndarray[i8,  ndim=1] sth_nd = np.empty(nc, dtype=np.int8)

        f32[::1] pnl_mv = pnl_nd
        i8[::1]  ent_mv = ent_nd
        f32[::1] ep_mv  = ep_nd
        i8[::1]  sd_mv  = sd_nd
        i8[::1]  sth_mv = sth_nd

    with nogil:
        for ci in prange(nc, schedule='static', num_threads=n_threads):
            cid          = cycle_ids[ci]
            start        = cycle_start_idx[cid]
            end          = cycle_end_idx[cid]
            entered_flag = False
            stop_hit     = False
            ep           = 0.0
            last_pu      = 0.5
            entry_side   = 0
            stop_level   = 0.0

            for i in range(start, end):
                tr   = time_remaining[i]
                p    = best_prob[i]
                side = best_side[i]
                last_pu = prob_up[i]

                if not entered_flag:
                    if t_min <= tr <= t_max and p >= prob_entry_min:
                        entered_flag = True
                        ep          = p
                        entry_side  = side
                        stop_level  = ep - stop_loss_delta
                elif stop_loss_delta > 0.0 and not stop_hit:
                    # Stop baseado em best_prob (o lado operado)
                    if best_prob[i] <= stop_level:
                        stop_hit = True

            if entered_flag:
                ent_mv[ci] = 1
                ep_mv[ci]  = ep
                sd_mv[ci]  = entry_side
                sth_mv[ci] = 1 if stop_hit else 0
                if stop_hit:
                    pnl_mv[ci] = -stop_loss_delta * size_shares
                else:
                    if entry_side == 0:
                        outcome_matches = last_pu > 0.5
                    else:
                        outcome_matches = last_pu <= 0.5
                    if outcome_matches:
                        pnl_mv[ci] = (1.0 - ep) * size_shares
                    else:
                        pnl_mv[ci] = -ep * size_shares
            else:
                ent_mv[ci]  = 0
                ep_mv[ci]   = 0.0
                sd_mv[ci]   = -1
                sth_mv[ci]  = 0
                pnl_mv[ci]  = 0.0

    return pnl_nd, ent_nd, ep_nd, sd_nd, sth_nd
