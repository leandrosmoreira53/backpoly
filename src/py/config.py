"""
config.py — Parâmetros globais do pipeline Polymarket 15m backtest.
Edite aqui antes de rodar qualquer fase.
"""
from __future__ import annotations
import os
import math

# ---------------------------------------------------------------------------
# Mercados suportados
# ---------------------------------------------------------------------------
MARKET_MAP: dict[str, int] = {
    "BTC15m": 0,
    "ETH15m": 1,
    "SOL15m": 2,
    "XRP15m": 3,
}
MARKET_ID_TO_NAME: dict[int, str] = {v: k for k, v in MARKET_MAP.items()}

# Duração do ciclo em segundos
CYCLE_LEN_S: int = 900  # 15 × 60

# ---------------------------------------------------------------------------
# Janela operacional padrão (usada fora do grid search)
# ---------------------------------------------------------------------------
T_MIN: int = 60    # não entrar nos últimos 60s (resolução muito próxima)
T_MAX: int = 240   # não entrar muito cedo (spread alto, menos informação)

# Tempo máximo para fill ser confirmado (reservado para fill real)
MAX_WAIT_FILL_S: int = 10

# Tamanho da posição em shares
SIZE_SHARES: float = 5.0

# ---------------------------------------------------------------------------
# Split temporal
# ---------------------------------------------------------------------------
TRAIN_DAYS: int = 21
OOS_DAYS:   int = 7

# ---------------------------------------------------------------------------
# Grid de parâmetros — 3 dimensões
# ---------------------------------------------------------------------------

# 1. prob_entry_min: limiar mínimo de probabilidade para entrar
#    Range 0.55 → 0.925 em steps de 0.025  (15 pontos)
PROB_GRID: list[float] = [round(0.55 + i * 0.025, 3) for i in range(16)]

# 2. Janelas de tempo restante (t_min, t_max) em segundos
#    Cada tupla = (entrada mínima em segundos antes do fim, máxima)
T_WIN_GRID: list[tuple[int, int]] = [
    (30,  120),   # entrada tardia  — mais certeza, menos tempo
    (60,  180),   # janela curta
    (60,  240),   # janela média (baseline original)
    (120, 300),   # janela ampla
    (180, 600),   # entrada antecipada — maior incerteza
]

# 3. Stop loss em pontos de probabilidade (0.0 = sem stop, hold to end)
#    Ex: 0.10 → sai se prob cair 10pp após entrada
STOP_LOSS_GRID: list[float] = [0.0, 0.05, 0.10, 0.15, 0.20]

# Total de combinações: 16 × 5 × 5 = 400 pontos
# Com Cython + ProcessPool → < 3s no total

# ---------------------------------------------------------------------------
# Score no grid
# ---------------------------------------------------------------------------
SCORE_MIN_TRADES: int = 30

# ---------------------------------------------------------------------------
# Paralelismo
# ---------------------------------------------------------------------------
# 0 = auto (usa os.cpu_count())
N_SIM_THREADS: int = 0
N_CLEAN_WORKERS: int = 0   # workers para CLEAN (ProcessPool)
N_GRID_WORKERS: int  = 0   # workers para grid search (ProcessPool)

def _auto(n: int) -> int:
    return os.cpu_count() or 4 if n == 0 else n

def get_sim_threads()   -> int: return _auto(N_SIM_THREADS)
def get_clean_workers() -> int: return _auto(N_CLEAN_WORKERS)
def get_grid_workers()  -> int: return _auto(N_GRID_WORKERS)

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
ROOT_DIR        = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_RAW_DIR    = os.path.join(ROOT_DIR, "data_raw")
BOT_LOGS_DIR    = os.path.join(DATA_RAW_DIR, "bot_logs")   # logs do bot de execução
DATA_CACHE_DIR  = os.path.join(ROOT_DIR, "data_cache")
LEAN_DIR        = os.path.join(DATA_CACHE_DIR, "lean")
MONTH_NPZ       = os.path.join(DATA_CACHE_DIR, "month.npz")
CYCLES_NPZ      = os.path.join(DATA_CACHE_DIR, "cycles.npz")
META_JSON       = os.path.join(DATA_CACHE_DIR, "meta.json")
REPORTS_DIR     = os.path.join(ROOT_DIR, "reports")
GRID_CSV        = os.path.join(REPORTS_DIR, "grid_train.csv")
BEST_PARAMS_JSON = os.path.join(REPORTS_DIR, "best_params.json")
OOS_CSV         = os.path.join(REPORTS_DIR, "oos_results.csv")
SUMMARY_MD      = os.path.join(REPORTS_DIR, "summary.md")
