"""
config.py — Parâmetros globais do pipeline Polymarket backtest multi-timeframe.
Edite aqui antes de rodar qualquer fase.
"""
from __future__ import annotations
import os

# ---------------------------------------------------------------------------
# Mercados suportados — (ativo, timeframe) → market_id
# Adicione/remova mercados aqui conforme os dados disponíveis em data_raw/.
# IMPORTANTE: nunca reutilize um market_id após remover um mercado.
# ---------------------------------------------------------------------------
MARKET_MAP: dict[str, int] = {
    # 5 minutos
    "BTC5m":   0,
    "ETH5m":   1,
    "SOL5m":   2,
    "XRP5m":   3,
    "HYPE5m":  4,
    # 15 minutos
    "BTC15m":  5,
    "ETH15m":  6,
    "SOL15m":  7,
    "XRP15m":  8,
    "HYPE15m": 9,
    # 1 hora
    "BTC1h":  10,
    "ETH1h":  11,
    "SOL1h":  12,
    "XRP1h":  13,
    "HYPE1h": 14,
    # 4 horas
    "BTC4h":  15,
    "ETH4h":  16,
    "SOL4h":  17,
    "XRP4h":  18,
    "HYPE4h": 19,
    # 1 dia
    "BTC1d":  20,
    "ETH1d":  21,
    "SOL1d":  22,
    "XRP1d":  23,
    "HYPE1d": 24,
}
MARKET_ID_TO_NAME: dict[int, str] = {v: k for k, v in MARKET_MAP.items()}

# ---------------------------------------------------------------------------
# Duração de cada ciclo em segundos (por timeframe)
# ---------------------------------------------------------------------------
CYCLE_LEN_MAP: dict[str, int] = {
    # 5 minutos
    "BTC5m":   300,
    "ETH5m":   300,
    "SOL5m":   300,
    "XRP5m":   300,
    "HYPE5m":  300,
    # 15 minutos
    "BTC15m":  900,
    "ETH15m":  900,
    "SOL15m":  900,
    "XRP15m":  900,
    "HYPE15m": 900,
    # 1 hora
    "BTC1h":   3_600,
    "ETH1h":   3_600,
    "SOL1h":   3_600,
    "XRP1h":   3_600,
    "HYPE1h":  3_600,
    # 4 horas
    "BTC4h":   14_400,
    "ETH4h":   14_400,
    "SOL4h":   14_400,
    "XRP4h":   14_400,
    "HYPE4h":  14_400,
    # 1 dia
    "BTC1d":   86_400,
    "ETH1d":   86_400,
    "SOL1d":   86_400,
    "XRP1d":   86_400,
    "HYPE1d":  86_400,
}

# Alias legado
CYCLE_LEN_S: int = 900  # equivalente a 15m

# ---------------------------------------------------------------------------
# Janela operacional padrão (legacy — apenas para 15m)
# ---------------------------------------------------------------------------
T_MIN: int = 60
T_MAX: int = 240

MAX_WAIT_FILL_S: int = 10
SIZE_SHARES: float = 5.0

# ---------------------------------------------------------------------------
# Split temporal
# ---------------------------------------------------------------------------
TRAIN_DAYS: int = 21
OOS_DAYS:   int = 7

# ---------------------------------------------------------------------------
# Grid de parâmetros — 3 dimensões
# ---------------------------------------------------------------------------

# 1. prob_entry_min (16 pontos: 0.550 → 0.925)
PROB_GRID: list[float] = [round(0.55 + i * 0.025, 3) for i in range(16)]

# 2. Janelas de tempo por timeframe — proporcionais à duração do ciclo
#    Proporções: ~3%, 7%, 7-27%, 13-33%, 20-67% do ciclo
T_WIN_MAP: dict[str, list[tuple[int, int]]] = {
    "5m": [
        ( 10,  40),
        ( 20,  60),
        ( 20,  80),
        ( 40, 100),
        ( 60, 200),
    ],
    "15m": [
        ( 30,  120),
        ( 60,  180),
        ( 60,  240),
        (120,  300),
        (180,  600),
    ],
    "1h": [
        (120,   480),
        (240,   720),
        (240,   960),
        (480,  1200),
        (720,  2400),
    ],
    "4h": [
        ( 480,  1920),
        ( 960,  2880),
        ( 960,  3840),
        (1920,  4800),
        (2880,  9600),
    ],
    "1d": [
        ( 2880,  11520),
        ( 5760,  17280),
        ( 5760,  23040),
        (11520,  28800),
        (17280,  57600),
    ],
}

# T_WIN_GRID padrão (15m) — usado nos scripts originais
T_WIN_GRID: list[tuple[int, int]] = T_WIN_MAP["15m"]

# 3. Stop loss (0.0 = hold to end)
STOP_LOSS_GRID: list[float] = [0.0, 0.05, 0.10, 0.15, 0.20]

# Total por timeframe: 16 × 5 × 5 = 400 combinações

# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------
SCORE_MIN_TRADES: int = 30

# ---------------------------------------------------------------------------
# Paralelismo (0 = auto)
# ---------------------------------------------------------------------------
N_SIM_THREADS:   int = 0
N_CLEAN_WORKERS: int = 0
N_GRID_WORKERS:  int = 0

def _auto(n: int) -> int:
    return os.cpu_count() or 4 if n == 0 else n

def get_sim_threads()   -> int: return _auto(N_SIM_THREADS)
def get_clean_workers() -> int: return _auto(N_CLEAN_WORKERS)
def get_grid_workers()  -> int: return _auto(N_GRID_WORKERS)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_timeframe(market_str: str) -> str:
    """Extrai o timeframe do nome do mercado. Ex: 'BTC15m' → '15m'."""
    for tf in ("1d", "4h", "1h", "15m", "5m"):
        if market_str.endswith(tf):
            return tf
    return "15m"  # fallback


def get_cycle_len(market_str: str) -> int:
    """Retorna duração do ciclo em segundos para o mercado dado."""
    return CYCLE_LEN_MAP.get(market_str, 900)


def get_t_win_grid(market_str: str) -> list[tuple[int, int]]:
    """Retorna T_WIN_GRID adequado para o timeframe do mercado."""
    return T_WIN_MAP[get_timeframe(market_str)]

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
ROOT_DIR        = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_RAW_DIR    = os.path.join(ROOT_DIR, "data_raw")
BOT_LOGS_DIR    = os.path.join(DATA_RAW_DIR, "bot_logs")
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
