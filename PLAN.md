# PLAN.md — Backtest + Grid Search Polymarket 15m (BTC/ETH/SOL/XRP)
# Stack: Cython + OpenMP · Polars · PyArrow/Parquet · orjson

---

## Visão geral

Pipeline **ultra-rápido** para backtest de mercados de predição Polymarket 15m.

```
JSONL raw  →  CLEAN (orjson + multiproc)  →  Parquet lean
           →  PACK  (Polars + PyArrow)    →  month.npz + cycles.npz
           →  SIM   (Cython + OpenMP)     →  pnl_per_cycle[]
           →  GRID  (ProcessPool)         →  grid_train.csv
           →  OOS   (validação)           →  oos_results.csv
           →  REPORT                      →  summary.md
```

**Throughput esperado vs plano original:**

| Fase         | Original (est.)  | Este plano           | Speedup |
|--------------|------------------|----------------------|---------|
| CLEAN        | 8 MB/s (json)    | 35 MB/s (orjson+mp)  | ~4x     |
| PACK         | 2 M rows/s (pd)  | 20 M rows/s (polars) | ~10x    |
| Sim por prob | ~10 s (Python)   | ~15 ms (Cython+OMP)  | ~700x   |
| Grid 19 pts  | ~3 min           | ~300 ms              | ~600x   |

---

## Melhorias-chave vs plano anterior

### 1. CLEAN — `orjson` + multiprocessing

- **`orjson`** em vez de stdlib `json`: 3–5x mais rápido no parse
- **`multiprocessing.Pool`**: um worker por arquivo raw em paralelo
- **Saída em Parquet** (PyArrow) em vez de JSONL lean:
  - Columnar + comprimido (snappy): 5–10x menor em disco
  - 10–20x mais rápido para carregar no PACK
- Filtros de validação compilados (sem regex dinâmica no loop)

### 2. PACK — Polars + PyArrow

- **Polars** para sort/groupby: 10–100x mais rápido que Pandas
- Conversão Arrow → NumPy **zero-copy** onde possível
- Arrays de saída com `np.ascontiguousarray` (cache-friendly)
- **`best_prob` e `best_side`** pré-calculados no PACK (nunca no grid)
- **`entry_eligible`** flag pré-computado por linha (T_MIN ≤ tr ≤ T_MAX)

### 3. Cython core — OpenMP prange + memoryviews tipadas

- **Typed memoryviews** (`float32_t[::1]`): acesso zero-overhead
- **`prange` com OpenMP**: paralelismo real em múltiplos cores
- **`nogil`** liberado no loop principal
- Flags do compilador: `-O3 -march=native -ffast-math -fopenmp`
- Directives: `boundscheck=False wraparound=False cdivision=True nonecheck=False initializedcheck=False`
- Arrays de saída pré-alocados (`np.empty`, não append de listas)
- Sem criação de objetos Python no loop quente

### 4. Grid Search — ProcessPoolExecutor

- `concurrent.futures.ProcessPoolExecutor` com `max_workers=cpu_count()`
- Cada worker avalia um `prob_entry_min` independentemente
- Arrays `month.npz`/`cycles.npz` carregados via `mmap_mode='r'` → sem cópia
- Split treino/OOS por `cycle_ids` (array pré-filtrado, não re-leitura)
- Score: `Sharpe diário` (mais robusto que PnL bruto) com penalidade por n_trades baixo

### 5. Carregamento — memory-mapped arrays

- `np.load('month.npz', mmap_mode='r')` → sem load completo em RAM
- Todos arrays em `dtype` mínimo (float32, int16, int8) → metade da RAM
- `cycles.npz` é pequeno (~C × 3 inteiros) → carregado inteiro

---

## Definições do dado

### Campos fonte (JSONL original)

```
ts_ms           int64   timestamp milissegundos
market          str     "BTC15m" | "ETH15m" | "SOL15m" | "XRP15m"
window_start    int64   unix segundos (início do ciclo de 15min)
derived.prob_up   float  probabilidade UP derivada
derived.prob_down float  probabilidade DOWN derivada
derived.overround float  (opcional)
yes.mid         float   fallback se derived ausente
no.mid          float   fallback
fetch.latency_ms float  (opcional, para análise de qualidade)
err             any     flag de erro do coletor
```

### Mapeamento do ciclo 15m

```
cycle_start_ts  = window_start
cycle_end_ts    = window_start + 900
ts_s            = ts_ms // 1000
time_remaining  = cycle_end_ts - ts_s   ∈ [0, 900]
```

### Probabilidades

Prioridade:
1. `derived.prob_up`, `derived.prob_down`
2. fallback: `yes.mid` (prob_up), `no.mid` (prob_down)

Descartar linha se nenhum par disponível ou fora de [0.001, 0.999].

### Determinação do outcome (backtest)

Sem dado de resolução real → usar proxy:
- `outcome_up = last_prob_up > 0.5` (última obs com `time_remaining ≤ 5`)
- Documentado como limitação; plugar resolução real quando disponível.

### Modelo de PnL

```
entry_price = prob_up no momento de entrada (= preço YES em USDC por share)
pnl_per_cycle:
  if outcome_up:  (1.0 − entry_price) × SIZE_SHARES   ← ganho
  else:           −entry_price × SIZE_SHARES            ← perda
  if not entered: 0.0
```

---

## Schema LEAN (Parquet)

Campos por linha (um segundo observado):

| Campo          | dtype   | Notas                        |
|----------------|---------|------------------------------|
| ts_s           | int64   |                              |
| market_id      | int8    | 0=BTC 1=ETH 2=SOL 3=XRP     |
| cycle_end_ts   | int64   | chave do ciclo               |
| time_remaining | int16   | [0, 900]                     |
| prob_up        | float32 |                              |
| prob_down      | float32 |                              |
| best_prob      | float32 | max(prob_up, prob_down)      |
| best_side      | int8    | 0=UP 1=DOWN                  |
| entry_eligible | int8    | T_MIN ≤ tr ≤ T_MAX (0/1)    |
| overround      | float32 | (nullable)                   |
| err_flag       | int8    | 0=ok 1=erro                  |

---

## Estrutura de pastas

```
backpoly/
├── data_raw/
│   ├── BTC15m/    *.jsonl
│   ├── ETH15m/
│   ├── SOL15m/
│   └── XRP15m/
├── data_cache/
│   ├── lean/      YYYY-MM-DD.MARKET.parquet
│   ├── month.npz  arrays contíguos (todos mercados)
│   ├── cycles.npz offsets por ciclo
│   └── meta.json  stats, market map, date range
├── reports/
│   ├── grid_train.csv
│   ├── best_params.json
│   ├── oos_results.csv
│   └── summary.md
├── src/
│   ├── py/
│   │   ├── config.py
│   │   ├── clean.py
│   │   ├── pack.py
│   │   ├── metrics.py
│   │   ├── gridsearch.py
│   │   ├── validate.py
│   │   └── report.py
│   └── cy/
│       ├── sim_core.pyx
│       └── sim_core.pxd
├── run_clean.py
├── run_pack.py
├── run_train_grid.py
├── run_validate_oos.py
├── setup.py
├── pyproject.toml
└── README.md
```

---

## FASE 0 — Configuração (`src/py/config.py`)

```python
MARKET_MAP = {"BTC15m": 0, "ETH15m": 1, "SOL15m": 2, "XRP15m": 3}
CYCLE_LEN_S  = 900

# Janela operacional (segundos restantes)
T_MIN = 60
T_MAX = 240

# Execução
MAX_WAIT_FILL_S = 10   # (reservado para fill real)
SIZE_SHARES     = 5.0

# Split temporal
TRAIN_DAYS = 21
OOS_DAYS   = 7

# Grid de probabilidade mínima de entrada
PROB_GRID = [round(0.50 + i * 0.025, 3) for i in range(21)]  # 0.500..1.000

# Score no grid
SCORE_MIN_TRADES = 30   # ciclos com trade mínimos para parâmetro ser válido

# Parallelism
N_SIM_THREADS = 0       # 0 = auto (os.cpu_count())
```

---

## FASE 1 — CLEAN (`src/py/clean.py` + `run_clean.py`)

### Algoritmo

```
Para cada arquivo raw (.jsonl) em paralelo (Pool):
  1. Abrir arquivo com open() + leitura linha a linha
  2. orjson.loads(line)        ← 3-5x mais rápido que json.loads
  3. Validar campos obrigatórios (ts_ms, window_start, market)
  4. Extrair prob_up / prob_down (derived ou fallback yes/no.mid)
  5. Calcular time_remaining; descartar fora [0, 900]
  6. Descartar probs fora (0, 1)
  7. Calcular best_prob, best_side, entry_eligible
  8. Acumular em lista de dicts ou buffer Arrow
  9. Dedup: manter ÚLTIMA obs por (market_id, cycle_end_ts, ts_s)
 10. Escrever Parquet via PyArrow (snappy)
```

### Dedup

Manter **última** ocorrência por `(market_id, cycle_end_ts, ts_s)`.
Consistente: uma regra, sempre.

### Saída

`data_cache/lean/YYYY-MM-DD.MARKET.parquet` por arquivo de entrada.

### Stats logadas

```json
{
  "file": "BTC15m/2024-01-15.jsonl",
  "kept": 82340,
  "dropped": 1200,
  "drop_pct": 1.44,
  "latency_ms_p50": 45.2,
  "latency_ms_p99": 312.0
}
```

---

## FASE 2 — PACK (`src/py/pack.py` + `run_pack.py`)

### Algoritmo

```
1. Glob data_cache/lean/*.parquet
2. Polars read_parquet (lazy scan + concat)
3. Sort por (market_id, cycle_end_ts, ts_s)            ← Polars: ~10M rows/s
4. Assign cycle_id sequencial por (market_id, cycle_end_ts)
5. Calcular cycle_start_idx, cycle_end_idx por cycle_id  ← offsets
6. Extrair arrays NumPy (zero-copy via Arrow onde possível)
7. np.savez_compressed('month.npz', ...)
8. np.savez_compressed('cycles.npz', ...)
9. Atualizar meta.json
```

### Arrays em `month.npz`

```
ts_s            int64[N]
market_id       int8[N]
cycle_id        int32[N]
time_remaining  int16[N]
prob_up         float32[N]
prob_down       float32[N]
best_prob       float32[N]
best_side       int8[N]
entry_eligible  int8[N]
err_flag        int8[N]
```

### Arrays em `cycles.npz`

```
cycle_start_idx   int32[C]   offset inicio no month.npz
cycle_end_idx     int32[C]   offset fim (exclusive)
cycle_market_id   int8[C]
cycle_end_ts      int64[C]   para filtrar por data
n_cycles          scalar int
```

### `meta.json`

```json
{
  "n_rows": 9700000,
  "n_cycles": 10752,
  "markets": {"BTC15m": 0, ...},
  "date_min": "2024-01-01",
  "date_max": "2024-01-28",
  "train_cycle_ids": [0, 1, ..., 9216],
  "oos_cycle_ids":  [9217, ..., 10751]
}
```

---

## FASE 3 — Cython Core (`src/cy/sim_core.pyx`)

### Flags de compilação

```python
# setup.py
extra_compile = ['-O3', '-march=native', '-ffast-math', '-fopenmp']
extra_link    = ['-fopenmp']

compiler_directives = {
    'language_level':   '3',
    'boundscheck':      False,
    'wraparound':       False,
    'cdivision':        True,
    'nonecheck':        False,
    'initializedcheck': False,
}
```

### API principal

```python
run_cycles(
    time_remaining,   # int16[::1]  — todos rows
    prob_up,          # float32[::1]
    cycle_start_idx,  # int32[::1]  — por ciclo
    cycle_end_idx,    # int32[::1]
    cycle_ids,        # int32[::1]  — subset (treino ou OOS)
    prob_entry_min,   # float
    t_min, t_max,     # int
    size_shares,      # float
    n_threads,        # int
) -> (pnl[nc], entered[nc], entry_prob[nc])   # float32, int8, float32
```

### Loop quente (pseudo-código C equivalente)

```cython
with nogil:
    for ci in prange(nc, schedule='static', num_threads=n_threads):
        cid   = cycle_ids[ci]
        start = cycle_start[cid]
        end   = cycle_end[cid]
        entered   = 0
        ep        = 0.0
        last_p    = -1.0

        for i in range(start, end):
            tr = time_remaining[i]
            p  = prob_up[i]
            if tr <= 5:
                last_p = p          # proxy de outcome
            if not entered and t_min <= tr <= t_max and p >= prob_entry_min:
                entered = 1
                ep = p

        pnl[ci] = (1.0 - ep) * size if entered and last_p > 0.5
                  else -ep * size   if entered
                  else 0.0
        ent[ci] = entered
        ep_out[ci] = ep
```

### Por que prange é seguro

Cada `ci` escreve em posições distintas de `pnl[ci]`, `ent[ci]`, `ep_out[ci]`.
Leituras são somente dos arrays compartilhados (read-only). Zero race condition.

---

## FASE 4 — Grid Search (`src/py/gridsearch.py` + `run_train_grid.py`)

### Split temporal

```python
# Por cycle_end_ts, não por linha (evita leakage de segundos)
cutoff_ts = sorted_unique_dates[-OOS_DAYS] * 86400   # unix segundos
train_ids = np.where(cycle_end_ts < cutoff_ts)[0].astype(np.int32)
oos_ids   = np.where(cycle_end_ts >= cutoff_ts)[0].astype(np.int32)
```

### Paralelismo

```python
from concurrent.futures import ProcessPoolExecutor

def _eval(prob):
    pnl, ent, ep = sim_core.run_cycles(..., cycle_ids=train_ids, prob_entry_min=prob)
    return compute_score(pnl, ent)

with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
    results = list(pool.map(_eval, PROB_GRID))
```

### Score

```python
score(pnl_arr, entered_arr):
    n_trades = entered_arr.sum()
    if n_trades < SCORE_MIN_TRADES:
        return -np.inf
    daily_pnl = aggregate_by_day(pnl_arr)    # array de PnL diário
    sharpe    = daily_pnl.mean() / (daily_pnl.std() + 1e-9) * sqrt(252)
    max_dd    = compute_max_drawdown(pnl_arr.cumsum())
    return sharpe - 0.3 * abs(max_dd)
```

### Saída

`reports/grid_train.csv`:
```
prob_entry_min,n_trades,total_pnl,sharpe,max_drawdown,score
0.500,8420,312.5,1.82,-45.0,1.68
...
```

`reports/best_params.json`:
```json
{"prob_entry_min": 0.725, "score": 2.41, "n_trades": 382}
```

---

## FASE 5 — Out-of-Sample (`src/py/validate.py` + `run_validate_oos.py`)

```python
best = json.load(open('reports/best_params.json'))
pnl, ent, ep = sim_core.run_cycles(..., cycle_ids=oos_ids,
                                   prob_entry_min=best['prob_entry_min'])

# Métricas OOS
per_market: por cycle_market_id → total_pnl, n_trades, sharpe
per_day:    por date(cycle_end_ts) → total_pnl, n_trades
```

Critérios de alerta:
- OOS sharpe < 0.5 × train sharpe → "overfit provável"
- n_trades OOS < SCORE_MIN_TRADES / OOS_DAYS × 5 → "poucos trades"
- max_drawdown OOS > 2 × train max_drawdown → "risco elevado OOS"

---

## FASE 6 — Report (`src/py/report.py`)

Gera `reports/summary.md` com:
- Parâmetros escolhidos
- Tabela treino vs OOS (total_pnl, n_trades, sharpe, max_dd)
- Tabela por mercado (OOS)
- Tabela por dia (OOS)
- Alertas automáticos

---

## Ordem de execução (CLI)

```bash
# 1. CLEAN
python run_clean.py

# 2. PACK
python run_pack.py

# 3. Compilar Cython
python setup.py build_ext --inplace

# 4. Grid Search (treino)
python run_train_grid.py

# 5. Validação OOS
python run_validate_oos.py

# Relatórios ficam em reports/
```

---

## Otimizações aplicadas desde o início

| Otimização                          | Onde          | Impacto         |
|-------------------------------------|---------------|-----------------|
| `orjson` em vez de `json`           | CLEAN         | 3–5x parse      |
| Multiprocessing por arquivo         | CLEAN         | Nx (N=cpu)      |
| Parquet (snappy) em vez de JSONL    | CLEAN→PACK    | 10x I/O         |
| Polars em vez de Pandas             | PACK          | 10–50x sort/grp |
| float32/int16/int8 (metade RAM)     | PACK+SIM      | 2x cache        |
| `np.ascontiguousarray`              | PACK→SIM      | cache-friendly  |
| Typed memoryviews Cython            | SIM           | zero-overhead   |
| `prange` + OpenMP                   | SIM           | Nx cores        |
| `-O3 -march=native -ffast-math`     | SIM           | 2–4x SIMD       |
| Offsets por ciclo (sem groupby)     | SIM           | O(1) acesso     |
| Features pré-calc no PACK           | SIM           | 0 recalc        |
| `mmap_mode='r'` no np.load          | GRID          | sem cópia RAM   |
| ProcessPoolExecutor no grid         | GRID          | Nx cores        |
| cycle_ids pré-filtrado (treino/OOS) | GRID+OOS      | sem re-leitura  |

---

## Dependências

```toml
[tool.poetry.dependencies]
python    = "^3.10"
numpy     = "^1.26"
cython    = "^3.0"
polars    = "^0.20"
pyarrow   = "^14.0"
orjson    = "^3.9"
tqdm      = "^4.66"
```

---

## Entregáveis

- `src/py/config.py` — parâmetros globais
- `src/py/clean.py` + `run_clean.py` — CLEAN fase 1
- `src/py/pack.py` + `run_pack.py` — PACK fase 2
- `src/cy/sim_core.pyx` + `src/cy/sim_core.pxd` — simulador Cython
- `setup.py` — build com OpenMP + SIMD
- `src/py/metrics.py` — funções de score
- `src/py/gridsearch.py` + `run_train_grid.py` — grid search
- `src/py/validate.py` + `run_validate_oos.py` — validação OOS
- `src/py/report.py` — geração do relatório
- `pyproject.toml` — dependências
- `README.md` — instruções completas
