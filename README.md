# Polymarket Backtest Pipeline — 15m (BTC/ETH/SOL/XRP)

Pipeline ultra-rápido para backtest de mercados de predição Polymarket com janela de 15 minutos.

## Stack de velocidade

| Componente | Tecnologia | Ganho vs baseline |
|---|---|---|
| Parse JSON | `orjson` | ~4x |
| CLEAN paralelo | `multiprocessing.Pool` | Nx (N=CPUs) |
| Armazenamento lean | Parquet/snappy | 10x menor, 10x I/O mais rápido |
| Sort/groupby | `Polars` | 10–100x vs Pandas |
| Simulador | Cython + OpenMP | ~700x vs Python puro |
| Grid search | `ProcessPoolExecutor` | Nx (N=CPUs) |
| Carregamento arrays | `np.load(mmap_mode='r')` | sem cópia RAM |

## Requisitos

```bash
pip install numpy cython polars pyarrow orjson tqdm
```

## Estrutura dos dados raw

```
data_raw/
├── BTC15m/   *.jsonl
├── ETH15m/   *.jsonl
├── SOL15m/   *.jsonl
└── XRP15m/   *.jsonl
```

Formato de cada linha JSONL:
```json
{
  "ts_ms": 1704067200000,
  "market": "BTC15m",
  "window_start": 1704067200,
  "derived": {"prob_up": 0.723, "prob_down": 0.251, "overround": 1.026},
  "fetch": {"latency_ms": 42.1},
  "err": null
}
```

## Execução passo a passo

### 1. Configuração

Edite `src/py/config.py` conforme necessário:
- `T_MIN`, `T_MAX` — janela de entrada (segundos restantes)
- `SIZE_SHARES` — tamanho da posição
- `TRAIN_DAYS`, `OOS_DAYS` — split temporal
- `PROB_GRID` — valores a testar no grid

### 2. CLEAN — JSONL raw → Parquet lean

```bash
python run_clean.py
# Com N workers explícitos:
python run_clean.py --workers 8
# Apenas alguns mercados:
python run_clean.py --markets BTC15m ETH15m
```

Saída: `data_cache/lean/*.parquet`

### 3. PACK — Parquet lean → arrays NumPy

```bash
python run_pack.py
```

Saída:
- `data_cache/month.npz` — todos os dados linha a linha
- `data_cache/cycles.npz` — offsets por ciclo + split treino/OOS
- `data_cache/meta.json` — estatísticas

### 4. Compilar o simulador Cython

```bash
python setup.py build_ext --inplace
```

Verifica se compilou:
```bash
python -c "from src.cy.sim_core import run_cycles; print('OK')"
```

### 5. Grid Search (treino)

```bash
python run_train_grid.py
# Modo sequencial para debug:
python run_train_grid.py --seq
```

Saída:
- `reports/grid_train.csv`
- `reports/best_params.json`

### 6. Validação Out-of-Sample

```bash
python run_validate_oos.py
```

Saída:
- `reports/oos_results.csv`
- `reports/summary.md`

## Exemplo de saída do grid

```
 prob  trades    total_pnl   sharpe    max_dd     score
0.500    8420      312.50    1.8200    -45.00     1.6750
0.525    7310      298.30    1.9100    -38.50     1.7545
...
0.750     420      185.20    2.4100    -12.30     2.3210  ← melhor
...
0.975      12       15.10    0.3200     -3.10    -inf     ← abaixo do mín trades
```

## Modelo de PnL (MVP)

```
Entrada: compra YES a prob_up (= preço por share)
Saída:   aguarda resolução do ciclo

Outcome proxy: última obs com time_remaining ≤ 5s
  se prob_up_final > 0.5 → outcome = UP
  senão → outcome = DOWN

PnL:
  outcome UP:   (1 − entry_price) × SIZE_SHARES   [ganho]
  outcome DOWN: − entry_price × SIZE_SHARES        [perda]
```

> **Limitação**: o outcome é aproximado pela última prob observada.
> Para resultados reais, plugue a resolução real no `sim_core.pyx`.

## Flags de compilação Cython

```
-O3           otimização máxima
-march=native SIMD automático (AVX2/AVX-512)
-ffast-math   matemática rápida
-fopenmp      prange paralelo (Linux/macOS com libomp)
```

Directives:
```
boundscheck=False   wraparound=False   cdivision=True
nonecheck=False     initializedcheck=False
```

## Benchmarks esperados (28 dias, 4 mercados, ~10M linhas)

| Fase | Tempo |
|---|---|
| CLEAN (8 workers) | ~30s |
| PACK (Polars) | ~5s |
| Compilação Cython | ~15s |
| Grid (19 pts × 8 cores) | < 500ms |
| OOS | < 50ms |

## Estrutura de arquivos

```
src/py/config.py         parâmetros globais
src/py/clean.py          lógica CLEAN
src/py/pack.py           lógica PACK
src/py/metrics.py        funções de score
src/py/gridsearch.py     grid search paralelo
src/py/validate.py       validação OOS
src/py/report.py         geração de relatório Markdown
src/cy/sim_core.pyx      simulador Cython + OpenMP
src/cy/sim_core.pxd      declarações Cython
setup.py                 build Cython
run_clean.py             CLI FASE 1
run_pack.py              CLI FASE 2
run_train_grid.py        CLI FASE 4
run_validate_oos.py      CLI FASE 5 + 6
```
