# Backtest Multi-Timeframe — Guia Completo

Avalia todos os ativos (BTC, ETH, SOL, XRP, HYPE) em 5 timeframes (5m, 15m, 1h, 4h, 1d)
e produz uma tabela de ranking com recomendação **TRADE** / **SKIP** para cada combinação.

---

## Ativos e Timeframes suportados

| Ativo | 5m | 15m | 1h | 4h | 1d |
|-------|----|-----|----|----|----|
| BTC   | ✓  | ✓   | ✓  | ✓  | ✓  |
| ETH   | ✓  | ✓   | ✓  | ✓  | ✓  |
| SOL   | ✓  | ✓   | ✓  | ✓  | ✓  |
| XRP   | ✓  | ✓   | ✓  | ✓  | ✓  |
| HYPE  | ✓  | ✓   | ✓  | ✓  | ✓  |

Total: **25 mercados**

---

## Passo 1 — Estrutura de pastas e dados

### 1.1 Onde colocar os arquivos JSONL

Cada mercado tem uma pasta própria dentro de `data_raw/`:

```
data_raw/
├── BTC5m/          ← arquivos JSONL do BTC 5 minutos
│   ├── 2026-01-01.jsonl
│   ├── 2026-01-02.jsonl
│   └── ...
├── ETH5m/
├── SOL5m/
├── XRP5m/
├── HYPE5m/
├── BTC15m/         ← arquivos JSONL do BTC 15 minutos
├── ETH15m/
├── SOL15m/
├── XRP15m/
├── HYPE15m/
├── BTC1h/
├── ETH1h/
├── SOL1h/
├── XRP1h/
├── HYPE1h/
├── BTC4h/
├── ETH4h/
├── SOL4h/
├── XRP4h/
├── HYPE4h/
├── BTC1d/
├── ETH1d/
├── SOL1d/
├── XRP1d/
└── HYPE1d/
```

> Todas as pastas já foram criadas automaticamente. Basta copiar os arquivos JSONL para a pasta correta.

### 1.2 Formato do JSONL (igual ao BTC15m já existente)

Cada linha do arquivo `.jsonl` deve ter este formato:

```json
{
  "ts_ms": 1704067200000,
  "market": "BTC5m",
  "window_start": 1704067200,
  "derived": {
    "prob_up": 0.723,
    "prob_down": 0.251
  }
}
```

**Campos obrigatórios:**
- `ts_ms` — timestamp em milissegundos
- `market` — nome exato do mercado (ex: `"BTC5m"`, `"ETH1h"`, `"HYPE15m"`)
- `window_start` — início do ciclo em segundos Unix
- `derived.prob_up` e `derived.prob_down` — probabilidades (entre 0.001 e 0.999)

---

## Passo 2 — Instalar dependências

```bash
pip install orjson polars pyarrow numpy
```

---

## Passo 3 — Compilar o simulador Cython

Execute **uma vez** antes de rodar qualquer backtest:

```bash
python setup.py build_ext --inplace
```

Isso compila o motor de simulação em C com OpenMP (muito mais rápido que Python puro).

---

## Passo 4 — Rodar o pipeline completo

Execute os comandos **na ordem abaixo**:

### 4.1 — CLEAN: converte JSONL → Parquet

```bash
python run_clean.py
```

Processa automaticamente todos os mercados que tiverem arquivos em `data_raw/`.
Apenas os mercados com dados são processados (os demais são ignorados sem erro).

Opções:
```bash
python run_clean.py --workers 8               # usar 8 processos paralelos
python run_clean.py --markets BTC15m ETH15m   # processar só esses mercados
```

### 4.2 — PACK: consolida Parquet → NumPy

```bash
python run_pack.py
```

Gera `data_cache/month.npz` e `data_cache/cycles.npz` com todos os dados consolidados.

### 4.3 — AVALIAR TODOS OS MERCADOS (novo)

```bash
python run_evaluate_all_markets.py
```

Este é o script principal. Ele:
1. Para cada mercado, roda um grid search de 400 combinações de parâmetros
2. Usa janelas de tempo **proporcionais ao timeframe** (ex: 5m usa janelas de 10-200s, 1d usa 3h-16h)
3. Valida os melhores parâmetros no período OOS (últimos 7 dias)
4. Decide **TRADE** ou **SKIP** para cada mercado

```bash
python run_evaluate_all_markets.py --workers 8   # paralelizar com 8 processos
python run_evaluate_all_markets.py --min-trades 20  # aceitar mínimo de 20 trades
```

---

## Saída — O que você vai ver

### Terminal

```
=========================================================================================
  RANKING DE MERCADOS — AVALIAÇÃO MULTI-TIMEFRAME
=========================================================================================

  Moeda        prob   t_min   t_max    stop   PnL OOS   Max DD   WR treino  Sharpe OOS           Status
  ----------  ------  ------  ------  -----  ---------  --------  ----------  -----------  ----------------
  BTC15m       0.625    60s   240s    0.20    +34.35    -4.46       68.7%         2.41             TRADE
  ETH15m       0.650    30s   120s    0.05    +13.66    -2.69       68.5%         1.82             TRADE
  BTC1h        0.600   240s   720s    0.10     +8.20    -3.10       65.2%         1.44             TRADE
  SOL15m       0.550    30s   120s    0.15     +6.44    -6.93       75.8%         1.12             TRADE
  HYPE15m      0.575    60s   180s    0.10     +4.20    -5.50       62.0%         0.85             TRADE
  BTC5m        0.700    20s    60s    0.05     -2.10    -8.20       51.0%        -0.32              SKIP
  ETH4h        ...                                                                                  SKIP
  ...
=========================================================================================
  TRADE: 8  |  SKIP: 17
=========================================================================================
```

### Arquivos gerados

| Arquivo | Conteúdo |
|---------|----------|
| `reports/all_markets_ranking.csv` | Todos os dados em formato CSV |
| `reports/all_markets_ranking.md`  | Tabela formatada em Markdown |

---

## Passo 5 — Interpretar os resultados

### Critérios para TRADE

Um mercado recebe **TRADE** quando todos esses critérios são atendidos:

| Critério | Valor mínimo |
|----------|-------------|
| PnL OOS  | > 0         |
| Sharpe OOS | ≥ 0.5     |
| Trades OOS | ≥ 10      |
| Win Rate (treino) | ≥ 55% |
| Anti-overfit: Sharpe OOS ≥ 50% do Sharpe treino | — |

### Como ler a tabela

| Coluna | Significado |
|--------|-------------|
| `prob` | Limiar mínimo de probabilidade para entrar no trade |
| `t_min / t_max` | Janela de tempo antes do fim do ciclo para entrar |
| `stop` | Stop loss em pontos de probabilidade (0 = hold to end) |
| `PnL OOS` | Lucro/prejuízo no período de teste (últimos 7 dias) |
| `Max DD` | Maior drawdown (quanto perdeu do pico ao fundo) |
| `WR treino` | Win rate no período de treino (21 dias) |
| `Sharpe OOS` | Sharpe ratio no período OOS (maior = melhor) |

---

## Passo 6 — Configurar o bot com os melhores parâmetros

Depois de identificar os mercados TRADE, use os parâmetros da tabela para configurar o bot:

```python
# Exemplo: BTC15m
prob_entry_min = 0.625
t_min = 60
t_max = 240
stop_loss_delta = 0.20
```

---

## Comandos de referência rápida

```bash
# Pipeline completo
python run_clean.py && python run_pack.py && python run_evaluate_all_markets.py

# Só reprocessar alguns mercados (ex: depois de adicionar dados novos)
python run_clean.py --markets BTC1h ETH1h HYPE15m
python run_pack.py
python run_evaluate_all_markets.py

# Grid search global (parâmetros únicos para todos os mercados)
python run_train_grid.py
python run_validate_oos.py

# Parâmetros individuais por mercado (via grid global)
python run_train_grid_per_market.py
python run_validate_oos_per_market.py
```

---

## Adicionar novos ativos

1. Crie a pasta em `data_raw/`:
   ```bash
   mkdir data_raw/NEWTOKEN15m
   ```

2. Adicione o mercado em `src/py/config.py`:
   ```python
   MARKET_MAP = {
       ...
       "NEWTOKEN15m": 25,   # próximo ID disponível
   }
   CYCLE_LEN_MAP = {
       ...
       "NEWTOKEN15m": 900,
   }
   ```

3. Coloque os arquivos JSONL na pasta e rode o pipeline normalmente.

---

## Estrutura de dados de saída (cycles.npz)

| Array | Tipo | Descrição |
|-------|------|-----------|
| `cycle_start_idx` | int32 | Índice de início de cada ciclo em month.npz |
| `cycle_end_idx`   | int32 | Índice de fim (exclusivo) |
| `cycle_market_id` | int8  | ID do mercado (ver MARKET_MAP) |
| `cycle_end_ts`    | int64 | Timestamp de fim do ciclo (Unix s) |
| `train_cycle_ids` | int32 | IDs dos ciclos de treino (21 dias) |
| `oos_cycle_ids`   | int32 | IDs dos ciclos OOS (7 dias) |

---

## Troubleshooting

| Problema | Solução |
|---------|---------|
| `sim_core não compilado` | Rodar `python setup.py build_ext --inplace` |
| `month.npz não encontrado` | Rodar `python run_pack.py` |
| `Nenhum arquivo JSONL encontrado` | Verificar se os arquivos estão nas pastas corretas |
| `Apenas N dias disponíveis` | Ajustar `TRAIN_DAYS`/`OOS_DAYS` em `config.py` |
| Todos os mercados com SKIP | Aumentar `--min-trades` para valor menor, ou verificar se há dados suficientes |
| Market recebe SEM_DADOS | Adicionar mais dados JSONL na pasta ou reduzir `--min-trades` |
