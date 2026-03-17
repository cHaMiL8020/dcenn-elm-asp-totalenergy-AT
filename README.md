# Neuro-Symbolic Power Generation Forecasting — Austrian Energy Grid

A hybrid **Neuro-Symbolic AI** system that combines a deep learning forecaster with a formal logic reasoning layer to predict and verify 15-minute total power generation in the Austrian electricity grid.

> **Key idea:** The neural network (dCeNN-ELM) handles pattern recognition and prediction from weather and temporal data. Answer Set Programming (ASP) then acts as a *logical safety net* — applying hard physical and calendar constraints to detect predictions that are numerically plausible but logically impossible or operationally suspicious.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Repository Structure](#repository-structure)
3. [Dataset](#dataset)
4. [Feature Engineering](#feature-engineering)
5. [Model: dCeNN-ELM](#model-dcenn-elm)
6. [Symbolic Reasoning: ASP Safety Rules](#symbolic-reasoning-asp-safety-rules)
7. [Austrian Holiday Integration](#austrian-holiday-integration)
8. [Performance Results](#performance-results)
9. [Benchmark Sweep](#benchmark-sweep)
10. [Visualisation](#visualisation)
11. [Setup & Usage](#setup--usage)
12. [Makefile Targets](#makefile-targets)
13. [Output Files](#output-files)
14. [Dependencies](#dependencies)

---

## Architecture Overview

```
Raw Data (Excel + CSV)
        │
        ▼
┌─────────────────────────────┐
│   01 · Data Preprocessing   │  Feature engineering, cyclical encoding,
│   src/01_preprocess.py      │  lag features, Austrian holiday flags
└────────────────┬────────────┘
                 │  data/processed_15min.csv
                 ▼
┌─────────────────────────────────────────────────────────────┐
│              02 · dCeNN-ELM Model Training                  │
│              src/02_train_dcenn_elm.py                      │
│                                                             │
│  ┌──────────────────────┐    ┌──────────────────────────┐  │
│  │  CentroidEncoder     │    │  ELM Regressor           │  │
│  │  (PyTorch autoenc.)  │───▶│  (Extreme Learning       │  │
│  │  11 features → ℝ¹⁰  │    │   Machine, 256 neurons)  │  │
│  └──────────────────────┘    └──────────────────────────┘  │
└────────────────┬────────────────────────────────────────────┘
                 │  data/predictions_2024.csv
                 ▼
┌─────────────────────────────────────────────────────────────┐
│         03 · ASP Symbolic Safety Layer                      │
│         src/03_apply_asp.py  +  rules/grid_rules.lp         │
│                                                             │
│  Neural facts (pred/actual/cglo/holiday) → Clingo solver   │
│  6 physical & calendar rules → anomaly(T, Reason) atoms    │
└────────────────┬────────────────────────────────────────────┘
                 │  data/flagged_anomalies.csv
                 ▼
┌─────────────────────────────┐
│   04 · Visualisation        │  4-panel analytics figure
│   src/04_visualize_results  │  notebooks/neuro_symbolic_plot.png
└─────────────────────────────┘
```

The pipeline is fully reproducible via a single `make pipeline` command.

---

## Repository Structure

```
dcenn-elm-asp-totalenergy-AT/
│
├── data/
│   ├── gen_dataset.xlsx          # Raw power generation data (sheet: data_2023)
│   ├── weather_data_15min.csv    # Raw weather observations (15-min resolution)
│   ├── processed_15min.csv       # Merged & feature-engineered dataset (generated)
│   ├── predictions_2024.csv      # Model predictions for Jan 2024 (generated)
│   ├── flagged_anomalies.csv     # ASP-detected anomalies (generated)
│   └── benchmarks/
│       ├── dcenn_elm_benchmark_runs.csv      # All individual benchmark runs
│       ├── dcenn_elm_benchmark_summary.csv   # Aggregated by config
│       ├── dcenn_elm_best_config.json        # Best hyperparameter config
│       ├── dcenn_elm_benchmark_report.md     # Plain-text report
│       └── dcenn_elm_benchmark_report.xlsx   # Formatted Excel workbook (3 sheets)
│
├── src/
│   ├── 01_preprocess.py          # Data loading, merging, feature engineering
│   ├── 02_train_dcenn_elm.py     # dCeNN autoencoder + ELM training & prediction
│   ├── 03_apply_asp.py           # Translate predictions → ASP facts, run Clingo
│   ├── 04_visualize_results.py   # 4-panel professional analytics plot
│   └── 05_benchmark_dcenn_elm.py # Hyperparameter sweep + Excel report generation
│
├── rules/
│   └── grid_rules.lp             # ASP safety rules (Clingo syntax)
│
├── notebooks/
│   └── neuro_symbolic_plot.png   # Generated visualisation (generated)
│
├── Makefile                      # One-command pipeline orchestration
├── requirements.txt              # Python dependencies
└── README.md
```

---

## Dataset

| Source file | Description | Resolution |
|---|---|---|
| `gen_dataset.xlsx` (sheet `data_2023`) | Total Austrian power generation (MW) | 15-minute |
| `weather_data_15min.csv` | Weather observations: global radiation (cglo), wind speed (ffam), precipitation (rr), temperature (tl) | 15-minute |

- **Training period:** 2023 (approx. 35,040 timesteps)
- **Test period:** January 2024 (2,976 timesteps = 31 days)
- After merging and dropping NaN from lag features: **70,030 rows total**

---

## Feature Engineering

Performed in `src/01_preprocess.py`. All features are derived from the raw timestamps and weather measurements:

| Feature | Type | Description |
|---|---|---|
| `cglo` | Weather | Global solar radiation (W/m²) |
| `ffam` | Weather | Mean wind speed (m/s) |
| `rr` | Weather | Precipitation (mm) |
| `tl` | Weather | Air temperature (°C) |
| `hour` | Calendar | Hour of day (0–23) |
| `minute` | Calendar | Minute of hour (0, 15, 30, 45) |
| `month` | Calendar | Month of year (1–12) |
| `day_of_week` | Calendar | Day of week (0=Mon … 6=Sun) |
| `time_sin` / `time_cos` | Cyclical | Sine/cosine encoding of time-of-day — avoids discontinuity at midnight |
| `month_sin` / `month_cos` | Cyclical | Sine/cosine encoding of month — avoids discontinuity at Dec→Jan |
| `power_lag_24h` | Lag | Power generation exactly 24 hours prior (96 × 15-min steps back) |
| `is_holiday` | Calendar | Binary flag (1/0) — Austrian public holiday via `holidays.Austria()` |

**Why cyclical encoding?** A raw `hour=23` and `hour=0` are numerically far apart but temporally adjacent. Projecting onto a unit circle (`sin(2π·h/24)`, `cos(2π·h/24)`) makes them continuous neighbours for the neural network.

---

## Model: dCeNN-ELM

Implemented in `src/02_train_dcenn_elm.py`.

### Stage 1 — CentroidEncoder (dCeNN)

A symmetric PyTorch autoencoder that learns a compact latent representation of the input features:

```
Input (11 features)
  → Linear(11→32) + ReLU
  → Linear(32→16) + ReLU
  → Linear(16→10)   ← latent space z ∈ ℝ¹⁰
  → Linear(10→16) + ReLU
  → Linear(16→32) + ReLU
  → Linear(32→11)   ← reconstruction (training only)
```

Training objective: minimise MSE reconstruction loss (Adam optimiser, lr=0.005, 50 epochs). The decoder is discarded after training — only the encoder's latent output `z` is passed to the ELM.

**Why an autoencoder?** It forces the network to discover a low-dimensional *centroid* representation of the data manifold, filtering noise before the regression step.

### Stage 2 — Extreme Learning Machine (ELM)

A single-hidden-layer feedforward network where **only the output weights are learned**:

```
z ∈ ℝ¹⁰  →  H = ReLU(z · W_rand + b_rand)  →  ŷ = H · β
```

- `W_rand`, `b_rand`: randomly initialised once, never updated
- `β`: output weights computed analytically via the **Moore-Penrose pseudo-inverse** of `H`

This makes training instantaneous (no gradient descent) and fully deterministic given a fixed seed.

### Hyperparameters (benchmark-tuned)

| Parameter | Value |
|---|---|
| Feature set | `baseline_plus_calendar` (11 features) |
| Latent dim | 10 |
| ELM hidden neurons | 256 |
| Epochs (autoencoder) | 50 |
| Learning rate | 0.005 |
| Random seed | 42 |

---

## Symbolic Reasoning: ASP Safety Rules

Implemented in `rules/grid_rules.lp`, executed by the **Clingo 5.7.1** ASP solver via `src/03_apply_asp.py`.

### How it works

For each 15-minute timestep `T` in the January 2024 test window, the script emits Clingo *facts*:

```prolog
pred(T, 112).      % model's predicted generation (MW, cast to int)
actual(T, 147).    % true measured generation (MW)
cglo(T, 380).      % solar radiation (W/m²)
holiday(T).        % emitted only on Austrian public holidays
```

The solver then applies the rules below and returns every `anomaly(T, Reason)` atom that can be derived.

### Rules

| # | Rule | Trigger condition | Notes |
|---|---|---|---|
| 1 | Negative Generation Predicted | `pred(T, V), V < 0` | Physically impossible |
| 2 | Predicted Below Historical Grid Baseline | `pred(T, V), V < 20 MW` | Austria never drops below ~20 MW total |
| 3 | Ramp Rate Violation (Spike) | `pred(T) − pred(T−1) > 25 MW` | Turbine inertia limit; **workdays only** |
| 4 | Ramp Rate Violation (Drop) | `pred(T−1) − pred(T) > 25 MW` | Turbine inertia limit; **workdays only** |
| 5 | Critical AI Deviation (workday) | `|pred − actual| > 30 MW`, `not holiday(T)` | AI model failure veto |
| 6 | Critical AI Deviation (holiday) | `|pred − actual| > 45 MW`, `holiday(T)` | Relaxed threshold — model trained mostly on workdays |
| 7 | Ramp Rate Violation — Holiday (Spike) | `pred(T) − pred(T−1) > 40 MW`, `holiday(T)` | Relaxed ramp limit on holidays |
| 8 | Ramp Rate Violation — Holiday (Drop) | `pred(T−1) − pred(T) > 40 MW`, `holiday(T)` | Relaxed ramp limit on holidays |

Rules 3–5 use ASP **negation-as-failure** (`not holiday(T)`) — they are logically skipped on any timestep where a `holiday(T)` fact exists, and the holiday-specific rules (6–8) take over instead.

---

## Austrian Holiday Integration

The `holidays` Python library is used to automatically retrieve all Austrian public holidays (Bundesfeiertage) for any year in the dataset:

```python
import holidays
at_holidays = holidays.Austria(years=[2023, 2024])
# e.g. 2024-01-01 → 'Neujahr', 2024-04-01 → 'Ostermontag', ...
```

**Impact (Jan 2024 test window):**
- Jan 1 (Neujahr) has a max model deviation of **37.1 MW** — this *would* have triggered 3 false-positive anomalies under the strict 30 MW workday rule
- With the holiday-aware rule, those 3 timesteps are correctly suppressed
- Total false positives avoided across the full test window: **3 timesteps** on a single public holiday

---

## Performance Results

Evaluated on the January 2024 test set (2,976 × 15-min timesteps):

### Absolute Metrics

| Metric | Value |
|---|---|
| **RMSE** | **10.20 MW** |
| **MAE** | **7.63 MW** |

### Scale-Free Metrics

| Metric | Value | Interpretation |
|---|---|---|
| **MAPE** | **9.50 %** | Average per-timestep % error — below the industry 10% threshold |
| **NRMSE** (÷ mean, 81.94 MW) | **12.44 %** | Error as a fraction of mean generation level |
| NRMSE (÷ range, 145.65 MW) | 7.00 % | Error relative to full operating range |
| CV-RMSE (÷ std, 28.41 MW) | 35.89 % | High — reflects genuine generation volatility |
| **R²** | **0.8712** | Model explains **87.1 %** of generation variance |

### ASP Anomaly Summary (Jan 2024)

| Rule triggered | Count |
|---|---|
| Critical AI Deviation | 28 |
| Ramp Rate Violation (Drop) | 16 |
| **Total anomalies flagged** | **44** |
| False positives suppressed (holiday relaxation) | 3 |

### Performance Interpretation

**The good:**
- R² = 0.87 is solid for sub-hourly resolution energy forecasting
- MAPE = 9.5% meets the standard industry threshold of < 10%
- The neuro-symbolic layer correctly identifies physically implausible predictions and suppresses spurious holiday flags

### Hourly MAE Profile (full test set)

The model's accuracy varies significantly by time of day. This is the per-hour mean absolute error across all days in the test set:

| Hour | MAE (MW) | Hour | MAE (MW) |
|---:|---:|---:|---:|
| 00 | 4.77 | 12 | 9.72 |
| 01 | 4.25 | 13 | 8.95 |
| 02 | 4.08 | 14 | 8.67 |
| **03** | **3.95** ← best | 15 | 9.20 |
| 04 | 4.30 | 16 | 8.03 |
| 05 | 4.70 | 17 | 7.74 |
| 06 | 4.83 | 18 | 10.37 |
| 07 | 5.02 | 19 | 11.77 |
| 08 | 5.67 | 20 | 10.11 |
| 09 | 7.33 | 21 | 9.60 |
| 10 | 9.19 | **22** | **11.99** ← worst |
| 11 | 9.33 | 23 | 9.66 |

**Known limitations:**
- **Worst hour: 22h (11.99 MW MAE)** — late-evening demand is the hardest to predict, likely due to irregular evening activity patterns not captured by the 24h lag feature
- **Best hour: 3h (3.95 MW MAE)** — overnight generation is stable and predictable
- Morning ramp-up (09–12h) also shows elevated error (9–10 MW) as solar generation begins
- CV-RMSE of ~36% reflects that the model smooths over sharp generation ramps
- Ramp-rate violations dominate the anomaly set, confirming the model struggles with rapid transitions

**Context:** This is a research-grade prototype demonstrating the neuro-symbolic paradigm. The dCeNN-ELM architecture is intentionally lightweight. A production forecaster (e.g., Temporal Fusion Transformer, N-BEATS) would likely achieve MAPE < 5–6%. The **primary contribution is the explainable ASP safety layer**, which provides rule-governed, auditable anomaly detection — something a pure deep-learning model cannot offer.

---

## Benchmark Sweep

`src/05_benchmark_dcenn_elm.py` performs a systematic hyperparameter search across:

| Dimension | Values swept |
|---|---|
| Feature sets | `baseline`, `baseline_plus_calendar`, `baseline_plus_minute`, `weather_plus_time` |
| Latent dim | 8, 10 |
| ELM hidden neurons | 100, 256 |
| Epochs | 50, 120 |
| Learning rate | 0.01, 0.005 |
| Random seeds | 42, 1337 |

Full sweep: **128 configurations** (≈ 400 s). Quick mode (`--quick`): 16 configurations.

### Benchmark Outputs

All results are saved to `data/benchmarks/`:

| File | Description |
|---|---|
| `dcenn_elm_benchmark_runs.csv` | Every individual run with all metrics |
| `dcenn_elm_benchmark_summary.csv` | Aggregated mean ± std per config |
| `dcenn_elm_best_config.json` | Best configuration as JSON |
| `dcenn_elm_benchmark_report.md` | Plain-text top-10 table |
| `dcenn_elm_benchmark_report.xlsx` | **Formatted Excel workbook** (3 sheets — see below) |

### Excel Workbook Sheets

| Sheet | Contents |
|---|---|
| **Summary (Aggregated)** | All configs sorted by RMSE; navy header; alternating row stripes; gold highlight on winner; green→yellow→red conditional colour scale on RMSE column; frozen header |
| **All Runs (Raw)** | Every individual run with number-formatted columns; striped rows |
| **Best Config** | Card layout with Hyperparameters / Evaluation Metrics (mean ± std) / Runtime sections |

### Best Configuration Found

```json
{
  "feature_set": "baseline_plus_calendar",
  "latent_dim": 10,
  "elm_hidden_neurons": 256,
  "epochs": 50,
  "learning_rate": 0.005,
  "rmse_mean": 10.018,
  "mae_mean": 7.423,
  "mape_percent_mean": 9.38,
  "asp_anomaly_count_mean": 42
}
```

---

## Visualisation

`src/04_visualize_results.py` generates a **4-panel analytics figure** at `notebooks/neuro_symbolic_plot.png` (300 dpi, 16 × 13 inches):

| Panel | Contents |
|---|---|
| **A — Time Series** (full width) | Actual vs dCeNN-ELM prediction for the first 7 days of Jan 2024; ±10 MW confidence band; per-reason anomaly markers (X = Critical Deviation, ▲ = Ramp Spike, ▼ = Ramp Drop); amber shading on public holidays; arrow annotation on worst anomaly |
| **B — Residual Scatter** | Predicted value vs residual (Actual − Predicted) for every test timestep; anomalies highlighted red; rolling mean trend line to reveal systematic bias |
| **C — Hourly MAE Profile** (full test set) | Mean ± 1 std MAE for each hour of day (0–23); worst hour annotated; evening 18–23h band shaded to highlight the model's known weakness |
| **D — Anomaly Breakdown** (full width) | Horizontal bar chart of ASP anomaly counts by rule type, value-labelled |

Footer shows test RMSE, MAE, total anomaly count, and holiday legend.

---

## Setup & Usage

### Prerequisites

- Python 3.10+ (developed on 3.12.3)
- `make` (GNU Make)

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/cHaMiL8020/dcenn-elm-asp-totalenergy-AT.git
cd dcenn-elm-asp-totalenergy-AT

# 2. Create virtual environment and install dependencies
make install

# 3. Run the full pipeline end-to-end
make pipeline
```

This runs all four stages in sequence and produces:
- `data/processed_15min.csv`
- `data/predictions_2024.csv`
- `data/flagged_anomalies.csv`
- `notebooks/neuro_symbolic_plot.png`

### Run individual stages

```bash
make preprocess    # Stage 1: feature engineering
make train         # Stage 2: train dCeNN-ELM, generate predictions
make asp           # Stage 3: run ASP anomaly detection
make visualize     # Stage 4: generate 4-panel plot
```

### Run the benchmark sweep

```bash
make benchmark-quick   # 16 configs  (~60 s)
make benchmark         # 128 configs (~400 s), multi-seed
```

---

## Makefile Targets

| Target | Description |
|---|---|
| `make help` | Show all available targets |
| `make venv` | Create `.venv` virtual environment |
| `make install` | Install all dependencies from `requirements.txt` |
| `make preprocess` | Run `src/01_preprocess.py` |
| `make train` | Run `src/02_train_dcenn_elm.py` |
| `make asp` | Run `src/03_apply_asp.py` |
| `make visualize` | Run `src/04_visualize_results.py` |
| `make benchmark` | Full hyperparameter sweep (128 configs, 2 seeds) |
| `make benchmark-quick` | Quick sweep (16 configs, 1 seed) |
| `make pipeline` | `preprocess → train → asp → visualize` |
| `make clean` | Delete all generated output files |

---

## Output Files

| File | Generated by | Description |
|---|---|---|
| `data/processed_15min.csv` | `01_preprocess.py` | Merged, feature-engineered dataset (70,030 rows, 16 columns) |
| `data/predictions_2024.csv` | `02_train_dcenn_elm.py` | Test set predictions with timestamps and actuals |
| `data/flagged_anomalies.csv` | `03_apply_asp.py` | Timesteps flagged by ASP with reason label |
| `notebooks/neuro_symbolic_plot.png` | `04_visualize_results.py` | 4-panel analytics figure (300 dpi) |
| `data/benchmarks/*.csv` | `05_benchmark_dcenn_elm.py` | All run data and aggregated summaries |
| `data/benchmarks/*.xlsx` | `05_benchmark_dcenn_elm.py` | Formatted Excel workbook |
| `data/benchmarks/*.json` | `05_benchmark_dcenn_elm.py` | Best config as JSON |

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `torch` | 2.2.0 | CentroidEncoder autoencoder (dCeNN) |
| `numpy` | 1.26.3 | ELM matrix operations (pseudo-inverse) |
| `pandas` | 2.2.0 | Data loading, time-series manipulation |
| `scikit-learn` | 1.4.0 | StandardScaler, RMSE/MAE metrics |
| `clingo` | 5.7.1 | ASP solver (Answer Set Programming) |
| `matplotlib` | 3.8.2 | 4-panel visualisation |
| `seaborn` | 0.13.2 | Plot styling |
| `openpyxl` | latest | Excel reading (raw data) + Excel writing (benchmark report) |
| `holidays` | latest | Austrian public holiday calendar |
| `pyarrow` | latest | Pandas Arrow backend (performance) |
| `pytz` | 2024.1 | Timezone handling |

Install all with:
```bash
pip install -r requirements.txt
```