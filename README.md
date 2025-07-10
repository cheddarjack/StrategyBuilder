# STRATEGY BUILDER

**Strategy Builder** is an advanced parameter-tuning framework for market-edge strategies using Optuna-driven Bayesian optimization (and TPE). It automates hyperparameter search across multiple self‑designed indicators and time‑based trade triggers, scaling from CPU to GPU workers. Ideal for researchers and developers exploring automated strategy calibration on tick data with specified indicators.

> **Note:** This repository provides the orchestration and scripts. Model checkpoints, raw market data, and Optuna database credentials must be supplied by the user.

## Table of Contents

* [Project Structure](#project-structure)
* [.gitignore](#gitignore)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Configuration](#configuration)
* [Data Layout](#data-layout)
* [Processing & Optimization](#processing--optimization)
* [Workflow](#workflow)
* [Module Overview](#module-overview)
* [Contributing](#contributing)
* [License](#license)

## Project Structure

```plaintext
.
├── c__config/
│   └── config.yaml                # Hyperparameter ranges, data paths, optimization settings
│
├── d__data/                       # Example data tree (fill with your own `.parquet` files)
│   ├── 1_converted_data/
│   ├── 2_snipped_data/
│   ├── 3_day_data_preprocessed/
│   ├── 3_preprocessed_data/
│   ├── 4_processed_tick_data/
│   └── 5_model_preped_data/
│
├── e__output/                     # Optimization results and logs (generated at runtime)
│   ├── optimization.log
│   ├── optimization_history.html
│   ├── optimization_history.png
│   ├── param_importances.html
│   ├── param_importances.png
│   └── results.parquet
│
├── f__modules/                    # Core strategy modules and utilities
│   ├── Strat__History/            # Historical indicator scripts (AvgRate_only, ComplexValue, Stochastic)
│   ├── dat_handler.py            # Data I/O and session slicing
│   ├── metrics.py                # Compute trade performance metrics
│   ├── strat.py                  # Main trade decision logic & custom indicators
│   └── trades_simulate.py        # Signal generation and trade simulation engine
│
├── g__scripts/                    # Orchestration scripts and optimization drivers
│   ├── 2_1__process_test.py       # Single-chunk test harness
│   ├── 2__process_data_chunks_multi.py # Multi-process data slicing
│   ├── 3__Bayesian.py            # Main Bayesian optimization (Optuna + RDB storage)
│   ├── 3__Bayesian_GPU.py        # GPU-accelerated variant
│   ├── 3__TPE.py                 # TPE-based optimization alternative
│   ├── Archive/                   # Legacy preprocessing scripts
│   ├── PrePreProcess/            # Parquet conversion/snipping utilities
│   └── prepro_large2VAmodel.py    # Large-scale preprocessing for VAmodel compatibility
│
├── requirements.txt               # Python dependencies
├── .gitignore                     # Patterns to exclude large data and credentials
└── README.md                      # This overview
```

## .gitignore

```gitignore
# Exclude raw and processed market data for legal resasons
d__data/
e__output/

# Optuna database credentials
.env

# Bytecode and cache\ n__pycache__/
*.py[cod]
```

## Prerequisites

* **Python 3.8+**
* Key libraries:

  * `optuna` (>=2.0)
  * `pandas`, `numpy`, `pyyaml`, `psutil`
  * `sqlalchemy` or `pymysql` for RDBStorage
  * `matplotlib` (for plotting)
  * `numba` (for speedups)

Install via:

```bash
pip install -r requirements.txt
```

## Installation

```bash
git clone git@github.com:<your-org>/BayesianMarketCurator.git
cd BayesianMarketCurator
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Edit `c__config/config.yaml` to set:

* Optuna storage URL (e.g., MySQL/MariaDB DSN)
* Study name and direction
* Hyperparameter ranges for:

  * `stoch_length`, `sma_period`, `AvgRate_N`, `ComplexValue` thresholds
  * `donchian_period`, `RoC_threshold`, `timer`, `take_profit`, `stop_loss`
* Data directories under `d__data/`
* Worker settings (`n_jobs`, CPU affinity, GPU flags)

Example snippet:

```yaml
optuna:
  storage: mysql+pymysql://user:pass@localhost:3306/optuna
  study_name: market_edge_opt
  direction: maximize

parameters:
  stoch_length:
    min: 3
    max: 30
  sma_period:
    min: 5
    max: 100
  AvgRate_N:
    min: 3
    max: 20
  ComplexValue:
    high_value: {min: 50, max: 300, step: 10}
    low_value:  {min: 10,  max: 200, step: 10}
  donchian_period:
    min: 2
    max: 20
  RoC_threshold:
    min: 10
    max: 200
  take_profit:
    min: 0.5
    max: 5.0
  stop_loss:
    min: 0.5
    max: 5.0
  timer:
    min: 30
    max: 3600

data:
  base_dir: ../d__data/3_day_data_preprocessed
  session_chunk: 10  # days per worker
  max_sessions: 20
workers:
  use_gpu: false
  n_jobs_cpu: 4
```

## Data Layout

Place your `.parquet` files under the corresponding `d__data` subfolders:

1. **1\_converted\_data/**: raw instrument snapshots
2. **2\_snipped\_data/**: front-month contract snips
3. **3\_day\_data\_preprocessed/**: daily bar/tick prepro
4. **3\_preprocessed\_data/**: multi-day aggregates
5. **4\_processed\_tick\_data/**: tick-by-tick streams
6. **5\_model\_preped\_data/**: final windowed inputs

## Processing & Optimization

1. **Preprocess** your market files using `g__scripts/2__process_data_chunks_multi.py` or legacy converters in `PrePreProcess/`.
2. **Run optimization**:

   * **Bayesian CPU**:

     ```bash
     python g__scripts/3__Bayesian.py --config c__config/config.yaml
     ```
   * **Bayesian GPU**:

     ```bash
     python g__scripts/3__Bayesian_GPU.py --config c__config/config.yaml
     ```
   * **TPE-based**:

     ```bash
     python g__scripts/3__TPE.py --config c__config/config.yaml
     ```
3. **Monitor logs** in `e__output/` and inspect `optimization_history.html` and `param_importances.html`.

## Workflow

1. **Session chunking**: Each worker receives 10-day segments; success proceeds to the next chunk, failure triggers parameter resampling.
2. **Indicator calculation** in `f__modules/strat.py`, including:

   * **Stochastic Oscillator** (variable lookbacks)
   * **Simple Moving Average (SMA)**
   * **Average Rate** of price change per N ticks
   * **Complex Value** indicator (piecewise linear mapping; see `calculate_complex_value`)
   * **Donchian Channels**
   * **Rate-of-Change** & **timer-based** filters
3. **Signal generation** and **trade simulation** in `trades_simulate.py`.
4. **Metrics** computed in `metrics.py` and passed to Optuna for objective evaluation.

## Module Overview

* **dat\_handler.py**: loads `.parquet`, slices by session ID
* **strat.py**: core feature functions & signal logic
* **trades\_simulate.py**: position management, SL/TP, time exits
* **metrics.py**: profitability, win rate, drawdown
* **g\_\_scripts/**: orchestration of preprocess, optimization, and result plotting

## Contributing

Pull requests welcome for:

* New optimization algorithms
* Advanced indicators
* Performance improvements (multiprocessing, GPU tuning)

Be sure to update `config.yaml` and `.gitignore` accordingly.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
