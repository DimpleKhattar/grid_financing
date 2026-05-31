# Grid Financing Analysis — GBM & NBB-SO Models

A comprehensive framework for analyzing electricity market dynamics, training scheduling optimization, and outage forecasting for data center operations in ERCOT.

## Overview

This project implements three integrated models:

1. **GBM Calibration & Monte Carlo Simulation** — Calibrates geometric Brownian motion parameters from ERCOT LMP and Henry Hub gas prices; generates 10,000 correlated price paths for spark spread risk analysis.

2. **NBB-SO Outage Estimator** — Negative Binomial Boosted—Simultaneous Optimization model jointly estimating mean and dispersion parameters of outage counts as functions of weather, market, and temporal covariates using alternating Newton-step gradient boosting.

3. **LSMC Optimal Training Schedule** — Least-squares Monte Carlo dynamic programming to compute optimal exercise boundaries for when to run AI model training vs. inference given hourly electricity prices.

4. **Wind & Solar Intermittency Analysis** — Characterizes renewable generation patterns and negative pricing regimes in ERCOT's West zone.

## Project Structure

```
grid-financing/
├── code/
│   ├── gbm_calibration_monte_carlo.py           # GBM parameter estimation & simulation
│   ├── nbb_so_estimator.py                      # Outage count regression model
│   ├── nbb_so_helper.py                         # Diagnostic plotting utilities
│   ├── download_ercot_lmp_new.py                # Master data downloader
│   ├── LSMC_Optimal_Training_Schedule_MaxRevenue.py  # Training scheduling optimization
│   └── wind_solar_intermittency_analysis.py     # Renewable generation analysis
├── data/                                        # (create this; populated by download script)
├── output/                                      # (create this; outputs go here)
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/grid-financing.git
   cd grid-financing
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create data and output directories:
   ```bash
   mkdir -p data output
   ```

## Usage

### 1. Download Market Data

Download ERCOT LMP, Henry Hub gas prices, wind, and solar generation data:

```bash
python code/download_ercot_lmp_new.py
```

This creates CSV files in `data/`:
- `ercot_lmp_hb_houston_hb_west.csv` — Hourly electricity prices (HB_HOUSTON, HB_WEST)
- `henry_hub_daily.csv` — Daily natural gas spot prices
- `ercot_wind_hourly.csv` — Hourly wind generation by region
- `ercot_solar_hourly.csv` — Hourly solar generation by region

### 2. Calibrate GBM Parameters & Run Monte Carlo

```bash
python code/gbm_calibration_monte_carlo.py
```

Outputs to `output/`:
- **CSV tables**: Calibration parameters, correlation matrix, terminal distribution statistics
- **PNG charts** (13 total): Log-return histograms, Q-Q plots, rolling parameter stability, simulated price paths, spark spread fan charts, risk metrics

Key metrics: µ (drift, /year), σ (volatility, /year), VaR/CVaR at 95% confidence.

### 3. Fit NBB-SO Outage Model

```bash
python code/nbb_so_estimator.py
```

Outputs to `output/`:
- **CSV tables**: Performance metrics (MAE, RMSE, AUC, McFadden R²), dispersion summary, feature importance
- **PNG charts**: ROC curves, actual vs predicted, confusion matrix, feature importance (F_µ and F_α)

Default uses synthetic ERCOT 2018–2023 data; provide real EIA-417 form data via `build_dataset(data_path='...')`.

### 4. Optimize AI Training Schedule (LSMC)

```bash
python code/LSMC_Optimal_Training_Schedule_MaxRevenue.py
```

Outputs exercise boundary thresholds (hourly LMP levels below which training should run), annual revenue comparison vs. naïve scheduling.

### 5. Analyze Wind & Solar Intermittency

```bash
python code/wind_solar_intermittency_analysis.py
```

Characterizes renewable generation variability and correlation with negative pricing events in the ERCOT West zone.

## Data Sources

- **ERCOT LMP**: Real-Time Market settlement point prices via ERCOT CDR (Report ID 13061)
- **Henry Hub**: Daily spot prices from FRED (DHHNGSP)
- **ERCOT Wind/Solar**: Hourly generation by region via ERCOT CDR (Report IDs 13028, 21809)

## Model Assumptions

### GBM Calibration
- Log-returns assumed i.i.d. Normal (Shapiro-Wilk test included)
- LMP floor at $1/MWh (GBM requires positive prices)
- Daily simulation steps (252/year) vs. hourly calibration (8,760/year) for computational speed
- Antithetic variates reduce Monte Carlo variance ~30–50%

### NBB-SO Outage Model
- Negative Binomial NB-2 parameterization: Var(Y) = µ + α·µ²
- Log links for both mean and dispersion ensure positivity
- Chronological train/test split (no temporal data leakage)
- Lewbel IV optional for endogeneity correction

### LSMC
- Correlated GBM paths capped at historical 1st–99th percentiles
- Basis functions: polynomial in LMP_HOU and LMP_WEST
- 1,000 paths × 182 days × 24 hours = 4.3M scenarios

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `N_PATHS` | 10,000 | Monte Carlo paths |
| `HORIZON_YEARS` | 1 | Simulation horizon |
| `HEAT_RATE` | 9.5 | MMBtu/MWh (simple-cycle turbine) |
| `GAS_ADDER_HOUSTON` | $3.00 | Houston Ship Channel premium |
| `LMP_FLOOR` | $1.00 | Floor price for GBM safety |

## Output Naming Convention

All outputs follow descriptive naming:
```
{MODEL}_{REGION/PARAMETER}_{METRIC}_{DATERANGE}.{EXT}
```

Example: `GBM_MonteCarlo_SparkSpread_HB-HOUSTON_10kpaths_1yr_horizon_..._.png`

## Performance Notes

- **GBM Calibration**: ~5–10 seconds (10k paths)
- **NBB-SO Fitting**: ~30–60 seconds per zone (100 boosting rounds)
- **LSMC**: ~2–5 minutes (1k paths × 182 days × 24 hours)
- **Data Download**: 5–15 minutes (network-dependent)

## Citation

If you use this code, please cite:

```
Grid Financing Analysis Framework (2026)
Author: Dimple Khattar
University of Chicago, Harris School
```

## License

MIT License

## Support

For issues, questions, or data source changes, please check ERCOT's CDR directory at https://www.ercot.com/misapp/GetReports.do
