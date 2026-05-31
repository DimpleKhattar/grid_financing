"""
Author: Dimple
Date created: 18 May 2026
Purpose: GBM Calibration and Monte Carlo Simulation Framework


WHAT THIS FILE DOES
-------------------
1. Calibrates GBM parameters (µ, σ) from historical price series via MLE:
     ERCOT LMP  — HB_HOUSTON and HB_WEST (hourly, Feb–May 2026)
     Henry Hub  — Natural gas spot price (daily, Jan 2025–May 2026)
2. Tests log-return normality (Shapiro-Wilk).
3. Plots rolling-window parameter stability.
4. Simulates 10,000 correlated price paths via Cholesky decomposition +
   antithetic variates for variance reduction.
5. Computes spark spread paths and risk metrics (VaR, CVaR at 95%).
6. Saves 18 outputs (13 PNGs + 5 CSVs) to output/ with comprehensive names.

GBM MODEL
---------
  Continuous form:  dS = µ S dt + σ S dW
  Discrete update:  S(t+dt) = S(t) · exp[(µ − σ²/2)·dt + σ·√dt·Z],  Z ~ N(0,1)
  MLE estimates:    µ_annual = (mean log-return)/dt + σ²/2  (drift correction)
                    σ annual = std(log-returns) / √dt


ASSUMPTIONS & MARKET STANDARDS
-------------------------------
1. GBM log-normality assumption
   GBM assumes log-returns are i.i.d. Normal. Electricity prices exhibit spikes,
   mean-reversion, and fat tails — GBM is a first-order approximation used here
   for tractability and comparability with standard energy finance practice
   (e.g., Black-Scholes spark spread options). Results should be interpreted
   with awareness of heavy-tailed downside risk.

2. LMP price floor at $1/MWh (LMP_FLOOR)
   ERCOT allows LMPs from −$2,000/MWh (LCAP) to +$5,000/MWh (HCAP) since 2022.
   Negative prices occur when wind/solar over-generates. GBM requires positive
   prices for log-return computation. We floor at $1/MWh before log-return
   calculation and during simulation. This is standard practice in energy
   quant models to handle the positive-domain requirement of GBM.

3. Daily simulation steps (not hourly)
   Simulation uses 252 daily steps per year (N_STEPS_DAILY) even though LMP
   is calibrated from hourly data. Both use the same annualised µ and σ, so
   the expectation of terminal prices is identical. Daily steps are 35× faster
   (252 vs 8,760) and sufficient for annual horizon risk analysis.

4. Annualisation convention
   LMP (hourly):    dt = 1/8,760 yr;  σ_annual = σ_hourly × √8,760
   Henry Hub (daily): dt = 1/252 yr;  σ_annual = σ_daily × √252
   252 trading days/year is the CFTC/CME natural gas futures market standard.
   8,760 hours/year is the ERCOT market convention for capacity calculations.

5. Drift correction
   MLE drift: µ_annual = (mean log-return / dt) + σ²/2
   The + σ²/2 Itô correction converts the log-space drift to the real-space
   expected return, so E[S(T)] = S₀ · exp(µ_annual · T).

6. Spark spread formula
   Spark Spread ($/MWh) = LMP − (HenryHub + gas_adder) × heat_rate
   where:
     heat_rate = 9.5 MMBtu/MWh   (ERCOT simple-cycle peaker; industry standard
                                   for open-cycle gas turbine, per EIA Form 860)
     gas_adder_Houston = $3.00/MMBtu  (HSC transport + fuel + variable O&M)
     gas_adder_West    = $0.00/MMBtu  (Waha basis; modelled at zero adder
                                        to represent wellhead gas cost)
   A positive spark spread indicates the plant earns above its variable cost.

7. Correlated simulation via Cholesky decomposition
   Correlation matrix estimated from daily log-returns over the overlapping
   data period. Cholesky factorisation L is applied: Z_corr = L @ Z, where
   Z are i.i.d. standard normals. This preserves the empirical cross-correlations
   between LMP hubs and Henry Hub gas in the simulated paths.

8. Antithetic variates (variance reduction)
   Half of N_PATHS use standard normals Z; the other half use −Z. This reduces
   Monte Carlo standard error by ~30-50% for smooth functionals at no extra cost.
   The estimator remains unbiased.

9. VaR and CVaR at 95% confidence
   VaR_95  = −P5 of terminal distribution (5th percentile loss threshold)
   CVaR_95 = −E[X | X ≤ P5]  (expected shortfall; coherent risk measure)
   Reported for terminal spark spread and annual mean spark spread.

10. McFadden R² (NBB-SO reference metric; not GBM)
    Mentioned in assumptions for completeness — see nbb_so_estimator.py.

"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")

# Paths

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Simulation config

N_PATHS        = 10_000       # Monte Carlo paths
HORIZON_YEARS  = 1            # forward simulation horizon
DT_HOURLY      = 1 / 8_760   # fraction of a year per hour
DT_DAILY       = 1 / 252     # fraction of a year per trading day
N_STEPS_HOURLY = int(HORIZON_YEARS / DT_HOURLY)   # 8 760 hourly steps
N_STEPS_DAILY  = int(HORIZON_YEARS / DT_DAILY)    # 252 daily steps

HEAT_RATE  = 9.5   # MMBtu/MWh  (simple-cycle turbine)
GAS_ADDER_HOUSTON = 3.0   # $/MMBtu  Houston Ship Channel toll + VOM
GAS_ADDER_WEST    = 0.0   # $/MMBtu  Waha (West Texas)

LMP_FLOOR  = 1.0   # $/MWh  — floor before log-return computation
SEED       = 42

rng = np.random.default_rng(SEED)


#DATA LOADING

def load_data():
    lmp = pd.read_csv(
        os.path.join(DATA_DIR, "ercot_lmp_hb_houston_hb_west.csv"),
        parse_dates=["datetime_cst"],
    ).sort_values("datetime_cst").reset_index(drop=True)

    hh = pd.read_csv(
        os.path.join(DATA_DIR, "henry_hub_daily.csv"),
        parse_dates=["date"],
    ).sort_values("date").reset_index(drop=True)

    print(f"LMP  : {len(lmp):>5} hourly rows  "
          f"[{lmp['datetime_cst'].min().date()} → {lmp['datetime_cst'].max().date()}]")
    print(f"HenHub: {len(hh):>4} daily rows   "
          f"[{hh['date'].min().date()} → {hh['date'].max().date()}]")
    return lmp, hh



# GBM CALIBRATION

def calibrate_gbm(series: np.ndarray, dt: float, label: str):
    """
    MLE calibration of GBM parameters from a price series.
    """
    s = np.maximum(series, LMP_FLOOR)           # floor for log-return safety
    lr = np.diff(np.log(s))                     # log returns

    mu_dt    = float(np.mean(lr))
    sigma_dt = float(np.std(lr, ddof=1))

    sigma_annual = sigma_dt / np.sqrt(dt)
    mu_annual    = mu_dt / dt + 0.5 * sigma_annual ** 2  # drift correction

    half_life = np.log(2) / abs(mu_annual) * 252 if mu_annual != 0 else np.inf

    result = dict(
        label        = label,
        S0           = float(s[-1]),
        mu_annual    = round(mu_annual,    6),
        sigma_annual = round(sigma_annual, 6),
        mu_daily     = round(mu_dt / dt * DT_DAILY, 8),
        sigma_daily  = round(sigma_dt / np.sqrt(dt) * np.sqrt(DT_DAILY), 8),
        half_life_days = round(half_life, 2),
        log_returns  = lr,
        n_obs        = len(lr),
    )

    # Normality test on log returns
    stat, pval = stats.shapiro(lr[:5000])   # Shapiro-Wilk (max 5 000 obs)
    result["shapiro_stat"] = round(float(stat), 4)
    result["shapiro_pval"] = round(float(pval), 4)
    result["normal_at_5pct"] = pval > 0.05

    print(f"  {label:<22} µ={mu_annual:+.3f}/yr  σ={sigma_annual:.3f}/yr  "
          f"S0=${s[-1]:.2f}  n={len(lr)}")
    return result



# ROLLING PARAMETER STABILITY


def rolling_calibration(series: np.ndarray, dt: float, window: int = 168):
    """Rolling window GBM calibration (default 168 h = 1 week)."""
    s  = np.maximum(series, LMP_FLOOR)
    lr = np.diff(np.log(s))
    n  = len(lr)
    mu_roll    = np.full(n, np.nan)
    sigma_roll = np.full(n, np.nan)
    for i in range(window, n + 1):
        chunk = lr[i - window:i]
        mu_roll[i - 1]    = np.mean(chunk) / dt + 0.5 * (np.std(chunk, ddof=1) / np.sqrt(dt)) ** 2
        sigma_roll[i - 1] = np.std(chunk, ddof=1) / np.sqrt(dt)
    return mu_roll, sigma_roll



# CORRELATION MATRIX  (daily-aggregated LMP vs Henry Hub)


def compute_correlation(lmp: pd.DataFrame, hh: pd.DataFrame):
    """Daily log-return correlation among HB_HOUSTON, HB_WEST, Henry Hub."""
    daily_lmp = (
        lmp.set_index("datetime_cst")[["HB_HOUSTON", "HB_WEST"]]
        .resample("D").mean()
        .dropna()
    )
    daily_lmp.index = pd.to_datetime(daily_lmp.index).normalize()

    hh_daily = hh.set_index("date")[["henry_hub_price_mmbtu"]].copy()
    hh_daily.index = pd.to_datetime(hh_daily.index).normalize()

    merged = daily_lmp.join(hh_daily, how="inner").dropna()
    merged.columns = ["HB_HOUSTON", "HB_WEST", "HenryHub"]

    lr_daily = np.log(merged.clip(lower=LMP_FLOOR).pct_change() + 1).dropna()
    corr = lr_daily.corr()
    print(f"\n  Correlation matrix (daily log-returns, n={len(lr_daily)}):")
    print(corr.round(3).to_string())
    return corr, merged



#  MONTE CARLO SIMULATION  (correlated GBM, antithetic variates)

def simulate_correlated_gbm(params: list[dict], corr_matrix: np.ndarray,
                             n_paths: int, n_steps: int, dt: float):
    """
    Simulate N correlated GBM paths via Cholesky decomposition.

    params      : list of dicts (one per series), each with S0, mu_annual, sigma_annual
    corr_matrix : (k×k) correlation matrix
    n_paths     : number of simulation paths
    n_steps     : number of time steps
    dt          : time step in years

    """
    k   = len(params)
    L   = np.linalg.cholesky(corr_matrix)        # Cholesky factor

    # Antithetic variates: generate half, negate for the other half
    half = n_paths // 2
    Z_raw = rng.standard_normal((k, half, n_steps))
    Z     = np.concatenate([Z_raw, -Z_raw], axis=1)  # (k, n_paths, n_steps)

    # Correlate: Z_corr[i] = L @ Z[:,j,t]  → apply along k-axis
    Z_corr = np.einsum("ki,ijn->kjn", L, Z)         # (k, n_paths, n_steps)

    paths = np.zeros((k, n_paths, n_steps + 1))
    for i, p in enumerate(params):
        paths[i, :, 0] = p["S0"]
        drift = (p["mu_annual"] - 0.5 * p["sigma_annual"] ** 2) * dt
        diff  = p["sigma_annual"] * np.sqrt(dt)
        for t in range(n_steps):
            paths[i, :, t + 1] = (
                paths[i, :, t] * np.exp(drift + diff * Z_corr[i, :, t])
            )
        paths[i] = np.maximum(paths[i], LMP_FLOOR)

    return paths



# SPARK SPREAD  &  CASH FLOW

def compute_spark_spread(lmp_paths: np.ndarray, hh_paths: np.ndarray,
                         gas_adder: float):
    """
    Spark spread = LMP − (HH + gas_adder) × HEAT_RATE
    Shape: (n_paths, n_steps+1)
    """
    tolling = (hh_paths + gas_adder) * HEAT_RATE
    return lmp_paths - tolling


def path_statistics(paths_2d: np.ndarray, label: str):
    """
    Summary statistics across paths at the terminal step.
    """
    terminal = paths_2d[:, -1]
    q = np.percentile(terminal, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    var95  = -np.percentile(terminal, 5)
    cvar95 = -terminal[terminal <= np.percentile(terminal, 5)].mean()
    return {
        "Label":    label,
        "Mean":     round(float(np.mean(terminal)),   4),
        "Std":      round(float(np.std(terminal)),    4),
        "Min":      round(float(np.min(terminal)),    4),
        "P1":       round(float(q[0]),  4),
        "P5":       round(float(q[1]),  4),
        "P10":      round(float(q[2]),  4),
        "P25":      round(float(q[3]),  4),
        "Median":   round(float(q[4]),  4),
        "P75":      round(float(q[5]),  4),
        "P90":      round(float(q[6]),  4),
        "P95":      round(float(q[7]),  4),
        "P99":      round(float(q[8]),  4),
        "Max":      round(float(np.max(terminal)),    4),
        "VaR_95":   round(float(var95),  4),
        "CVaR_95":  round(float(cvar95), 4),
        "PctPositive": round(float((terminal > 0).mean() * 100), 2),
    }



# SAVING UTILITIES

def savefig(fig, name: str):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    
    plt.close(fig)


def savecsv(df: pd.DataFrame, name: str):
    path = os.path.join(OUTPUT_DIR, name)
    df.to_csv(path)
    



# PLOTS

def plot_log_returns(calib_results: list[dict]):
    """Log-return distribution + Q-Q plot for each series."""
    for res in calib_results:
        lr   = res["log_returns"]
        lbl  = res["label"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(
            f"GBM Calibration — {lbl}\n"
            f"µ = {res['mu_annual']:+.3f}/yr   σ = {res['sigma_annual']:.3f}/yr   "
            f"n = {res['n_obs']:,}   S₀ = ${res['S0']:.2f}",
            fontsize=11, fontweight="bold"
        )

        # Histogram + normal fit
        ax = axes[0]
        ax.hist(lr, bins=80, density=True, color="steelblue", alpha=0.6,
                label="Observed log-returns")
        x = np.linspace(lr.min(), lr.max(), 300)
        ax.plot(x, stats.norm.pdf(x, lr.mean(), lr.std()), "r-", lw=1.5,
                label="Normal fit")
        ax.set(xlabel="Log Return", ylabel="Density",
               title=f"Distribution  (Shapiro p={res['shapiro_pval']:.3f})")
        ax.legend(fontsize=8)

        # Q-Q plot
        axes[1].set_title("Normal Q-Q Plot")
        stats.probplot(lr, dist="norm", plot=axes[1])
        axes[1].get_lines()[0].set(markersize=2, alpha=0.4)

        plt.tight_layout()
        safe = lbl.replace(" ", "_").replace("/", "-")
        savefig(fig,
            f"GBM_Calibration_{safe}"
            f"_log_returns_distribution_normality_QQ_test.png")


def plot_rolling_params(series: np.ndarray, times, label: str,
                        dt: float, window: int = 168):
    """Rolling µ and σ to visualise parameter stability."""
    mu_r, sig_r = rolling_calibration(series, dt, window)
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    fig.suptitle(
        f"GBM Rolling Parameter Stability — {label}\n"
        f"Window = {window} {'hours' if dt == DT_HOURLY else 'days'}",
        fontsize=11, fontweight="bold"
    )
    t = times[1:]
    axes[0].plot(t, mu_r,  color="steelblue", lw=0.8)
    axes[0].axhline(np.nanmean(mu_r), color="red", ls="--", lw=1,
                    label=f"Mean µ = {np.nanmean(mu_r):.3f}/yr")
    axes[0].set(ylabel="µ (drift, /yr)", title="Rolling Drift µ")
    axes[0].legend(fontsize=8)

    axes[1].plot(t, sig_r, color="darkorange", lw=0.8)
    axes[1].axhline(np.nanmean(sig_r), color="red", ls="--", lw=1,
                    label=f"Mean σ = {np.nanmean(sig_r):.3f}/yr")
    axes[1].set(ylabel="σ (volatility, /yr)", title="Rolling Volatility σ",
                xlabel="Date")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    safe = label.replace(" ", "_").replace("/", "-")
    savefig(fig,
        f"GBM_Calibration_{safe}"
        f"_rolling_drift_volatility_parameter_stability_window{window}.png")


def plot_simulated_paths(paths_2d: np.ndarray, label: str, unit: str,
                         n_show: int = 200, freq: str = "hourly"):
    """Fan chart of simulated paths with mean and confidence bands."""
    n_paths, n_steps = paths_2d.shape
    t = np.linspace(0, HORIZON_YEARS, n_steps)

    p5,  p25, p50 = (np.percentile(paths_2d, q, axis=0) for q in [5, 25, 50])
    p75, p95       = (np.percentile(paths_2d, q, axis=0) for q in [75, 95])
    mean           =  np.mean(paths_2d, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(
        f"GBM Monte Carlo Simulation — {label}\n"
        f"{n_paths:,} paths · {HORIZON_YEARS}-year horizon · {freq} steps",
        fontsize=11, fontweight="bold"
    )

    # Left: fan chart
    ax = axes[0]
    idx = rng.integers(0, n_paths, size=min(n_show, n_paths))
    for i in idx:
        ax.plot(t, paths_2d[i], color="steelblue", alpha=0.04, lw=0.5)
    ax.fill_between(t, p5,  p95, alpha=0.15, color="steelblue", label="5–95%")
    ax.fill_between(t, p25, p75, alpha=0.30, color="steelblue", label="25–75%")
    ax.plot(t, p50,  "steelblue",  lw=1.5, label="Median")
    ax.plot(t, mean, "tomato",     lw=1.5, ls="--", label="Mean")
    ax.set(xlabel="Years forward", ylabel=unit,
           title="Simulated Price Paths (fan chart)")
    ax.legend(fontsize=8)

    # Right: terminal distribution histogram
    ax2 = axes[1]
    terminal = paths_2d[:, -1]
    ax2.hist(terminal, bins=100, density=True, color="steelblue", alpha=0.65)
    ax2.axvline(np.mean(terminal),           color="tomato",   lw=1.5, ls="--",
                label=f"Mean = {np.mean(terminal):.2f}")
    ax2.axvline(np.percentile(terminal,  5), color="orange",   lw=1.2, ls=":",
                label=f"P5 = {np.percentile(terminal,5):.2f}")
    ax2.axvline(np.percentile(terminal, 95), color="orange",   lw=1.2, ls=":",
                label=f"P95 = {np.percentile(terminal,95):.2f}")
    ax2.set(xlabel=unit, ylabel="Density",
            title=f"Terminal Distribution at T={HORIZON_YEARS}yr")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    safe = label.replace(" ", "_").replace("/", "-").replace("$", "")
    savefig(fig,
        f"GBM_MonteCarlo_{safe}"
        f"_{n_paths//1000}kpaths_{HORIZON_YEARS}yr_horizon"
        f"_simulated_paths_CI_terminal_distribution.png")


def plot_spark_spread(ss_paths: np.ndarray, label: str):
    """Spark spread fan chart and profitability analysis."""
    n_paths, n_steps = ss_paths.shape
    t = np.linspace(0, HORIZON_YEARS, n_steps)

    p5  = np.percentile(ss_paths, 5,  axis=0)
    p25 = np.percentile(ss_paths, 25, axis=0)
    p50 = np.percentile(ss_paths, 50, axis=0)
    p75 = np.percentile(ss_paths, 75, axis=0)
    p95 = np.percentile(ss_paths, 95, axis=0)
    mean = np.mean(ss_paths, axis=0)

    pct_positive = (ss_paths > 0).mean(axis=0) * 100
    terminal     = ss_paths[:, -1]

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"GBM Monte Carlo — Spark Spread: {label}\n"
        f"Spark Spread = LMP − (HenryHub + Gas Adder) × {HEAT_RATE} MMBtu/MWh   "
        f"| {n_paths:,} paths · {HORIZON_YEARS}-yr horizon",
        fontsize=11, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1. Fan chart
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.fill_between(t, p5,  p95, alpha=0.15, color="seagreen", label="5–95%")
    ax1.fill_between(t, p25, p75, alpha=0.30, color="seagreen", label="25–75%")
    ax1.plot(t, p50,  "seagreen", lw=1.5,       label="Median")
    ax1.plot(t, mean, "tomato",   lw=1.5, ls="--", label="Mean")
    ax1.axhline(0, color="black", lw=1)
    ax1.fill_between(t, np.minimum(p5, 0), 0, alpha=0.1, color="red")
    ax1.set(xlabel="Years forward", ylabel="$/MWh",
            title="Spark Spread Paths (fan chart)")
    ax1.legend(fontsize=8)

    # 2. Probability of positive spread over time
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(t, pct_positive, color="steelblue", lw=1.2)
    ax2.axhline(50, color="black", lw=0.8, ls="--")
    ax2.set(xlabel="Years forward", ylabel="% of paths",
            title="Probability Spark Spread > 0",
            ylim=[0, 100])

    # 3. Terminal histogram
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(terminal, bins=100, density=True,
             color="seagreen" if terminal.mean() > 0 else "tomato",
             alpha=0.7)
    ax3.axvline(0,                             color="black",  lw=1.2)
    ax3.axvline(np.mean(terminal),             color="tomato", lw=1.5, ls="--",
                label=f"Mean = {np.mean(terminal):.1f}")
    ax3.axvline(np.percentile(terminal,  5),   color="orange", lw=1.2, ls=":",
                label=f"P5 = {np.percentile(terminal,5):.1f}")
    ax3.axvline(np.percentile(terminal, 95),   color="orange", lw=1.2, ls=":",
                label=f"P95 = {np.percentile(terminal,95):.1f}")
    ax3.set(xlabel="$/MWh", ylabel="Density",
            title=f"Terminal Spark Spread Distribution (T={HORIZON_YEARS}yr)")
    ax3.legend(fontsize=7)

    # 4. Annual average spark spread per path
    annual_mean = ss_paths.mean(axis=1)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(annual_mean, bins=100, density=True, color="steelblue", alpha=0.7)
    ax4.axvline(0,                               color="black",  lw=1.2)
    ax4.axvline(np.mean(annual_mean),            color="tomato", lw=1.5, ls="--",
                label=f"Mean = {np.mean(annual_mean):.1f}")
    pct_prof = (annual_mean > 0).mean() * 100
    ax4.set(xlabel="$/MWh", ylabel="Density",
            title=f"Annual Mean Spark Spread per Path\n({pct_prof:.1f}% paths profitable)")
    ax4.legend(fontsize=7)

    # 5. VaR / CVaR bar
    ax5 = fig.add_subplot(gs[1, 2])
    levels = [1, 5, 10, 25]
    var_vals  = [-np.percentile(terminal, lv) for lv in levels]
    cvar_vals = [-terminal[terminal <= np.percentile(terminal, lv)].mean()
                 for lv in levels]
    x = np.arange(len(levels))
    ax5.bar(x - 0.2, var_vals,  width=0.35, label="VaR",  color="steelblue", alpha=0.8)
    ax5.bar(x + 0.2, cvar_vals, width=0.35, label="CVaR", color="tomato",    alpha=0.8)
    ax5.set(xticks=x, xticklabels=[f"{lv}%" for lv in levels],
            xlabel="Confidence Level", ylabel="$/MWh",
            title="VaR & CVaR — Terminal Spark Spread")
    ax5.axhline(0, color="black", lw=0.8)
    ax5.legend(fontsize=8)

    safe = label.replace(" ", "_").replace("/", "-")
    savefig(fig,
        f"GBM_MonteCarlo_SparkSpread_{safe}"
        f"_{n_paths//1000}kpaths_{HORIZON_YEARS}yr_horizon"
        f"_fan_chart_profitability_VaR_CVaR.png")


def plot_correlation_matrix(corr: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle(
        "GBM Calibration — Correlation Matrix\n"
        "Daily Log-Returns: HB_HOUSTON, HB_WEST, Henry Hub Natural Gas",
        fontsize=11, fontweight="bold"
    )
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    labels = list(corr.columns)
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)),
           xticklabels=labels, yticklabels=labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.values[i, j]:.3f}",
                    ha="center", va="center", fontsize=12, fontweight="bold",
                    color="black")
    plt.tight_layout()
    savefig(fig,
        "GBM_Calibration_AllSeries_correlation_matrix"
        "_daily_log_returns_HB-HOUSTON_HB-WEST_HenryHub.png")


def plot_calibration_summary(calib_results: list[dict]):
    """Bar chart comparing µ and σ across all calibrated series."""
    labels = [r["label"] for r in calib_results]
    mus    = [r["mu_annual"]    for r in calib_results]
    sigs   = [r["sigma_annual"] for r in calib_results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        "GBM Calibration — Parameter Comparison Across All Series\n"
        "Annualised MLE Estimates",
        fontsize=11, fontweight="bold"
    )
    colors = ["steelblue" if m >= 0 else "tomato" for m in mus]
    axes[0].bar(labels, mus, color=colors, alpha=0.8)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set(ylabel="µ (drift, /yr)", title="Annual Drift µ")
    for i, v in enumerate(mus):
        axes[0].text(i, v + (0.02 if v >= 0 else -0.08), f"{v:+.3f}",
                     ha="center", fontsize=9)

    axes[1].bar(labels, sigs, color="darkorange", alpha=0.8)
    axes[1].set(ylabel="σ (volatility, /yr)", title="Annual Volatility σ")
    for i, v in enumerate(sigs):
        axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    savefig(fig,
        "GBM_Calibration_AllSeries_parameter_comparison"
        "_annual_drift_volatility_bar_chart.png")



# MAIN

def main():
    
    # Load data 
    
    lmp, hh = load_data()

    hou_prices = lmp["HB_HOUSTON"].values
    wst_prices = lmp["HB_WEST"].values
    hh_prices  = hh["henry_hub_price_mmbtu"].values
    lmp_times  = lmp["datetime_cst"].values
    hh_times   = hh["date"].values

    # GBM calibration
    
    cal_hou = calibrate_gbm(hou_prices, DT_HOURLY, "ERCOT LMP HB_HOUSTON")
    cal_wst = calibrate_gbm(wst_prices, DT_HOURLY, "ERCOT LMP HB_WEST")
    cal_hh  = calibrate_gbm(hh_prices,  DT_DAILY,  "Henry Hub Natural Gas")
    calib_results = [cal_hou, cal_wst, cal_hh]

    # Save calibration parameters table
    param_rows = []
    for r in calib_results:
        param_rows.append({
            "Series":            r["label"],
            "S0":                r["S0"],
            "mu_annual":         r["mu_annual"],
            "sigma_annual":      r["sigma_annual"],
            "mu_daily":          r["mu_daily"],
            "sigma_daily":       r["sigma_daily"],
            "half_life_days":    r["half_life_days"],
            "n_observations":    r["n_obs"],
            "shapiro_stat":      r["shapiro_stat"],
            "shapiro_pval":      r["shapiro_pval"],
            "log_returns_normal_at_5pct": r["normal_at_5pct"],
        })
    savecsv(
        pd.DataFrame(param_rows).set_index("Series"),
        "GBM_Calibration_AllSeries"
        "_parameters_drift_volatility_halflife_normality_MLE_estimates.csv"
    )

    #  Correlation 
    
    corr_df, merged_daily = compute_correlation(lmp, hh)
    savecsv(
        corr_df,
        "GBM_Calibration_AllSeries"
        "_daily_log_return_correlation_matrix_HB-HOUSTON_HB-WEST_HenryHub.csv"
    )

    # Plots: calibration diagnostics 
    
    plot_log_returns(calib_results)
    plot_rolling_params(hou_prices, lmp_times, "ERCOT_LMP_HB_HOUSTON",
                        DT_HOURLY, window=168)
    plot_rolling_params(wst_prices, lmp_times, "ERCOT_LMP_HB_WEST",
                        DT_HOURLY, window=168)
    plot_rolling_params(hh_prices,  hh_times,  "HenryHub_NaturalGas",
                        DT_DAILY,  window=30)
    plot_calibration_summary(calib_results)
    plot_correlation_matrix(corr_df)

    # Monte Carlo simulation
    
    sim_params = [
        dict(S0=cal_hou["S0"], mu_annual=cal_hou["mu_annual"],
             sigma_annual=cal_hou["sigma_annual"]),
        dict(S0=cal_wst["S0"], mu_annual=cal_wst["mu_annual"],
             sigma_annual=cal_wst["sigma_annual"]),
        dict(S0=cal_hh["S0"],  mu_annual=cal_hh["mu_annual"],
             sigma_annual=cal_hh["sigma_annual"]),
    ]

    # Use daily steps for computational speed (hourly would be 8760 steps)
    # but keep the annualised parameters — identical in expectation
   
    paths = simulate_correlated_gbm(
        sim_params, corr_df.values,
        n_paths=N_PATHS, n_steps=N_STEPS_DAILY, dt=DT_DAILY
    )

    hou_paths = paths[0]   # (N_PATHS, N_STEPS_DAILY+1)
    wst_paths = paths[1]
    hh_paths  = paths[2]

    # Spark spreads
    ss_hou = compute_spark_spread(hou_paths, hh_paths, GAS_ADDER_HOUSTON)
    ss_wst = compute_spark_spread(wst_paths, hh_paths, GAS_ADDER_WEST)

    # ── Plots: simulated paths ────────────────────────────────────────────────
    
    plot_simulated_paths(hou_paths, "ERCOT_LMP_HB-HOUSTON",
                         "$/MWh", freq="daily")
    plot_simulated_paths(wst_paths, "ERCOT_LMP_HB-WEST",
                         "$/MWh", freq="daily")
    plot_simulated_paths(hh_paths,  "HenryHub_NaturalGas",
                         "$/MMBtu", freq="daily")
    plot_spark_spread(ss_hou,
        "HB-HOUSTON_HSC-gas_adder-3.00perMMBtu_HR-9.5")
    plot_spark_spread(ss_wst,
        "HB-WEST_Waha-gas_adder-0.00perMMBtu_HR-9.5")

    # Terminal distribution statistics table
    stat_rows = [
        path_statistics(hou_paths, "ERCOT LMP HB_HOUSTON ($/MWh)"),
        path_statistics(wst_paths, "ERCOT LMP HB_WEST ($/MWh)"),
        path_statistics(hh_paths,  "Henry Hub Natural Gas ($/MMBtu)"),
        path_statistics(ss_hou,    "Spark Spread HB_HOUSTON — HSC gas ($/MWh)"),
        path_statistics(ss_wst,    "Spark Spread HB_WEST — Waha gas ($/MWh)"),
    ]
    savecsv(
        pd.DataFrame(stat_rows).set_index("Label"),
        f"GBM_MonteCarlo_AllSeries"
        f"_{N_PATHS//1000}kpaths_{HORIZON_YEARS}yr_horizon"
        f"_terminal_distribution_statistics_mean_std_VaR_CVaR_percentiles.csv"
    )

    # Annual cashflow distribution per hub
    for ss, hub, adder in [
        (ss_hou, "HB-HOUSTON", GAS_ADDER_HOUSTON),
        (ss_wst, "HB-WEST",    GAS_ADDER_WEST),
    ]:
        annual_mean = ss.mean(axis=1)  # average $/MWh over the year per path
        cf_rows = [{
            "Metric":                   "Annual Mean Spark Spread ($/MWh)",
            "Mean":                     round(float(annual_mean.mean()),   4),
            "Std":                      round(float(annual_mean.std()),    4),
            "P5":                       round(float(np.percentile(annual_mean,  5)), 4),
            "P25":                      round(float(np.percentile(annual_mean, 25)), 4),
            "Median":                   round(float(np.percentile(annual_mean, 50)), 4),
            "P75":                      round(float(np.percentile(annual_mean, 75)), 4),
            "P95":                      round(float(np.percentile(annual_mean, 95)), 4),
            "VaR_95":                   round(float(-np.percentile(annual_mean, 5)), 4),
            "CVaR_95":                  round(float(-annual_mean[
                                            annual_mean <= np.percentile(annual_mean, 5)].mean()), 4),
            "Pct_paths_profitable":     round(float((annual_mean > 0).mean() * 100), 2),
            "Gas_adder_per_MMBtu":      adder,
            "Heat_rate_MMBtu_per_MWh":  HEAT_RATE,
            "N_paths":                  N_PATHS,
            "Horizon_years":            HORIZON_YEARS,
        }]
        savecsv(
            pd.DataFrame(cf_rows).set_index("Metric"),
            f"GBM_MonteCarlo_SparkSpread_{hub}"
            f"_annual_cashflow_distribution_statistics"
            f"_gasadder{adder:.2f}_HR{HEAT_RATE}"
            f"_{N_PATHS//1000}kpaths_{HORIZON_YEARS}yr.csv"
        )

    #  Summary 
    
    for f in sorted(os.listdir(OUTPUT_DIR)):
        kb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
        ext = "[csv]" if f.endswith(".csv") else "[img]"
        print(f"  {ext}  {f:<90}  {kb:>6.0f} KB")


if __name__ == "__main__":
    main()
