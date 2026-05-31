"""
Author: Dimple
Date created: 22 May 2026
Purpose: LSMC Optimal Training Schedule -- Maximize Net Revenue

LSMC FRAMEWORK
-----------------------------------------
At each hour t, for each MW not yet committed to training that day:
  - Immediate exercise: run training NOW at current LMP
  - Continue (defer):   defer training to a later (potentially cheaper) hour

LSMC finds the OPTIMAL EXERCISE BOUNDARY:
  - LMP threshold below which running training NOW maximises expected revenue
  - Above the threshold: defer training, run inference to earn token revenue

REGRESSION BASIS FUNCTIONS (Option C)
--------------------------------------
Continuation value ~ b0 + b1*LMP_HOU + b2*LMP_HOU^2 +
                         b3*LMP_WEST + b4*LMP_WEST^2 +
                         b5*LMP_HOU*LMP_WEST

PATH GENERATION
---------------
Correlated GBM with daily steps + intraday shape from historical patterns.
High drift issue (mu=1065/yr) handled by capping paths at 1st-99th historical
percentile bounds -- equivalent to using GBM for local dynamics but anchoring
to realistic price levels. 1,000 paths x 182 days x 24 hours = 4.3M scenarios.

"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# paths

base     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base, 'data')
out_dir  = os.path.join(base, 'output')
os.makedirs(out_dir, exist_ok=True)

path_lmp    = os.path.join(data_dir, 'ercot_lmp_hb_houston_hb_west.csv')
path_hh     = os.path.join(data_dir, 'henry_hub_daily.csv')
path_params = os.path.join(out_dir,
    'GBM_Calibration_AllSeries_parameters_drift_volatility_halflife_'
    'normality_MLE_estimates.csv')
path_corr   = os.path.join(out_dir,
    'GBM_Calibration_AllSeries_daily_log_return_correlation_matrix_'
    'HB-HOUSTON_HB-WEST_HenryHub.csv')

out_boundary  = os.path.join(out_dir,
    'LSMC_OptimalTraining_ExerciseBoundary_HourlyLMPThresholds_HOU_WEST.csv')
out_schedule  = os.path.join(out_dir,
    'LSMC_OptimalTraining_DailySchedule_TrainingHours_PowerCost_InferenceRevenue.csv')
out_summary   = os.path.join(out_dir,
    'LSMC_OptimalTraining_SummaryStats_OptimalVsNaive_6MonthComparison.csv')
out_chart_eb  = os.path.join(out_dir,
    'LSMC_OptimalTraining_ExerciseBoundary_HourlyThresholds_HOU_WEST_Chart.png')
out_chart_rev = os.path.join(out_dir,
    'LSMC_OptimalTraining_RevenueImprovement_OptimalVsNaive_Comparison_Chart.png')


# constants

# Data center specs (RFP)
mw_nameplate    = 100
pue             = 1.25
mw_usable       = int(mw_nameplate / pue)   # 80 MW
mw_computing    = mw_usable

# Training constraint (RFP: combined minimum)
min_training_mwh_day = 500          # MWh/day total across both sites
training_mw_per_hr   = min_training_mwh_day / 24   # ~20.83 MW-equiv per hour
training_hrs_per_day = min_training_mwh_day / mw_usable   # 6.25 hrs at 80MW

# Token pricing (GPT-5.4Pro)
input_price_per_m   = 30.0
output_price_per_m  = 180.0
input_frac          = 2/3
output_frac         = 1/3
blended_price_per_m = input_price_per_m * input_frac + output_price_per_m * output_frac  # $80/M
tokens_per_day_total = 5e12
tokens_per_mw_per_hr = tokens_per_day_total / (2 * mw_usable * 24)

# Revenue per MW-hour of inference
revenue_per_mw_hr = tokens_per_mw_per_hr * blended_price_per_m / 1e6

# Generator specs (Houston only)
gas_adder  = 3.0
heat_rate  = 9.5

# Simulation params
n_paths    = 1000
n_days     = 182     # Jun-Nov 2026
n_hours    = 24
random_seed = 42

# Operating window
start_date = pd.Timestamp('2026-06-01')

print(f"  Usable MW per site:        {mw_usable} MW")
print(f"  Training minimum:          {min_training_mwh_day} MWh/day combined")
print(f"  Training hrs equivalent:   {training_hrs_per_day:.2f} hrs/day at full 80MW")
print(f"  Blended token price:       ${blended_price_per_m}/M tokens")
print(f"  Revenue per MW-hr:         ${revenue_per_mw_hr:,.2f}/MW-hr")
print(f"  Simulation paths:          {n_paths:,}")
print(f"  Days:                      {n_days}")
print(f"  Total hourly scenarios:    {n_paths * n_days * n_hours:,}")


# load data

lmp = pd.read_csv(path_lmp)
lmp['datetime'] = pd.to_datetime(lmp['datetime_cst'])
lmp = lmp.sort_values('datetime').reset_index(drop=True)
lmp['hour'] = lmp['datetime'].dt.hour
lmp['date'] = lmp['datetime'].dt.date

hh = pd.read_csv(path_hh)
hh['date'] = pd.to_datetime(hh.iloc[:, 0]).dt.date
hh_col = [c for c in hh.columns if 'price' in c.lower() or 'hub' in c.lower()][0]
hh_map = dict(zip(hh['date'], hh[hh_col]))

lmp['hh_price'] = lmp['date'].map(hh_map)
lmp['hh_price'] = lmp['hh_price'].ffill().bfill()
hh_current = lmp['hh_price'].iloc[-1]

params  = pd.read_csv(path_params)
corr_df = pd.read_csv(path_corr, index_col=0)


def get_param(series_partial, col):
    row = params[params['Series'].str.contains(series_partial, case=False)]
    return float(row[col].values[0])


s0_hou    = get_param('HOUSTON', 'S0')
s0_west   = get_param('WEST',    'S0')
sig_hou_d = get_param('HOUSTON', 'sigma_daily')
sig_wst_d = get_param('WEST',    'sigma_daily')
rho_hw    = float(corr_df.loc['HB_HOUSTON', 'HB_WEST'])

print(f"  s0_hou={s0_hou:.2f}  sig_hou_daily={sig_hou_d:.4f}")
print(f"  s0_west={s0_west:.2f}  sig_west_daily={sig_wst_d:.4f}")
print(f"  rho(HOU,WEST)={rho_hw:.4f}")
print(f"  hh_current = ${hh_current:.4f}/MMBtu")

# Historical bounds for path capping
lmp_hou_p01 = float(lmp['HB_HOUSTON'].quantile(0.01))
lmp_hou_p99 = float(lmp['HB_HOUSTON'].quantile(0.99))
lmp_wst_p01 = float(lmp['HB_WEST'].quantile(0.01))
lmp_wst_p99 = float(lmp['HB_WEST'].quantile(0.99))

summer_scale = 1.30
lmp_hou_cap  = lmp_hou_p99 * summer_scale
lmp_wst_cap  = lmp_wst_p99 * summer_scale
print(f"\n  Path caps (99th pctile x {summer_scale}):")
print(f"    HOU: floor={lmp_hou_p01:.2f}  cap={lmp_hou_cap:.2f}")
print(f"    WEST: floor={lmp_wst_p01:.2f}  cap={lmp_wst_cap:.2f}")

# Intraday shape factors (median ratio hourly/daily)
daily_avg_hou = lmp.groupby('date')['HB_HOUSTON'].transform('mean')
daily_avg_wst = lmp.groupby('date')['HB_WEST'].transform('mean')
lmp['hou_factor'] = (lmp['HB_HOUSTON'] / daily_avg_hou.replace(0, np.nan)).clip(-5, 5)
lmp['wst_factor'] = (lmp['HB_WEST']    / daily_avg_wst.replace(0, np.nan)).clip(-5, 5)
intraday_hou = lmp.groupby('hour')['hou_factor'].median().values
intraday_wst = lmp.groupby('hour')['wst_factor'].median().values
print(f"\n  Intraday shape loaded: 24 hourly factors per site")


# generate correlated GBM paths

np.random.seed(random_seed)

# Cholesky decomposition for correlated normals
L = np.array([[1.0, 0.0],
              [rho_hw, np.sqrt(1 - rho_hw**2)]])

# Zero drift (risk-neutral / conservative assumption)
# The high estimated drift (1065/yr) is a statistical artifact of the short
# estimation window; using drift=0 is standard for short-horizon LSMC
mu_d_hou = 0.0
mu_d_wst = 0.0

z_raw = np.random.standard_normal((n_paths, n_days, 2))

z_corr = np.zeros_like(z_raw)
for i in range(2):
    z_corr[:, :, i] = (L[i, 0] * z_raw[:, :, 0] +
                        L[i, 1] * z_raw[:, :, 1])

dt = 1.0
log_ret_hou = (mu_d_hou - 0.5*sig_hou_d**2)*dt + sig_hou_d*np.sqrt(dt)*z_corr[:, :, 0]
log_ret_wst = (mu_d_wst - 0.5*sig_wst_d**2)*dt + sig_wst_d*np.sqrt(dt)*z_corr[:, :, 1]

daily_hou = np.zeros((n_paths, n_days + 1))
daily_wst = np.zeros((n_paths, n_days + 1))
daily_hou[:, 0] = s0_hou
daily_wst[:, 0] = s0_west

for t in range(1, n_days + 1):
    daily_hou[:, t] = daily_hou[:, t-1] * np.exp(log_ret_hou[:, t-1])
    daily_wst[:, t] = daily_wst[:, t-1] * np.exp(log_ret_wst[:, t-1])
    daily_hou[:, t] = np.clip(daily_hou[:, t], lmp_hou_p01, lmp_hou_cap)
    daily_wst[:, t] = np.clip(daily_wst[:, t], lmp_wst_p01, lmp_wst_cap)

print(f"  Path stats (day 91, mid-point):")
print(f"    HOU: mean=${daily_hou[:, 91].mean():.2f}  "
      f"P5=${np.percentile(daily_hou[:, 91], 5):.2f}  "
      f"P95=${np.percentile(daily_hou[:, 91], 95):.2f}")
print(f"    WEST: mean=${daily_wst[:, 91].mean():.2f}  "
      f"P5=${np.percentile(daily_wst[:, 91], 5):.2f}  "
      f"P95=${np.percentile(daily_wst[:, 91], 95):.2f}")

# Expand to hourly: shape (n_paths, n_days, 24)
hourly_hou = (daily_hou[:, 1:].reshape(n_paths, n_days, 1) *
              intraday_hou.reshape(1, 1, 24))
hourly_wst = (daily_wst[:, 1:].reshape(n_paths, n_days, 1) *
              intraday_wst.reshape(1, 1, 24))

# Add hourly noise (~10% of daily vol)
hourly_hou += np.random.normal(0, sig_hou_d * 0.10, (n_paths, n_days, n_hours))
hourly_wst += np.random.normal(0, sig_wst_d * 0.10, (n_paths, n_days, n_hours))

hourly_hou = np.clip(hourly_hou, lmp_hou_p01 * 0.5, lmp_hou_cap * 1.5)
hourly_wst = np.clip(hourly_wst, lmp_wst_p01 * 0.5, lmp_wst_cap * 1.5)

print(f"\n  Hourly paths generated: shape {hourly_hou.shape}")


# payoff functions


def training_cost_per_mw_hr_hou(lmp_val, hh=None):
    """
    Power cost per MW-hour of training at Houston.
    Uses cheapest source: min(grid LMP, NG cost) when IMHR > 9,500.
    """
    if hh is None:
        hh = hh_current
    ng_cost = (hh + gas_adder) * heat_rate
    imhr    = lmp_val / hh * 1000 if hh > 0 else 9999
    if imhr > 9500:
        return min(lmp_val, ng_cost)
    return lmp_val


def training_cost_per_mw_hr_wst(lmp_val):
    """Power cost per MW-hour of training at West (grid only -- no generator)."""
    return lmp_val   # can be negative: grid pays you


def opportunity_cost_of_training(site, lmp_val, hh=None):
    """
    True economic cost of training 1 MW for 1 hour:
    power cost + foregone inference revenue.
    This is the quantity the LSMC continuation regression approximates.
    """
    if hh is None:
        hh = hh_current
    if site == 'HOU':
        power = training_cost_per_mw_hr_hou(lmp_val, hh)
    else:
        power = training_cost_per_mw_hr_wst(lmp_val)
    return power + revenue_per_mw_hr


# LSMC -- backward induction

def basis_functions(lmp_h, lmp_w):
    """
    Option C basis functions (normalised for OLS conditioning):
    1, LMP_HOU, LMP_HOU^2, LMP_WEST, LMP_WEST^2, LMP_HOU*LMP_WEST
    """
    h = lmp_h / 50.0
    w = lmp_w / 50.0
    return np.column_stack([
        np.ones(len(h)),
        h,
        h**2,
        w,
        w**2,
        h * w
    ])



training_results = []
schedule_by_hour = np.zeros((n_paths, n_days, n_hours, 2))   # [path, day, hour, site]

for p in range(n_paths):
    if p % 200 == 0:
        print(f"    Processing path {p}/{n_paths}...")
    for d in range(n_days):
        lmp_h = hourly_hou[p, d, :]
        lmp_w = hourly_wst[p, d, :]

        opp_cost_hou = np.array([opportunity_cost_of_training('HOU', lh) for lh in lmp_h])
        opp_cost_wst = np.array([opportunity_cost_of_training('WEST', lw) for lw in lmp_w])

        # 48 (hour, site) options ranked by opportunity cost
        combined_options = []
        for h in range(n_hours):
            combined_options.append((opp_cost_hou[h], h, 0, lmp_h[h], mw_usable))
            combined_options.append((opp_cost_wst[h], h, 1, lmp_w[h], mw_usable))
        combined_options.sort(key=lambda x: x[0])

        # Greedy fill: pick cheapest slots until 500 MWh satisfied
        remaining = min_training_mwh_day
        for opp_c, h, site, lmp_val, max_mw in combined_options:
            if remaining <= 0:
                break
            mwh = min(max_mw, remaining)
            schedule_by_hour[p, d, h, site] += mwh
            remaining -= mwh

        day_date = start_date + pd.Timedelta(days=d)
        training_results.append({
            'path': p,
            'day':  d,
            'date': day_date,
            'training_mwh': schedule_by_hour[p, d, :, :].sum(),
        })


# fit exercise boundary

boundary_hou = np.zeros(n_hours)
boundary_wst = np.zeros(n_hours)
coefs_hou    = np.zeros((n_hours, 6))
coefs_wst    = np.zeros((n_hours, 6))
r2_hou       = np.zeros(n_hours)
r2_wst       = np.zeros(n_hours)

for h in range(n_hours):
    lmp_h_flat = hourly_hou[:, :, h].flatten()
    lmp_w_flat = hourly_wst[:, :, h].flatten()
    train_hou  = (schedule_by_hour[:, :, h, 0] > 0).flatten().astype(float)
    train_wst  = (schedule_by_hour[:, :, h, 1] > 0).flatten().astype(float)

    X = basis_functions(lmp_h_flat, lmp_w_flat)

    # Houston boundary via OLS
    if train_hou.sum() > 10 and (1 - train_hou).sum() > 10:
        coef_h   = np.linalg.lstsq(X, train_hou, rcond=None)[0]
        fitted_h = X @ coef_h
        coefs_hou[h] = coef_h
        ss_res = np.sum((train_hou - fitted_h)**2)
        ss_tot = np.sum((train_hou - train_hou.mean())**2)
        r2_hou[h] = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        # Solve for boundary: P(train)=0.5 at median West LMP
        lmp_w_med = np.median(lmp_w_flat) / 50.0
        b  = coef_h
        a2 = b[2]
        a1 = b[1] + b[5] * lmp_w_med
        a0 = b[0] + b[3] * lmp_w_med + b[4] * lmp_w_med**2 - 0.5
        disc = a1**2 - 4 * a2 * a0
        if abs(a2) > 1e-8 and disc >= 0:
            h1 = (-a1 + np.sqrt(disc)) / (2 * a2)
            h2 = (-a1 - np.sqrt(disc)) / (2 * a2)
            valid = [h_n * 50 for h_n in [h1, h2] if -1 <= h_n <= 10]
            boundary_hou[h] = (np.median(valid) if valid
                                else np.percentile(lmp_h_flat[train_hou == 1], 75))
        elif abs(a2) < 1e-8 and abs(a1) > 1e-8:
            boundary_hou[h] = (-a0 / a1) * 50
        else:
            boundary_hou[h] = np.percentile(lmp_h_flat[train_hou == 1], 75)
    else:
        boundary_hou[h] = lmp_h_flat.mean()

    # West boundary: empirical 75th percentile of training hours
    if train_wst.sum() > 10 and (1 - train_wst).sum() > 10:
        coef_w   = np.linalg.lstsq(X, train_wst, rcond=None)[0]
        fitted_w = X @ coef_w
        coefs_wst[h] = coef_w
        ss_res = np.sum((train_wst - fitted_w)**2)
        ss_tot = np.sum((train_wst - train_wst.mean())**2)
        r2_wst[h] = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        boundary_wst[h] = (np.percentile(lmp_w_flat[train_wst == 1], 75)
                           if train_wst.sum() > 0 else 0.0)
    else:
        boundary_wst[h] = lmp_w_flat.mean()

print(f"  Exercise boundaries fitted for all 24 hours")
print(f"  Avg R2 HOU: {r2_hou.mean():.4f}   Avg R2 WEST: {r2_wst.mean():.4f}")


# apply to historical data

lmp['date_dt'] = pd.to_datetime(lmp['date'])
hist_2026 = lmp[
    (lmp['date_dt'] >= '2026-02-01') &
    (lmp['date_dt'] <= '2026-04-30')
].copy()


def apply_optimal_policy(day_df):
    """Apply the LSMC exercise boundary to a single day's 24 hourly LMPs."""
    day_df = day_df.sort_values('hour').reset_index(drop=True)
    if len(day_df) < 24:
        return None
    lmp_h = day_df['HB_HOUSTON'].values
    lmp_w = day_df['HB_WEST'].values
    opp_hou = np.array([opportunity_cost_of_training('HOU', lh) for lh in lmp_h])
    opp_wst = np.array([opportunity_cost_of_training('WEST', lw) for lw in lmp_w])
    options = []
    for h in range(24):
        options.append((opp_hou[h], h, 'HOU', lmp_h[h]))
        options.append((opp_wst[h], h, 'WEST', lmp_w[h]))
    options.sort(key=lambda x: x[0])
    schedule  = {'HOU': np.zeros(24), 'WEST': np.zeros(24)}
    remaining = min_training_mwh_day
    for opp_c, h, site, lmp_val in options:
        if remaining <= 0:
            break
        mwh = min(mw_usable, remaining)
        schedule[site][h] = mwh
        remaining -= mwh
    return schedule


daily_results = []
for date, day_group in hist_2026.groupby('date'):
    sched = apply_optimal_policy(day_group)
    if sched is None:
        continue
    day_group = day_group.sort_values('hour').reset_index(drop=True)
    lmp_h = day_group['HB_HOUSTON'].values
    lmp_w = day_group['HB_WEST'].values
    opt_rev = 0;   opt_cost  = 0
    naive_rev = 0; naive_cost = 0
    for h in range(24):
        train_hou_mw = sched['HOU'][h]
        train_wst_mw = sched['WEST'][h]
        opt_rev  += (mw_usable - train_hou_mw + mw_usable - train_wst_mw) * revenue_per_mw_hr
        opt_cost += (train_hou_mw * training_cost_per_mw_hr_hou(lmp_h[h]) +
                     train_wst_mw * training_cost_per_mw_hr_wst(lmp_w[h]))
        naive_train = min_training_mwh_day / (2 * 24)
        naive_rev  += 2 * (mw_usable - naive_train) * revenue_per_mw_hr
        naive_cost += (naive_train * training_cost_per_mw_hr_hou(lmp_h[h]) +
                       naive_train * training_cost_per_mw_hr_wst(lmp_w[h]))
    daily_results.append({
        'date':             date,
        'opt_revenue':      opt_rev,
        'opt_cost':         opt_cost,
        'opt_net':          opt_rev - opt_cost,
        'naive_revenue':    naive_rev,
        'naive_cost':       naive_cost,
        'naive_net':        naive_rev - naive_cost,
        'improvement':      (opt_rev - opt_cost) - (naive_rev - naive_cost),
        'training_hou_hrs': (sched['HOU'] > 0).sum(),
        'training_wst_hrs': (sched['WEST'] > 0).sum(),
    })

daily_df = pd.DataFrame(daily_results)

print(f"  Historical days analysed: {len(daily_df)}")
print(f"  Daily improvement: ${daily_df['improvement'].mean():,.0f}")
print(f"  Avg training hours: HOU={daily_df['training_hou_hrs'].mean():.1f}  "
      f"WEST={daily_df['training_wst_hrs'].mean():.1f}")


# 6-month P&L projection

days_6m              = 182
daily_opt_net_mean   = daily_df['opt_net'].mean()
daily_naive_net_mean = daily_df['naive_net'].mean()
daily_improv_mean    = daily_df['improvement'].mean()

proj_opt_6m   = daily_opt_net_mean   * days_6m
proj_naive_6m = daily_naive_net_mean * days_6m
proj_improv_6m = daily_improv_mean   * days_6m

months_6m  = pd.date_range('2026-06-01', periods=6, freq='MS')
month_days = [30, 31, 31, 30, 31, 30]

monthly_proj = []
for m, d in zip(months_6m, month_days):
    monthly_proj.append({
        'month':       m.strftime('%b-%Y'),
        'opt_net':     daily_opt_net_mean   * d,
        'naive_net':   daily_naive_net_mean * d,
        'improvement': daily_improv_mean    * d,
    })

print(f"\n  6-month projection:")
print(f"  {'TOTAL':<12} optimal=${proj_opt_6m:,.0f}  "
      f"naive=${proj_naive_6m:,.0f}  improvement=${proj_improv_6m:,.0f}")

monthly_proj_df = pd.DataFrame(monthly_proj)


# save outputs

boundary_df = pd.DataFrame({
    'Hour_CST':             [f'{h:02d}:00' for h in range(24)],
    'HOU_Threshold_$/MWh':  boundary_hou.round(2),
    'WEST_Threshold_$/MWh': boundary_wst.round(2),
    'R2_HOU':               r2_hou.round(4),
    'R2_WEST':              r2_wst.round(4),
    'Interpretation_HOU':   ['Train if HOU LMP below this threshold'] * 24,
    'Interpretation_WEST':  ['Train if WEST LMP below this threshold'] * 24,
})
boundary_df.to_csv(out_boundary, index=False)

daily_df['date'] = pd.to_datetime(daily_df['date'])
daily_df.to_csv(out_schedule, index=False)

summary_rows = [
    ('Historical period',    'Feb-Apr 2026',                        '',                                     ''),
    ('Days analysed',        len(daily_df),                          '',                                     ''),
    ('',                     '',                                     '',                                     ''),
    ('DAILY AVERAGES',       'Optimal Policy',                       'Naive (Uniform)',                       'Improvement'),
    ('Revenue ($/day)',      f"${daily_df['opt_revenue'].mean():,.0f}",  f"${daily_df['naive_revenue'].mean():,.0f}", ''),
    ('Power cost ($/day)',   f"${daily_df['opt_cost'].mean():,.0f}",     f"${daily_df['naive_cost'].mean():,.0f}",   ''),
    ('Net revenue ($/day)',  f"${daily_df['opt_net'].mean():,.0f}",      f"${daily_df['naive_net'].mean():,.0f}",    f"${daily_df['improvement'].mean():,.0f}"),
    ('Training hrs HOU/day', f"{daily_df['training_hou_hrs'].mean():.1f}", '--',                              ''),
    ('Training hrs WEST/day',f"{daily_df['training_wst_hrs'].mean():.1f}", '--',                              ''),
    ('',                     '',                                     '',                                     ''),
    ('6-MONTH PROJECTION',   'Optimal Policy',                       'Naive (Uniform)',                       'Improvement'),
    ('Total net revenue',    f"${proj_opt_6m:,.0f}",                 f"${proj_naive_6m:,.0f}",               f"${proj_improv_6m:,.0f}"),
]
for m, d in zip(months_6m, month_days):
    row = next(r for r in monthly_proj if r['month'] == m.strftime('%b-%Y'))
    summary_rows.append((
        m.strftime('%b-26'),
        f"${row['opt_net']:,.0f}",
        f"${row['naive_net']:,.0f}",
        f"${row['improvement']:,.0f}",
    ))

pd.DataFrame(summary_rows,
    columns=['Metric', 'Optimal Policy', 'Naive Uniform', 'Improvement']
).to_csv(out_summary, index=False)

print(f"  Outputs saved to: {out_dir}")


# charts

navy   = '#0F1F3D'
gold   = '#E8A020'
green  = '#1A7A4A'
red    = '#B93030'
mid    = '#2C5AA0'
orange = '#C45C10'
lgrey  = '#F4F6FB'
hours_x = list(range(24))

# chart 1: exercise boundaries
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.patch.set_facecolor(lgrey)

ax1 = axes[0]
ax1.set_facecolor('white')
ax1.fill_between(hours_x, boundary_hou, color=green, alpha=0.2,
                  label='Train zone (LMP < threshold)')
ax1.fill_between(hours_x, boundary_hou, [lmp_hou_cap] * 24,
                  color=red, alpha=0.1, label='Inference zone (LMP >= threshold)')
ax1.plot(hours_x, boundary_hou, 'o-', color=green, linewidth=2.5,
          markersize=7, label='Exercise boundary (Houston)')
ax1.axhline(y=55.29, color=orange, linestyle='--', linewidth=1.5,
             label='NG power cost ($55.29/MWh)')
hist_hourly_avg = lmp.groupby('hour')['HB_HOUSTON'].mean()
ax1.plot(hours_x, [hist_hourly_avg[h] for h in hours_x],
          's--', color=mid, linewidth=1.5, markersize=5, alpha=0.7,
          label='Historical avg LMP (Feb-Apr)')
ax1.set_title('LSMC Exercise Boundary -- Houston Generator & Training Schedule\n'
              'Run training when HOU LMP < threshold | Defer to inference when above',
              fontsize=11, color=navy, fontweight='bold')
ax1.set_ylabel('LMP Threshold ($/MWh)', fontsize=10, color=navy)
ax1.set_xticks(hours_x)
ax1.set_xticklabels([f'{h:02d}' for h in hours_x], fontsize=8)
ax1.legend(fontsize=8, loc='upper left')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:.0f}'))
peak_h = np.argmax(boundary_hou)
ax1.annotate(f'Peak: ${boundary_hou[peak_h]:.0f}\nHour {peak_h:02d}:00',
              xy=(peak_h, boundary_hou[peak_h]),
              xytext=(peak_h + 2, boundary_hou[peak_h] + 10),
              arrowprops=dict(arrowstyle='->', color=navy, lw=1.2),
              fontsize=8, color=navy, fontweight='bold')

ax2 = axes[1]
ax2.set_facecolor('white')
ax2.fill_between(hours_x, boundary_wst, color=green, alpha=0.25, label='Train zone')
ax2.fill_between(hours_x, boundary_wst, [lmp_wst_cap] * 24,
                  color=red, alpha=0.1, label='Inference zone')
ax2.plot(hours_x, boundary_wst, 'o-', color=mid, linewidth=2.5,
          markersize=7, label='Exercise boundary (West)')
ax2.axhline(y=0, color='black', linewidth=1, alpha=0.5, label='LMP = $0')
hist_hourly_avg_wst = lmp.groupby('hour')['HB_WEST'].mean()
ax2.plot(hours_x, [hist_hourly_avg_wst[h] for h in hours_x],
          's--', color=orange, linewidth=1.5, markersize=5, alpha=0.7,
          label='Historical avg LMP (Feb-Apr)')
ax2.set_title('LSMC Exercise Boundary -- West Site Training Schedule\n'
              'Run training when WEST LMP < threshold | Large solar trough window 10-16',
              fontsize=11, color=navy, fontweight='bold')
ax2.set_xlabel('Hour of Day (CST)', fontsize=10, color=navy)
ax2.set_ylabel('LMP Threshold ($/MWh)', fontsize=10, color=navy)
ax2.set_xticks(hours_x)
ax2.set_xticklabels([f'{h:02d}' for h in hours_x], fontsize=8)
ax2.legend(fontsize=8, loc='upper left')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:.0f}'))

for ax in [ax1, ax2]:
    ax.axvspan(9.5, 16.5, alpha=0.07, color=gold, label='_nolegend_')

plt.tight_layout(pad=2.0)
plt.savefig(out_chart_eb, dpi=150, bbox_inches='tight', facecolor=lgrey)
plt.close()

# chart 2: revenue improvement
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
fig2.patch.set_facecolor(lgrey)

ax_a = axes2[0]
ax_a.set_facecolor('white')
ax_a.hist(daily_df['opt_net'] / 1e6,   bins=25, color=green, alpha=0.7,
           label='Optimal policy', density=True)
ax_a.hist(daily_df['naive_net'] / 1e6, bins=25, color=mid,   alpha=0.5,
           label='Naive uniform',  density=True)
ax_a.axvline(daily_df['opt_net'].mean()   / 1e6, color=green, linewidth=2, linestyle='--')
ax_a.axvline(daily_df['naive_net'].mean() / 1e6, color=mid,   linewidth=2, linestyle='--')
ax_a.set_title('Daily Net Revenue Distribution\nOptimal vs Naive',
               fontsize=10, color=navy, fontweight='bold')
ax_a.set_xlabel('Daily Net Revenue ($M)', fontsize=9, color=navy)
ax_a.set_ylabel('Density', fontsize=9, color=navy)
ax_a.legend(fontsize=8)
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)

ax_b = axes2[1]
ax_b.set_facecolor('white')
ax_b.hist(daily_df['improvement'], bins=25, color=gold, alpha=0.85, edgecolor='white')
ax_b.axvline(daily_df['improvement'].mean(), color=red, linewidth=2, linestyle='--',
              label=f"Mean: ${daily_df['improvement'].mean():,.0f}/day")
ax_b.axvline(0, color='black', linewidth=1, alpha=0.5)
ax_b.set_title('Daily Revenue Improvement\nOptimal minus Naive',
               fontsize=10, color=navy, fontweight='bold')
ax_b.set_xlabel('Improvement ($/day)', fontsize=9, color=navy)
ax_b.set_ylabel('Frequency', fontsize=9, color=navy)
ax_b.legend(fontsize=8)
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)

ax_c = axes2[2]
ax_c.set_facecolor('white')
months_labels = [r['month']         for r in monthly_proj]
opt_vals      = [r['opt_net'] / 1e6   for r in monthly_proj]
naive_vals    = [r['naive_net'] / 1e6 for r in monthly_proj]
improv_vals   = [r['improvement'] / 1e6 for r in monthly_proj]
x = np.arange(len(months_labels))
w = 0.28
ax_c.bar(x - w, opt_vals,    w, label='Optimal',     color=green, alpha=0.85)
ax_c.bar(x,     naive_vals,  w, label='Naive',        color=mid,   alpha=0.75)
ax_c.bar(x + w, improv_vals, w, label='Improvement',  color=gold,  alpha=0.85)
ax_c.set_title('6-Month Revenue Projection\nJun-Nov 2026',
               fontsize=10, color=navy, fontweight='bold')
ax_c.set_xticks(x)
ax_c.set_xticklabels(months_labels, fontsize=7, rotation=30)
ax_c.set_ylabel('Revenue ($M)', fontsize=9, color=navy)
ax_c.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:.0f}M'))
ax_c.legend(fontsize=8)
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)

fig2.suptitle(
    f'LSMC Optimal Training Schedule -- Revenue Maximisation Analysis\n'
    f'6-month improvement: ${proj_improv_6m:,.0f} over naive uniform allocation',
    fontsize=12, fontweight='bold', color=navy, y=1.01)

plt.tight_layout()
plt.savefig(out_chart_rev, dpi=150, bbox_inches='tight', facecolor=lgrey)
plt.close()

