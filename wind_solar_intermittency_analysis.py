"""
Author: Dimple
Date created on 26 May 2026
Purpose: Wind & Solar Intermittency Analysis — ERCOT West Zone
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')


# PATHS


BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, 'data')
OUT_DIR  = os.path.join(BASE, 'output')

# Input files
PATH_WIND   = os.path.join(DATA_DIR, 'ercot_wind_hourly.csv')
PATH_SOLAR  = os.path.join(DATA_DIR, 'ercot_solar_hourly.csv')
PATH_LMP    = os.path.join(DATA_DIR, 'ercot_lmp_hb_houston_hb_west.csv')
PATH_HH     = os.path.join(DATA_DIR, 'henry_hub_daily.csv')

# Output files — comprehensive descriptive names
OUT_SCATTER = os.path.join(OUT_DIR, 'ERCOT_West_Intermittency_ScatterPlots_RenewableGen_vs_LMP_West_HourOfDay.png')
OUT_CHARTS  = os.path.join(OUT_DIR, 'ERCOT_West_Intermittency_GenerationProfile_NegativePricingPatterns_Hourly.png')
OUT_CSV_THR = os.path.join(OUT_DIR, 'ERCOT_West_Intermittency_ThresholdTable_GenerationRegime_NegativePriceFrequency.csv')
OUT_CSV_HLY = os.path.join(OUT_DIR, 'ERCOT_West_Intermittency_HourlyAggregates_WindMW_SolarMW_MeanLMP_PctNegative.csv')

# Create output directory if it doesn't exist
os.makedirs(OUT_DIR, exist_ok=True)


# COLOURS

NAVY    = '#0F1F3D'
GOLD    = '#E8A020'
GREEN   = '#1A7A4A'
RED     = '#B93030'
MID     = '#2C5AA0'
ORANGE  = '#C45C10'
PURPLE  = '#5030A0'
LGREY   = '#F4F6FB'
DKGREY  = '#3A4560'

def hour_color(h):
    if 10 <= h <= 16:   return GOLD
    elif 19 <= h <= 21: return RED
    elif h <= 6:        return MID
    else:               return ORANGE


# LOAD DATA

wind  = pd.read_csv(PATH_WIND)
solar = pd.read_csv(PATH_SOLAR)
lmp   = pd.read_csv(PATH_LMP)
hh    = pd.read_csv(PATH_HH)

# Parse datetimes
wind['datetime']  = pd.to_datetime(wind['datetime_cst'])
solar['datetime'] = pd.to_datetime(solar['datetime_cst'])
lmp['datetime']   = pd.to_datetime(lmp['datetime_cst'])
hh['date']        = pd.to_datetime(hh.iloc[:, 0])
hh.columns        = ['date_raw', 'hh_price'] if len(hh.columns) == 2 else list(hh.columns)
hh['date']        = pd.to_datetime(hh.iloc[:, 0])
hh_price_col      = [c for c in hh.columns if 'price' in c.lower() or 'hub' in c.lower() or 'hh' in c.lower()][0]

# Deduplicate wind and solar (both contain many duplicate rows from ERCOT download)
wind  = wind.drop_duplicates('datetime').sort_values('datetime').reset_index(drop=True)
solar = solar.drop_duplicates('datetime').sort_values('datetime').reset_index(drop=True)
lmp   = lmp.sort_values('datetime').reset_index(drop=True)

print(f"Wind  rows after dedup : {len(wind):,}  |  "
      f"{wind['datetime'].min().date()} → {wind['datetime'].max().date()}")
print(f"Solar rows after dedup : {len(solar):,}  |  "
      f"{solar['datetime'].min().date()} → {solar['datetime'].max().date()}")
print(f"LMP   rows             : {len(lmp):,}  |  "
      f"{lmp['datetime'].min().date()} → {lmp['datetime'].max().date()}")


# FIX MISSING HENRY HUB PRICES (forward-fill weekends/holidays)


lmp['date'] = lmp['datetime'].dt.date
hh_map = dict(zip(pd.to_datetime(hh['date']).dt.date, hh[hh_price_col]))
lmp['hh_price'] = lmp['date'].map(hh_map)

missing_before = lmp['hh_price'].isna().sum()
lmp['hh_price'] = lmp['hh_price'].ffill().bfill()
missing_after  = lmp['hh_price'].isna().sum()

print(f"Missing HH rows before fix : {missing_before}")
print(f"Missing HH rows after fix  : {missing_after}")

# Derive IMHR and spark spread
lmp['allin_gas']      = lmp['hh_price'] + 3.0
lmp['ng_power_cost']  = lmp['allin_gas'] * 9.5
lmp['spark_spread']   = lmp['HB_HOUSTON'] - lmp['ng_power_cost']
lmp['imhr']           = np.where(lmp['hh_price'] > 0, lmp['HB_HOUSTON'] / lmp['hh_price'] * 1000, np.nan)

# Restrict to Feb–Apr 2026
lmp = lmp[(lmp['datetime'] >= '2026-02-01') & (lmp['datetime'] <= '2026-04-30')].copy()
print(f"LMP rows (Feb–Apr 2026)    : {len(lmp):,}")


# EXTRACT HELPER COLUMNS

wind['hour']  = wind['datetime'].dt.hour
solar['hour'] = solar['datetime'].dt.hour
lmp['hour']   = lmp['datetime'].dt.hour
lmp['month']  = lmp['datetime'].dt.to_period('M').astype(str)

def regime(h):
    if h <= 6:          return 'Wind Peak (00-06)'
    elif h <= 9:        return 'Transition Morning (07-09)'
    elif h <= 16:       return 'Solar Peak (10-16)'
    elif h <= 18:       return 'Transition Evening (17-18)'
    elif h <= 21:       return 'Demand Peak (19-21)'
    else:               return 'Night Wind (22-23)'

lmp['regime'] = lmp['hour'].apply(regime)


# AGGREGATE BY HOUR OF DAY

# Wind: WGRPP_LZ_WEST = West Load Zone real-time actual wind generation (MW)
wind_by_hour = wind.groupby('hour')['WGRPP_LZ_WEST'].mean().round(1)

# Solar: sum of FarWest + NorthWest + CenterWest PVGRPP regions
# (these are the West Texas solar zones most relevant to HB_WEST price formation)
solar_west_by_hour = {}
for h in range(24):
    s = solar[solar['hour'] == h]
    if len(s) > 0:
        solar_west_by_hour[h] = float(
            s['PVGRPP_FarWest'].mean() +
            s['PVGRPP_NorthWest'].mean() +
            s['PVGRPP_CenterWest'].mean()
        )
    else:
        solar_west_by_hour[h] = 0.0

# LMP: Feb-Apr 2026 West prices
lmp_by_hour = lmp.groupby('hour').agg(
    mean_lmp_west  = ('HB_WEST', 'mean'),
    median_lmp_west= ('HB_WEST', 'median'),
    std_lmp_west   = ('HB_WEST', 'std'),
    pct_neg        = ('HB_WEST', lambda x: (x < 0).mean() * 100),
    n_obs          = ('HB_WEST', 'count'),
    mean_lmp_hou   = ('HB_HOUSTON', 'mean'),
).round(3)

# Build master hourly table
hourly = pd.DataFrame({'hour': range(24)})
hourly['wind_west_mw']         = hourly['hour'].map(wind_by_hour)
hourly['solar_west_mw']        = hourly['hour'].map(solar_west_by_hour)
hourly['total_renewables_mw']  = hourly['wind_west_mw'] + hourly['solar_west_mw']
hourly['mean_lmp_west']        = hourly['hour'].map(lmp_by_hour['mean_lmp_west'])
hourly['median_lmp_west']      = hourly['hour'].map(lmp_by_hour['median_lmp_west'])
hourly['pct_neg']              = hourly['hour'].map(lmp_by_hour['pct_neg'])
hourly['regime']               = hourly['hour'].apply(regime)

print("Hourly aggregates:")
print(hourly[['hour','wind_west_mw','solar_west_mw',
              'total_renewables_mw','mean_lmp_west','pct_neg']].to_string(index=False))


# CORRELATIONS

r_total_lmp, p_total_lmp = stats.pearsonr(hourly['total_renewables_mw'], hourly['mean_lmp_west'])
r_solar_lmp, p_solar_lmp = stats.pearsonr(hourly['solar_west_mw'],       hourly['mean_lmp_west'])
r_wind_lmp,  p_wind_lmp  = stats.pearsonr(hourly['wind_west_mw'],        hourly['mean_lmp_west'])
r_total_neg, p_total_neg = stats.pearsonr(hourly['total_renewables_mw'], hourly['pct_neg'])
r_solar_neg, p_solar_neg = stats.pearsonr(hourly['solar_west_mw'],       hourly['pct_neg'])
r_wind_neg,  p_wind_neg  = stats.pearsonr(hourly['wind_west_mw'],        hourly['pct_neg'])

print(f"Total renewables vs Mean LMP West  :  r = {r_total_lmp:+.4f}  (p={p_total_lmp:.4f})")
print(f"Solar West       vs Mean LMP West  :  r = {r_solar_lmp:+.4f}  (p={p_solar_lmp:.4f})")
print(f"Wind West        vs Mean LMP West  :  r = {r_wind_lmp:+.4f}  (p={p_wind_lmp:.4f})")
print(f"Total renewables vs %% Neg Hours   :  r = {r_total_neg:+.4f}  (p={p_total_neg:.4f})")
print(f"Solar West       vs %% Neg Hours   :  r = {r_solar_neg:+.4f}  (p={p_solar_neg:.4f})")
print(f"Wind West        vs %% Neg Hours   :  r = {r_wind_neg:+.4f}  (p={p_wind_neg:.4f})")

solar_pk  = lmp[lmp['hour'].between(10, 16)]
non_solar = lmp[~lmp['hour'].between(10, 16)]
ratio = (solar_pk['HB_WEST'] < 0).mean() / (non_solar['HB_WEST'] < 0).mean()
print(f"\nSolar peak hours (10-16): {(solar_pk['HB_WEST']<0).mean()*100:.1f}% negative")
print(f"Non-solar hours:          {(non_solar['HB_WEST']<0).mean()*100:.1f}% negative")
print(f"Solar hours are {ratio:.2f}x more likely to be negative")


# REGIME THRESHOLD TABLE

regime_lmp = lmp.groupby('regime').agg(
    hours        = ('HB_WEST', 'count'),
    pct_neg      = ('HB_WEST', lambda x: (x < 0).mean() * 100),
    mean_lmp     = ('HB_WEST', 'mean'),
    median_lmp   = ('HB_WEST', 'median'),
).round(2)

# Add estimated generation from May 2026 baseline
gen_by_regime = {
    'Wind Peak (00-06)':           {'wind_mw': 14000, 'solar_mw': 0,     'total_mw': 14000},
    'Transition Morning (07-09)':  {'wind_mw': 11000, 'solar_mw': 11600, 'total_mw': 22600},
    'Solar Peak (10-16)':          {'wind_mw':  9700, 'solar_mw': 21700, 'total_mw': 31400},
    'Transition Evening (17-18)':  {'wind_mw': 15200, 'solar_mw': 14400, 'total_mw': 29600},
    'Demand Peak (19-21)':         {'wind_mw': 15500, 'solar_mw':  1200, 'total_mw': 16700},
    'Night Wind (22-23)':          {'wind_mw': 16900, 'solar_mw':     0, 'total_mw': 16900},
}

regime_order = ['Wind Peak (00-06)', 'Transition Morning (07-09)', 'Solar Peak (10-16)',
                'Transition Evening (17-18)', 'Demand Peak (19-21)', 'Night Wind (22-23)']

print(f"\n{'Regime':<32} {'Wind MW':>9} {'Solar MW':>9} {'Total MW':>9} "
      f"{'Hours':>7} {'%Neg':>7} {'Mean LMP':>10}")


threshold_rows = []
for r in regime_order:
    g = gen_by_regime[r]
    s = regime_lmp.loc[r]
    print(f"{r:<32} {g['wind_mw']:>9,} {g['solar_mw']:>9,} {g['total_mw']:>9,} "
          f"{int(s['hours']):>7,} {s['pct_neg']:>6.1f}% {s['mean_lmp']:>9.2f}")
    threshold_rows.append({
        'Regime': r,
        'Wind_West_MW_baseline': g['wind_mw'],
        'Solar_West_MW_baseline': g['solar_mw'],
        'Total_Renewables_MW_baseline': g['total_mw'],
        'Hours_in_Feb_Apr': int(s['hours']),
        'Pct_Negative_LMP': round(s['pct_neg'], 2),
        'Mean_LMP_West': round(s['mean_lmp'], 2),
        'Median_LMP_West': round(s['median_lmp'], 2),
        'Data_source_generation': 'ERCOT WGRPP/PVGRPP May 2026 baseline (11 days)',
        'Data_source_LMP': 'ERCOT HB_WEST real-time LMP Feb-Apr 2026 (2135 hours)',
    })

threshold_df = pd.DataFrame(threshold_rows)
threshold_df.to_csv(OUT_CSV_THR, index=False)
print(f"\nThreshold table saved → {OUT_CSV_THR}")

# Save hourly aggregates CSV
hourly.to_csv(OUT_CSV_HLY, index=False)
print(f"Hourly aggregates saved → {OUT_CSV_HLY}")


# SCATTER PLOTS

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor(LGREY)
gs_fig = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

# --- Scatter 1 (main): Total Renewables vs Mean LMP — by hour of day ---------
ax1 = fig.add_subplot(gs_fig[0, :2])
ax1.set_facecolor('white')

c1 = [hour_color(h) for h in hourly['hour']]
ax1.scatter(hourly['total_renewables_mw'], hourly['mean_lmp_west'],
            c=c1, s=200, zorder=5, edgecolors='white', linewidths=1.5)

sl1, ic1, r1, _, _ = stats.linregress(hourly['total_renewables_mw'], hourly['mean_lmp_west'])
xl1 = np.linspace(hourly['total_renewables_mw'].min(),
                   hourly['total_renewables_mw'].max(), 100)
ax1.plot(xl1, sl1*xl1+ic1, '--', color=NAVY, linewidth=2, alpha=0.7,
         label=f'Regression  r = {r1:.3f},  slope = ${sl1*1000:.2f}/GW')

for _, row in hourly.iterrows():
    ax1.annotate(f"{int(row['hour']):02d}",
                 (row['total_renewables_mw'], row['mean_lmp_west']),
                 textcoords='offset points', xytext=(0, 8),
                 fontsize=7.5, ha='center', color=NAVY, fontweight='bold')

ax1.axhline(y=0, color='grey', linewidth=1.2, linestyle='-', alpha=0.6)
ax1.fill_between(
    [hourly['total_renewables_mw'].min()-300, hourly['total_renewables_mw'].max()+300],
    hourly['mean_lmp_west'].min()-2, 0,
    alpha=0.07, color=GREEN, label='Negative LMP zone'
)

from matplotlib.patches import Patch as MPatch
legend_els = [
    MPatch(facecolor=GOLD,   label='Solar Peak (10–16)'),
    MPatch(facecolor=RED,    label='Demand Peak (19–21)'),
    MPatch(facecolor=MID,    label='Wind Peak (00–06)'),
    MPatch(facecolor=ORANGE, label='Transition hours'),
    plt.Line2D([0],[0], color=NAVY, linestyle='--',
               label=f'Regression  r = {r1:.3f}'),
    MPatch(facecolor=GREEN, alpha=0.3, label='Negative LMP zone'),
]
ax1.legend(handles=legend_els, fontsize=9, loc='upper right', framealpha=0.9)
ax1.set_title(
    'MAIN SCATTER — Total West Renewables (MW) vs Mean LMP West ($/MWh)\n'
    'Each point = one hour of day (labeled 00–23 CST) | '
    'Generation: May 2026 | LMP: Feb–Apr 2026',
    fontsize=11, color=NAVY, fontweight='bold')
ax1.set_xlabel('Total West Renewable Generation (MW)  [Wind LZ West + Solar FarWest + NW + CW]',
               fontsize=10, color=NAVY)
ax1.set_ylabel('Mean LMP West ($/MWh)', fontsize=10, color=NAVY)
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v/1000:.0f} GW'))
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:.0f}'))
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.text(0.02, 0.97,
         f'r = {r1:.3f}   R² = {r1**2:.3f}\n'
         f'Slope: ${sl1*1000:.2f} per GW of renewables\n'
         f'Intercept: ${ic1:.1f}/MWh',
         transform=ax1.transAxes, fontsize=9.5, verticalalignment='top', color=NAVY,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85,
                   edgecolor='#CCCCCC'))

# --- Scatter 2: Solar MW vs % Negative hours ----------------------------------
ax2 = fig.add_subplot(gs_fig[0, 2])
ax2.set_facecolor('white')

c2 = [GOLD if v > 1000 else MID for v in hourly['solar_west_mw']]
ax2.scatter(hourly['solar_west_mw'], hourly['pct_neg'],
            c=c2, s=150, zorder=5, edgecolors='white', linewidths=1.2)

sl2, ic2, r2, _, _ = stats.linregress(hourly['solar_west_mw'], hourly['pct_neg'])
xl2 = np.linspace(0, hourly['solar_west_mw'].max(), 100)
ax2.plot(xl2, sl2*xl2+ic2, '--', color=NAVY, linewidth=1.8, alpha=0.7)

for _, row in hourly.iterrows():
    if row['solar_west_mw'] > 500:
        ax2.annotate(f"{int(row['hour']):02d}",
                     (row['solar_west_mw'], row['pct_neg']),
                     textcoords='offset points', xytext=(0, 6),
                     fontsize=7.5, ha='center', color=NAVY, fontweight='bold')

ax2.set_title('Solar Generation West\nvs % Hours Negative LMP', fontsize=10,
              color=NAVY, fontweight='bold')
ax2.set_xlabel('Solar West MW\n(FarWest + NorthWest + CenterWest)', fontsize=9, color=NAVY)
ax2.set_ylabel('% Hours with Negative LMP', fontsize=9, color=NAVY)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v/1000:.1f} GW'))
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.text(0.05, 0.95, f'r = {r2:.3f}', transform=ax2.transAxes,
         fontsize=10, color=NAVY, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

# --- Scatter 3: Raw LMP observations vs generation proxy ----------------------
ax3 = fig.add_subplot(gs_fig[1, :2])
ax3.set_facecolor('white')

lmp['total_proxy'] = lmp['hour'].map(
    dict(zip(hourly['hour'], hourly['total_renewables_mw']))
)
sample = lmp.iloc[::2].copy()   # every other row for visual clarity
s_colors = [hour_color(h) for h in sample['hour']]
ax3.scatter(sample['total_proxy'], sample['HB_WEST'],
            c=s_colors, s=12, alpha=0.4, zorder=3)

ax3.axhline(y=0, color='black', linewidth=1.2, alpha=0.6, label='LMP = $0')
ax3.set_ylim(-40, 125)

sl3, ic3, r3, _, _ = stats.linregress(lmp['total_proxy'], lmp['HB_WEST'])
xl3 = np.linspace(lmp['total_proxy'].min(), lmp['total_proxy'].max(), 100)
ax3.plot(xl3, sl3*xl3+ic3, '-', color=NAVY, linewidth=2.5, alpha=0.85, zorder=5,
         label=f'OLS regression  r = {r3:.3f}')

from matplotlib.patches import Patch as MPatch2
legend2 = [
    MPatch2(facecolor=GOLD,   alpha=0.7, label='Solar Peak (10–16)'),
    MPatch2(facecolor=RED,    alpha=0.7, label='Demand Peak (19–21)'),
    MPatch2(facecolor=MID,    alpha=0.7, label='Wind Peak (00–06)'),
    MPatch2(facecolor=ORANGE, alpha=0.7, label='Transition'),
    plt.Line2D([0],[0], color=NAVY, linewidth=2.5,
               label=f'OLS regression  r = {r3:.3f}'),
]
ax3.legend(handles=legend2, fontsize=8.5, loc='upper right', framealpha=0.9)
ax3.set_title(
    f'RAW SCATTER — All {len(lmp):,} Hourly LMP Observations vs Estimated Renewable Generation\n'
    'Each point = one actual HB_WEST LMP observation | '
    'Generation = mean for that hour (proxy)',
    fontsize=10, color=NAVY, fontweight='bold')
ax3.set_xlabel('Estimated Renewable Generation at that Hour (MW)', fontsize=9, color=NAVY)
ax3.set_ylabel('Actual HB_WEST LMP ($/MWh)', fontsize=9, color=NAVY)
ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v/1000:.0f} GW'))
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:.0f}'))
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.text(0.02, 0.97, f'r = {r3:.3f}   n = {len(lmp):,}',
         transform=ax3.transAxes, fontsize=9, verticalalignment='top', color=NAVY,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

# --- Scatter 4: Wind MW vs % Negative hours -----------------------------------
ax4 = fig.add_subplot(gs_fig[1, 2])
ax4.set_facecolor('white')

c4 = [hour_color(h) for h in hourly['hour']]
ax4.scatter(hourly['wind_west_mw'], hourly['pct_neg'],
            c=c4, s=150, zorder=5, edgecolors='white', linewidths=1.2)

sl4, ic4, r4, _, _ = stats.linregress(hourly['wind_west_mw'], hourly['pct_neg'])
xl4 = np.linspace(hourly['wind_west_mw'].min(), hourly['wind_west_mw'].max(), 100)
ax4.plot(xl4, sl4*xl4+ic4, '--', color=NAVY, linewidth=1.8, alpha=0.7)

for _, row in hourly.iterrows():
    if int(row['hour']) <= 2 or 10 <= int(row['hour']) <= 13:
        ax4.annotate(f"{int(row['hour']):02d}",
                     (row['wind_west_mw'], row['pct_neg']),
                     textcoords='offset points', xytext=(0, 6),
                     fontsize=7.5, ha='center', color=NAVY, fontweight='bold')

ax4.set_title('Wind Generation West\nvs % Hours Negative LMP', fontsize=10,
              color=NAVY, fontweight='bold')
ax4.set_xlabel('Wind LZ West MW\n(WGRPP_LZ_WEST real-time actual)', fontsize=9, color=NAVY)
ax4.set_ylabel('% Hours with Negative LMP', fontsize=9, color=NAVY)
ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v/1000:.0f} GW'))
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.text(0.05, 0.95, f'r = {r4:.3f}', transform=ax4.transAxes,
         fontsize=10, color=NAVY, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

fig.suptitle(
    'ERCOT West Zone: Renewable Generation Intermittency & Negative LMP — Scatter Analysis\n'
    f'r(total renewables, mean LMP) = {r1:.3f}  |  '
    f'r(solar, %% negative) = {r2:.3f}  |  '
    f'r(total, %% negative) = {r_total_neg:.3f}',
    fontsize=12, fontweight='bold', color=NAVY, y=0.99)

plt.savefig(OUT_SCATTER, dpi=150, bbox_inches='tight', facecolor=LGREY)

plt.close()


# 8. GENERATION + NEGATIVE PRICING PATTERN CHARTS

fig2 = plt.figure(figsize=(16, 14))
fig2.patch.set_facecolor(LGREY)
gs2 = gridspec.GridSpec(3, 2, figure=fig2, hspace=0.45, wspace=0.35)
hours_list = list(range(24))

# Chart 1: Stacked bar — wind + solar by hour
ax_a = fig2.add_subplot(gs2[0, :])
ax_a.set_facecolor('white')
w_vals = [hourly.loc[hourly['hour']==h, 'wind_west_mw'].values[0] for h in hours_list]
s_vals = [hourly.loc[hourly['hour']==h, 'solar_west_mw'].values[0] for h in hours_list]
ax_a.bar(hours_list, w_vals, label='Wind West (MW)', color=MID, alpha=0.85, width=0.6)
ax_a.bar(hours_list, s_vals, bottom=w_vals, label='Solar West approx (MW)',
          color=GOLD, alpha=0.85, width=0.6)
ax_a.axvspan(9.5, 16.5, alpha=0.07, color=GOLD, label='Solar peak window (10–16)')
ax_a.set_title('ERCOT West Zone: Wind + Solar Generation by Hour (May 2026 baseline)',
               fontsize=12, color=NAVY, fontweight='bold')
ax_a.set_xlabel('Hour of Day (CST)', fontsize=10, color=NAVY)
ax_a.set_ylabel('Generation (MW)', fontsize=10, color=NAVY)
ax_a.set_xticks(hours_list)
ax_a.set_xticklabels([f'{h:02d}' for h in hours_list], fontsize=8)
ax_a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v/1000:.0f} GW'))
ax_a.legend(fontsize=9, loc='upper right')
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)

# Chart 2: % Negative by hour
ax_b = fig2.add_subplot(gs2[1, 0])
ax_b.set_facecolor('white')
neg_vals = [hourly.loc[hourly['hour']==h, 'pct_neg'].values[0] for h in hours_list]
bar_colors = [GREEN if p > 40 else (GOLD if p > 20 else '#CCCCCC') for p in neg_vals]
ax_b.bar(hours_list, neg_vals, color=bar_colors, width=0.6, alpha=0.9)
ax_b.axhline(y=28.4, color=RED, linestyle='--', linewidth=1.5, label='Overall avg (28.4%)')
ax_b.axvspan(9.5, 16.5, alpha=0.07, color=GOLD)
ax_b.set_title('HB_WEST: % Hours with Negative LMP\n(Feb–Apr 2026)', fontsize=10,
               color=NAVY, fontweight='bold')
ax_b.set_xlabel('Hour of Day (CST)', fontsize=9, color=NAVY)
ax_b.set_ylabel('% Hours Negative', fontsize=9, color=NAVY)
ax_b.set_xticks(range(0, 24, 3))
ax_b.set_xticklabels([f'{h:02d}' for h in range(0, 24, 3)], fontsize=8)
ax_b.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
ax_b.legend(fontsize=8); ax_b.set_ylim(0, 65)
ax_b.annotate('Hour 11: 50%\nSolar peak', xy=(11, 50), xytext=(14, 58),
              arrowprops=dict(arrowstyle='->', color=NAVY, lw=1.2),
              fontsize=8, color=NAVY, fontweight='bold')
ax_b.spines['top'].set_visible(False); ax_b.spines['right'].set_visible(False)

# Chart 3: Mean LMP by hour
ax_c = fig2.add_subplot(gs2[1, 1])
ax_c.set_facecolor('white')
lmp_vals = [hourly.loc[hourly['hour']==h, 'mean_lmp_west'].values[0] for h in hours_list]
lmp_bar_colors = [GREEN if v < 0 else (RED if v > 40 else MID) for v in lmp_vals]
ax_c.bar(hours_list, lmp_vals, color=lmp_bar_colors, width=0.6, alpha=0.9)
ax_c.axhline(y=0, color='black', linewidth=0.8)
ax_c.axvspan(9.5, 16.5, alpha=0.07, color=GOLD)
ax_c.set_title('HB_WEST: Mean LMP by Hour\n(Feb–Apr 2026)', fontsize=10,
               color=NAVY, fontweight='bold')
ax_c.set_xlabel('Hour of Day (CST)', fontsize=9, color=NAVY)
ax_c.set_ylabel('Mean LMP ($/MWh)', fontsize=9, color=NAVY)
ax_c.set_xticks(range(0, 24, 3))
ax_c.set_xticklabels([f'{h:02d}' for h in range(0, 24, 3)], fontsize=8)
ax_c.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:.0f}'))
ax_c.spines['top'].set_visible(False); ax_c.spines['right'].set_visible(False)

# Chart 4: Regime comparison
ax_d = fig2.add_subplot(gs2[2, 0])
ax_d.set_facecolor('white')
rlabels  = ['Wind\n(00-06)', 'Trans\nMorn\n(07-09)', 'Solar\nPeak\n(10-16)',
            'Trans\nEve\n(17-18)', 'Demand\nPeak\n(19-21)', 'Night\nWind\n(22-23)']
rneg     = [18.3, 25.1, 49.4, 30.9, 10.1, 20.2]
rcolors  = [MID, GOLD, GREEN, ORANGE, RED, PURPLE]
rtotals  = [14000, 22600, 31400, 29600, 16700, 16900]
x_r = np.arange(len(rlabels))
bars_r = ax_d.bar(x_r, rneg, color=rcolors, alpha=0.85, width=0.6)
ax_d.axhline(y=28.4, color='grey', linestyle='--', linewidth=1, label='Overall avg (28.4%)')
for bar, tot, p in zip(bars_r, rtotals, rneg):
    ax_d.text(bar.get_x()+bar.get_width()/2, p+0.8,
              f'{tot/1000:.0f}GW', ha='center', va='bottom', fontsize=7.5, color=NAVY)
ax_d.set_title('Neg Price Freq by Generation Regime\n(GW labels = total renewable MW)',
               fontsize=10, color=NAVY, fontweight='bold')
ax_d.set_xticks(x_r); ax_d.set_xticklabels(rlabels, fontsize=8)
ax_d.set_ylabel('% Hours Negative LMP', fontsize=9, color=NAVY)
ax_d.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
ax_d.legend(fontsize=8); ax_d.set_ylim(0, 62)
ax_d.spines['top'].set_visible(False); ax_d.spines['right'].set_visible(False)

# Chart 5: Monthly trend
ax_e = fig2.add_subplot(gs2[2, 1])
ax_e.set_facecolor('white')
months_l   = ['Feb-26', 'Mar-26', 'Apr-26']
mneg_pcts  = [21.43, 34.99, 28.19]
mavg_hh    = [3.61, 3.05, 2.75]
ax_e_twin  = ax_e.twinx()
ax_e.bar(months_l, mneg_pcts, color=[MID, GREEN, GOLD], alpha=0.75, width=0.4)
ax_e_twin.plot(months_l, mavg_hh, 'o-', color=RED, linewidth=2, markersize=8)
ax_e.set_title('Monthly: Neg Price Freq\nvs Henry Hub Price', fontsize=10,
               color=NAVY, fontweight='bold')
ax_e.set_ylabel('% Hours Negative LMP', fontsize=9, color=NAVY)
ax_e_twin.set_ylabel('Avg HH ($/MMBtu)', fontsize=9, color=RED)
ax_e.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
ax_e_twin.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:.2f}'))
ax_e.set_ylim(0, 45); ax_e_twin.set_ylim(2.0, 4.5)
ax_e.annotate('Peak curtailment\nMarch', xy=(1, 34.99), xytext=(1.45, 40),
              arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.2),
              fontsize=8, color=GREEN, fontweight='bold')
ax_e.spines['top'].set_visible(False)

fig2.suptitle(
    'ERCOT West Zone: Generation Profiles & Negative Pricing Patterns\n'
    'Generation (May 2026 baseline) | LMP patterns (Feb–Apr 2026 actual)',
    fontsize=12, fontweight='bold', color=NAVY, y=0.99)

plt.savefig(OUT_CHARTS, dpi=150, bbox_inches='tight', facecolor=LGREY)
plt.close()
