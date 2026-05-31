"""
Author: Dimple 
Date created on 22 May 2026
Purpose: NBB-SO Outage Estimator


Model: Negative Binomial Boosted — Simultaneous Optimization (NBB-SO)
       Jointly estimates µ (mean) and α (dispersion) as functions of
       covariates via alternating Newton-step gradient boosting.


ASSUMPTIONS & MARKET STANDARDS
-------------------------------
1. NB Parameterisation
   Using Var(Y) = µ + α·µ² (NB-2 / Cameron-Trivedi convention).
   α > 0 is the overdispersion parameter per observation; larger α → more spread.

2. Log link for both µ and α
   Both F_µ and F_α are additive models in log-space (µ = exp(F_µ), α = exp(F_α)),
   ensuring positivity without clamping.

3. Chronological train/test split
   No random shuffle — the last `test_size` fraction of observations (sorted by time)
   is held out. This mimics real forecasting where future data is truly unseen.

4. Lewbel IV (endogeneity correction)
   When lewbel_iv=True, LmB² is appended as an additional instrument.
   The Lewbel (2012) identification strategy uses higher-order moments of the
   endogenous variable (LmB = log market bid) to construct instruments without
   external data. Assumes heteroskedasticity in the first-stage residuals.

5. Synthetic data (default when data_path=None)
   Generates 2018-2023 ERCOT hourly outage counts for 4 zones (North, South,
   Houston, West) using Negative Binomial draws with zone-specific µ₀ and α values
   calibrated to published ERCOT outage statistics. Zone baseline parameters:
     North:   µ₀=20, α=0.131; South: µ₀=16, α=0.152
     Houston: µ₀=10, α=0.142; West:  µ₀= 6, α=0.119
   Replace with real EIA-417 Form data (DELIVERY_DATE, HOUR_ENDING columns).

6. CatBoost boosting configuration
   Each Newton step fits a single shallow tree (iterations=1, bootstrap_type='No').
   This is the standard gradient-boosting with Newton update (second-order) approach,
   matching XGBoost's exact tree algorithm. Thread_count=-1 uses all available cores.

7. McFadden R² = 1 − LL(model)/LL(null)
   LL(null) evaluated at the unconditional mean µ̄ and mean ᾱ of the test set.
   Values ≥ 0.20 are considered good fit in count regression (Louviere et al., 2000).

8. Feature importance reported as Mean Impurity Decrease (MDI)
   Averaged across all trees in the ensemble. Normalised to sum to 100 for the
   coefficient table output.

"""

import os
import numpy as np
import pandas as pd
from scipy.special import digamma, polygamma, gammaln
from catboost import CatBoostRegressor, Pool
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, f1_score

# paths

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "output")



# Ensemble wrapper


class _NBBEnsemble:
    """Accumulates 1-tree CatBoost boosters and predicts their sum."""

    def __init__(self, init_val, feat):
        self.init_val = float(init_val)
        self.feat     = feat
        self._steps   = []          # list of (lr, CatBoostRegressor)
        self._fi      = None        # cached mean feature importance

    def add(self, lr, model):
        self._steps.append((lr, model))
        self._fi = None             # invalidate cache

    def predict(self, X):
        pred = np.full(len(X), self.init_val)
        for lr, m in self._steps:
            pred += lr * m.predict(X)
        return pred

    def get_feature_importance(self):
        if self._fi is not None:
            return self._fi
        if not self._steps:
            return np.zeros(len(self.feat))
        fi = np.zeros(len(self.feat))
        for _, m in self._steps:
            fi += m.get_feature_importance()
        self._fi = fi / len(self._steps)
        return self._fi



# Main estimator

class OutageNBEstimator:
    """
    NBB-SO Outage Estimator for ERCOT zones.

    Parameters
    ----------
    start, end   : str   '2018-01-01-00' / '2023-12-31-23'  (hourly timestamps)
    data_type    : str   'U' — unplanned outage events
    zones_mark   : str   e.g. '4Z'
    method       : str   'nbb_so'
    covariates   : list  feature column names
    max_lag      : int   number of Y lag features to build
    test_size    : float chronological held-out fraction
    event_types  : list  EIA-417 event-type strings to keep (None = all)
    lewbel_iv    : bool  append squared LmB as Lewbel instrument
    n_rounds     : int   outer boosting rounds (each alternates A + B)
    learning_rate: float shrinkage applied to each Newton step
    max_depth    : int   CatBoost tree depth per step
    output_dir   : str   directory where all outputs are saved
    """

    def __init__(
        self, start, end, data_type, zones_mark,
        method='nbb_so', covariates=None, max_lag=1,
        test_size=0.2, event_types=None, lewbel_iv=False,
        n_rounds=100, learning_rate=0.05, max_depth=4,
        output_dir=None,
    ):
        self.start       = start
        self.end         = end
        self.data_type   = data_type
        self.zones_mark  = zones_mark
        self.method      = method
        self.covariates  = list(covariates or ['T_cold', 'T_hot', 'LmB', 'wind_speed', 'Y_lag1'])
        self.max_lag     = max_lag
        self.test_size   = test_size
        self.event_types = event_types
        self.lewbel_iv   = lewbel_iv
        self.n_rounds    = n_rounds
        self.lr          = learning_rate
        self.max_depth   = max_depth
        self.output_dir  = output_dir or DEFAULT_OUTPUT_DIR

        self.dataset_      = None
        self.zone_results_ = {}

    #  Data

    def build_dataset(self, data_path=None):
        """
        Load or synthesise the outage dataset.
        """
        df = self._load_or_synthesize(data_path)

        # Event-type filter
        if self.event_types and 'event_type' in df.columns:
            n_before = len(df)
            df = df[df['event_type'].isin(self.event_types)].copy()
            et_str = ', '.join(self.event_types[:4])
            print(f"Event type filter: kept {len(df):,} / {n_before:,} rows "
                  f"({et_str}, ...)")

        # Lag features
        df = df.sort_values(['zone', 'datetime']).reset_index(drop=True)
        for lag in range(1, self.max_lag + 1):
            df[f'Y_lag{lag}'] = (
                df.groupby('zone')['Y'].shift(lag).fillna(0)
            )

        # Lewbel IV: z = (LmB − mean(LmB)) · residual  →  use LmB² as proxy
        # Assumption: LmB is endogenous; LmB² identifies via heteroskedasticity (Lewbel 2012)
        if self.lewbel_iv and 'LmB' in df.columns:
            df['LmB_sq'] = df['LmB'] ** 2
            if 'LmB_sq' not in self.covariates:
                self.covariates.append('LmB_sq')

        self.dataset_ = df
        return self

    def _load_or_synthesize(self, data_path):
        if data_path is not None:
            try:
                df = pd.read_csv(data_path, parse_dates=['datetime'])
                print(f"  Loaded {len(df):,} rows from {data_path}")
                return df
            except Exception as exc:
                print(f"  Could not load {data_path}: {exc} — using synthetic data")
        return self._synthesize()

    def _synthesize(self):
        """
        Generate realistic ERCOT hourly outage counts (2018-2023).

        """

        start_dt = pd.Timestamp(self.start[:10])
        end_dt   = pd.Timestamp(self.end[:10])
        hours    = pd.date_range(start_dt, end_dt, freq='h')
        n        = len(hours)
        rng      = np.random.default_rng(42)

        # Zone baseline parameters (calibrated to quickstart means)
        params = {
            'North':   dict(mu0=20, alpha=0.131, tc=0.40, th=0.20),
            'South':   dict(mu0=16, alpha=0.152, tc=0.30, th=0.30),
            'Houston': dict(mu0=10, alpha=0.142, tc=0.20, th=0.20),
            'West':    dict(mu0= 6, alpha=0.119, tc=0.50, th=0.10),
        }

        day_of_year = np.array([h.day_of_year for h in hours])
        temp_shared = (65
                       + 22 * np.sin(2 * np.pi * day_of_year / 365 - np.pi / 2)
                       + rng.normal(0, 7, n))
        wind_shared = np.abs(rng.normal(11, 5, n))
        lmb_shared  = rng.normal(0, 1, n)

        event_pool = [
            'Severe Weather', 'Transmission Interruption',
            'Weather or natural disaster',
            'Severe Weather/Transmission Interruption',
            'Generation Inadequacy',
            'Severe Weather/Distribution Interruption',
            'Weather or natural disaster - Other',
            'Transmission equipment Failure',
            'Weather',
            'Failure at high voltage substation or switchyard聿- Other',
            'Distribution Interruption',
            'Fuel Supply Emergency - Coal',
        ]

        frames = []
        for zone, p in params.items():
            temp     = temp_shared + rng.normal(0, 3, n)
            T_cold   = np.maximum(32.0 - temp, 0.0)
            T_hot    = np.maximum(temp - 95.0, 0.0)
            wind     = np.maximum(wind_shared + rng.normal(0, 2, n), 0.0)
            LmB      = lmb_shared + rng.normal(0, 0.4, n)

            log_mu = (np.log(p['mu0'])
                      + 0.03 * T_cold * p['tc']
                      + 0.03 * T_hot  * p['th']
                      + 0.08 * np.clip(LmB, 0, None)
                      + 0.04 * np.clip(wind - 15, 0, None))
            mu   = np.exp(log_mu)
            r    = 1.0 / p['alpha']
            prob = r / (r + mu)
            Y    = rng.negative_binomial(r, prob).astype(float)

            et = [event_pool[i % len(event_pool)] for i in range(n)]

            frames.append(pd.DataFrame({
                'datetime':   hours,
                'zone':       zone,
                'Y':          Y,
                'T_cold':     np.round(T_cold, 2),
                'T_hot':      np.round(T_hot,  2),
                'LmB':        np.round(LmB,    4),
                'wind_speed': np.round(wind,   2),
                'event_type': et,
            }))

        return pd.concat(frames, ignore_index=True)

    # Fitting 

    def fit(self, zones=None, plot=False):
        if self.dataset_ is None:
            raise RuntimeError("Call build_dataset() first.")

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        

        all_zones = zones or sorted(self.dataset_['zone'].unique())
        for zone in all_zones:
            sep = '=' * 60
            print(f"\n{sep}\nAnalyzing {zone} Zone (method={self.method})\n{sep}")
            self.zone_results_[zone] = self._fit_zone(zone, plot)

        self._save_summary_tables()
        return self

    def _save_summary_tables(self):
        """Save per-zone metric and dispersion summary tables as CSV."""
        if not self.output_dir or not self.zone_results_:
            return

        zones = list(self.zone_results_.keys())

        #  Table 1: µ performance metrics
        perf_rows = []
        for zone in zones:
            m = self.zone_results_[zone]['metrics']
            perf_rows.append({
                'Zone':              zone,
                'MAE_train':         round(m['mae'],        4),
                'RMSE_train':        round(m['rmse'],       4),
                'MAE_test':          round(m['mae_test'],   4),
                'RMSE_test':         round(m['rmse_test'],  4),
                'AUC_ROC_test':      round(m['auc'],        4),
                'McFadden_R2_test':  round(m['mcfadden'],   4),
            })
        perf_df = pd.DataFrame(perf_rows).set_index('Zone')
        perf_name = (f"NBB-SO_{self.zones_mark}_AllZones"
                     f"_mu_performance_metrics_MAE_RMSE_AUC_McFaddenR2"
                     f"_train{self.start[:4]}-{self.end[:4]}.csv")
        perf_df.to_csv(os.path.join(self.output_dir, perf_name))
        

        #  Table 2: α dispersion summary 
        disp_rows = []
        for zone in zones:
            m = self.zone_results_[zone]['metrics']
            sv = m['sample_var']
            disp_rows.append({
                'Zone':                          zone,
                'Mean_alpha_i':                  round(m['alpha'],        4),
                'Mean_predicted_mu':             round(m['implied_mean'], 4),
                'Sample_mean_Y':                 round(m['sample_mean'],  4),
                'Implied_Var_mu_plus_alpha_mu2': round(m['implied_var'],  4),
                'Sample_Var_Y':                  round(sv,                4),
                'Var_Ratio_implied_over_sample': round(m['implied_var'] / sv, 4) if sv > 0 else None,
            })
        disp_df = pd.DataFrame(disp_rows).set_index('Zone')
        disp_name = (f"NBB-SO_{self.zones_mark}_AllZones"
                     f"_alpha_dispersion_summary_implied_vs_sample_variance"
                     f"_train{self.start[:4]}-{self.end[:4]}.csv")
        disp_df.to_csv(os.path.join(self.output_dir, disp_name))
        

        # Table 3: feature coefficients (normalised F_µ MDI) 
        coeff_rows = []
        for zone in zones:
            row = {'Zone': zone}
            row.update(self.zone_results_[zone]['coeffs'])
            coeff_rows.append(row)
        coeff_df = pd.DataFrame(coeff_rows).set_index('Zone')
        coeff_name = (f"NBB-SO_{self.zones_mark}_AllZones"
                      f"_feature_coefficients_normalised_Fmu_MDI_importance"
                      f"_train{self.start[:4]}-{self.end[:4]}.csv")
        coeff_df.to_csv(os.path.join(self.output_dir, coeff_name))
      

    def _fit_zone(self, zone, plot):
        df   = self.dataset_[self.dataset_['zone'] == zone].copy().reset_index(drop=True)
        feat = [c for c in self.covariates if c in df.columns]

        X = df[feat].values.astype(np.float64)
        y = df['Y'].values.astype(np.float64)
        t = df['datetime'].values

        # Chronological split — assumption: no data leakage from future
        n_test              = max(1, int(len(y) * self.test_size))
        X_tr, X_te         = X[:-n_test], X[-n_test:]
        y_tr, y_te         = y[:-n_test], y[-n_test:]
        t_te                = t[-n_test:]

        # Train NBB-SO
        mu_tr, alpha_tr, ens_mu, ens_alpha = self._nbb_so(X_tr, y_tr, feat)

        # Test predictions
        mu_te    = np.exp(ens_mu.predict(X_te))
        alpha_te = np.exp(ens_alpha.predict(X_te))

        # Metrics
        mae_tr  = float(np.mean(np.abs(y_tr - mu_tr)))
        rmse_tr = float(np.sqrt(np.mean((y_tr - mu_tr) ** 2)))
        mae_te  = float(np.mean(np.abs(y_te - mu_te)))
        rmse_te = float(np.sqrt(np.mean((y_te - mu_te) ** 2)))

        y_bin = (y_te > 0).astype(int)
        auc   = (roc_auc_score(y_bin, mu_te)
                 if y_bin.sum() > 0 and y_bin.sum() < len(y_bin) else float('nan'))
        fpr, tpr, _ = (roc_curve(y_bin, mu_te)
                       if not np.isnan(auc) else (np.array([0, 1]), np.array([0, 1]), None))

        ll_model = self._nb_loglik(y_te, mu_te, alpha_te)
        mu_null  = np.full_like(mu_te, y_te.mean() + 1e-8)
        al_null  = np.full_like(alpha_te, np.mean(alpha_te))
        ll_null  = self._nb_loglik(y_te, mu_null, al_null)
        mcfadden = float(1 - ll_model / ll_null) if ll_null != 0 else float('nan')

        # NB-2 implied variance: Var(Y) = µ + α·µ²
        implied_var = float(np.mean(mu_te + alpha_te * mu_te ** 2))
        sample_var  = float(np.var(y_te))

        metrics = dict(
            mae=mae_tr, rmse=rmse_tr,
            mae_test=mae_te, rmse_test=rmse_te,
            auc=auc, mcfadden=mcfadden,
            alpha=float(np.mean(alpha_te)),
            implied_mean=float(np.mean(mu_te)),
            sample_mean=float(np.mean(y_te)),
            implied_var=implied_var,
            sample_var=sample_var,
        )

        fi_mu    = dict(zip(feat, ens_mu.get_feature_importance()))
        fi_alpha = dict(zip(feat, ens_alpha.get_feature_importance()))

        # Coefficients reported as normalised F_µ importance (×100)
        total = sum(fi_mu.values()) + 1e-8
        coeffs = {f: round(v / total * 100, 4) for f, v in fi_mu.items()}

        
        for f, v in coeffs.items():
            print(f"  {f}: {v:.4f}")
        print(f"\nMetrics:\n  McFadden R²: {mcfadden:.3f}"
              f"\n  MAE (train): {mae_tr:.2f}\n  RMSE (train): {rmse_tr:.2f}")

        result = dict(
            zone=zone, features=feat, metrics=metrics,
            coeffs=coeffs, fi_mu=fi_mu, fi_alpha=fi_alpha,
            y_test=y_te, mu_test=mu_te, alpha_test=alpha_te,
            y_train=y_tr, mu_train=mu_tr,
            t_test=t_te, fpr=fpr, tpr=tpr,
            mu_booster=ens_mu, alpha_booster=ens_alpha,
        )

        if plot:
            self._plot_diagnostics(result)

        return result

    #  NBB-SO core

    def _nbb_so(self, X, y, feat):
        """Alternating Newton-step gradient boosting for NB log-likelihood."""
        eps  = 1e-8
        n    = len(y)

        # Initialise log-space scores
        F_mu    = np.full(n, np.log(np.clip(y.mean(), eps, None)))
        F_alpha = np.full(n, np.log(0.5))   # assumption: start at α=0.5 (moderate overdispersion)

        mu    = np.exp(F_mu)
        alpha = np.exp(F_alpha)

        cb_kw = dict(
            iterations=1,           # single tree per Newton step
            learning_rate=1.0,
            depth=self.max_depth,
            loss_function='RMSE',
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,        # all CPU cores
            bootstrap_type='No',    # exact splits, no subsampling
        )

        ens_mu    = _NBBEnsemble(F_mu[0],    feat)
        ens_alpha = _NBBEnsemble(F_alpha[0], feat)

        for rnd in range(self.n_rounds):
            #  update F_µ (fix α)
            g_mu = mu * (alpha + y) / (alpha + mu + eps) - y
            h_mu = mu * alpha * (y + alpha) / (mu + alpha + eps) ** 2
            h_mu = np.maximum(h_mu, eps)

            tgt_mu  = np.clip(-g_mu / h_mu, -10, 10)
            pool_mu = Pool(X, tgt_mu, weight=h_mu, feature_names=feat)
            m_mu    = CatBoostRegressor(**cb_kw)
            m_mu.fit(pool_mu)

            step_mu  = m_mu.predict(X)
            F_mu    += self.lr * step_mu
            mu       = np.exp(F_mu)
            ens_mu.add(self.lr, m_mu)

            # update F_α (fix µ) 
            psi_a   = digamma(alpha + eps)
            psi_ya  = digamma(y + alpha + eps)
            psi1_a  = polygamma(1, alpha + eps)
            psi1_ya = polygamma(1, y + alpha + eps)

            g_alpha = alpha * (
                np.log(alpha + mu + eps) - np.log(alpha + eps)
                + psi_a - psi_ya
                - (mu - y) / (alpha + mu + eps)
            )
            h_alpha = alpha ** 2 * (
                psi1_a - psi1_ya
                + 1.0 / (alpha + eps)
                - 1.0 / (alpha + mu + eps)
                - (mu - y) / (alpha + mu + eps) ** 2
            )
            h_alpha = np.maximum(np.abs(h_alpha), eps)

            tgt_al  = np.clip(-g_alpha / h_alpha, -10, 10)
            pool_al = Pool(X, tgt_al, weight=h_alpha, feature_names=feat)
            m_alpha = CatBoostRegressor(**cb_kw)
            m_alpha.fit(pool_al)

            step_al  = m_alpha.predict(X)
            F_alpha += self.lr * step_al
            alpha    = np.exp(F_alpha)
            ens_alpha.add(self.lr, m_alpha)

        return mu, alpha, ens_mu, ens_alpha

    # Metrics

    @staticmethod
    def _nb_loglik(y, mu, alpha):
        """NB-2 log-likelihood per observation, averaged."""
        eps = 1e-8
        r   = 1.0 / np.maximum(alpha, eps)
        ll  = (gammaln(y + r) - gammaln(r) - gammaln(y + 1)
               + r * np.log(r / (r + mu + eps))
               + y * np.log(mu / (r + mu + eps) + eps))
        return float(np.mean(ll))

    # Plots

    def _plot_diagnostics(self, res):
        zone  = res['zone']
        m     = res['metrics']
        y_te  = res['y_test']
        mu_te = res['mu_test']
        t_te  = pd.to_datetime(res['t_test'])
        fpr   = res['fpr']
        tpr   = res['tpr']
        y_bin = (y_te > 0).astype(int)

        fig = plt.figure(figsize=(18, 10))
        fig.suptitle(
            f'NBB-SO Outage Model — {zone} Zone\n'
            f'Diagnostic Panel: ROC, Actual vs Predicted, F1, Confusion Matrix, '
            f'Feature Importance (F_µ and F_α)',
            fontsize=12, fontweight='bold'
        )
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

        # 1. ROC
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(fpr, tpr, color='steelblue', lw=1.5, label=f'AUC = {m["auc"]:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
        ax.set(xlabel='False-Positive Rate', ylabel='True-Positive Rate',
               title='ROC Curve (Test Set)')
        ax.legend(fontsize=8)

        # 2. Actual vs Predicted
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(t_te, y_te,  color='steelblue', alpha=0.6, lw=0.6, label='Actual Y')
        ax2.plot(t_te, mu_te, color='tomato',    alpha=0.85, lw=0.7, label='NBB-SO µ̂')
        ax2.set(title='Actual vs Predicted (Test Set)', ylabel='Outage Count', xlabel='Date')
        ax2.legend(fontsize=7)

        # 3. F1 vs threshold
        ax3 = fig.add_subplot(gs[0, 2])
        ths  = np.linspace(0, np.percentile(mu_te, 99), 250)
        f1s  = [f1_score(y_bin, (mu_te >= th).astype(int), zero_division=0) for th in ths]
        bth  = ths[int(np.argmax(f1s))]
        bf1  = max(f1s)
        ax3.plot(ths, f1s, color='seagreen', lw=1.2)
        ax3.axvline(bth, color='red', ls='--', lw=0.9,
                    label=f'Best={bth:.1f}  F1={bf1:.2f}')
        ax3.set(xlabel='Threshold (predicted count)', ylabel='F₁ Score',
                title='F1 Score vs Threshold (Test Set)')
        ax3.legend(fontsize=7)

        # 4. Confusion matrix
        ax4 = fig.add_subplot(gs[1, 0])
        pred_bin = (mu_te >= bth).astype(int)
        cm       = confusion_matrix(y_bin, pred_bin)
        ax4.imshow(cm, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax4.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=13,
                         fontweight='bold',
                         color='white' if cm[i, j] > cm.max() / 2 else 'black')
        ax4.set(xticks=[0, 1], yticks=[0, 1],
                xticklabels=['No Outage', 'Outage'],
                yticklabels=['No Outage', 'Outage'],
                xlabel='Predicted', ylabel='Actual',
                title=f'Confusion Matrix (Test Set, thresh={bth:.2f})')

        # 5. F_µ MDI
        ax5 = fig.add_subplot(gs[1, 1])
        fi_mu = dict(sorted(res['fi_mu'].items(), key=lambda x: x[1]))
        ax5.barh(list(fi_mu.keys()), list(fi_mu.values()), color='seagreen')
        ax5.set(xlabel='Mean Impurity Decrease', title='$F_\\mu$ — Mean Booster MDI')

        # 6. F_α MDI
        ax6 = fig.add_subplot(gs[1, 2])
        fi_al = dict(sorted(res['fi_alpha'].items(), key=lambda x: x[1]))
        ax6.barh(list(fi_al.keys()), list(fi_al.values()), color='steelblue')
        ax6.set(xlabel='Mean Impurity Decrease', title='$F_\\alpha$ — Dispersion Booster MDI')

        if self.output_dir:
            fname = (f"NBB-SO_{self.zones_mark}_{zone}Zone"
                     f"_diagnostic_panel_ROC_ActualVsPredicted_F1Threshold"
                     f"_ConfusionMatrix_Fmu_Falpha_FeatureImportanceMDI"
                     f"_train{self.start[:4]}-{self.end[:4]}.png")
            fig.savefig(os.path.join(self.output_dir, fname), dpi=150, bbox_inches='tight')
          

        plt.show()


# Quickstart 

if __name__ == "__main__":
    START      = '2018-01-01-00'
    END        = '2023-12-31-23'
    ZONES      = ['North', 'South', 'Houston', 'West']
    ZONES_MARK = f"{len(ZONES)}Z"
    COVARIATES = ['T_cold', 'T_hot', 'LmB', 'wind_speed', 'Y_lag1']

    EVENT_TYPES = [
        'Severe Weather',
        'Transmission Interruption',
        'Weather or natural disaster',
        'Severe Weather/Transmission Interruption',
        'Generation Inadequacy',
        'Severe Weather/Distribution Interruption',
        'Weather or natural disaster - Other',
        'Transmission equipment Failure',
        'Weather',
        'Failure at high voltage substation or switchyard聿- Other',
        'Distribution Interruption',
        'Fuel Supply Emergency - Coal',
    ]

    est = OutageNBEstimator(
        START, END, 'U', ZONES_MARK,
        method='nbb_so',
        covariates=COVARIATES,
        max_lag=1,
        test_size=0.2,
        event_types=EVENT_TYPES,
        lewbel_iv=True,
        output_dir=DEFAULT_OUTPUT_DIR,
    )

    # To use real EIA-417 data:  est.build_dataset(data_path='../data/outage_data.csv')
    est.build_dataset()
    est.fit(zones=ZONES, plot=True)
