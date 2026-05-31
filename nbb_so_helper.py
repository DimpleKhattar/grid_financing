"""
Author: Dimple
Created on 25 May 2026
Purpose: NBB-SO Helper Utilities — Financing the Grid Project

Model: Negative Binomial Boosted — Simultaneous Optimization (NBB-SO)


ASSUMPTIONS & MARKET STANDARDS
-------------------------------
1. MDI (Mean Impurity Decrease) is averaged across all trees in each ensemble.
   MDI can overstate importance for high-cardinality continuous features; use
   with caution if covariates have very different cardinalities.

2. F_µ and F_α are separate boosting ensembles. A feature important in F_µ
   but not F_α shifts expected outage count without affecting overdispersion,
   and vice versa. This decomposition is unique to NBB-SO vs standard GBM-NB.

3. Figures saved at 150 dpi, tight bounding box — sufficient for academic
   publication and slide-deck use.

"""

import os
import matplotlib.pyplot as plt


def plot_nbb_so_feature_importance(
    zone_result,
    figsize=(12, 4),
    output_dir=None,
    zones_mark='',
    start_year='',
    end_year='',
):
    """
    Plot F_µ and F_α MDI feature importances side by side for one zone.

    Parameters
    ----------
    zone_result : dict   one entry from OutageNBEstimator.zone_results_[zone]
    figsize     : tuple
    output_dir  : str    if provided, figure is saved here
    zones_mark  : str    e.g. '4Z' — included in saved filename
    start_year  : str    training start year for filename
    end_year    : str    training end year for filename

    """
    zone     = zone_result['zone']
    fi_mu    = zone_result['fi_mu']
    fi_alpha = zone_result['fi_alpha']

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(
        f'NBB-SO Feature Importance — {zone} Zone\n'
        f'Mean Impurity Decrease: F_µ (Mean Booster) vs F_α (Dispersion Booster)',
        fontsize=11, fontweight='bold'
    )

    for ax, fi, title, color in [
        (axes[0], fi_mu,
         '$F_\\mu$ — Mean Booster MDI\n(drives expected outage count)', 'seagreen'),
        (axes[1], fi_alpha,
         '$F_\\alpha$ — Dispersion Booster MDI\n(drives overdispersion)', 'steelblue'),
    ]:
        sorted_fi = dict(sorted(fi.items(), key=lambda x: x[1]))
        ax.barh(list(sorted_fi.keys()), list(sorted_fi.values()), color=color)
        ax.set_xlabel('Mean Impurity Decrease')
        ax.set_title(title)

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fname = (f"NBB-SO_{zones_mark}_{zone}Zone"
                 f"_feature_importance_Fmu_MeanBooster_Falpha_DispersionBooster_MDI"
                 f"_train{start_year}-{end_year}.png")
        fig.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
       

    return fig
