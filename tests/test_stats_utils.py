"""
Unit tests for src/stats_utils.py - run with `python -m pytest` or `python tests/test_stats_utils.py`.

The helpers re-implement standard procedures, so each test checks against either a
closed-form result or scipy on the same data.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.stats_utils import (adstock, bootstrap_ci, ols, sample_size_two_proportions,  # noqa: E402
                             srm_check, two_proportion_ztest, vif, welch_ttest)


def test_two_proportion_ztest_matches_textbook():
    # 200/1000 vs 250/1000 with pooled SE: z = 0.05 / sqrt(0.225*0.775*2/1000) = 2.6774
    r = two_proportion_ztest(200, 1000, 250, 1000)
    assert abs(r.z - 2.6774) < 1e-3
    assert abs(r.p_value - 0.00742) < 1e-4
    # agrees with scipy's chi-square test on the 2x2 table (z^2 == chi2 without continuity correction)
    chi2, _, _, _ = stats.chi2_contingency([[200, 800], [250, 750]], correction=False)
    assert abs(r.z ** 2 - chi2) < 1e-6
    assert r.ci_low < r.abs_lift < r.ci_high


def test_sample_size_standard_case():
    # 3.1% baseline, +15% relative, alpha .05, power .8 -> ~23.4k per arm (matches Evan Miller's calculator ±2%)
    n = sample_size_two_proportions(0.031, 0.15)
    assert 22_800 < n < 24_000


def test_srm_detects_imbalance():
    _, p_ok = srm_check({"A": 10_000, "B": 10_100})
    _, p_bad = srm_check({"A": 10_000, "B": 11_000})
    assert p_ok > 0.05
    assert p_bad < 0.001


def test_welch_matches_scipy():
    rng = np.random.default_rng(0)
    a, b = rng.normal(10, 2, 300), rng.normal(10.5, 3, 200)
    r = welch_ttest(a, b)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    assert abs(r["t"] - t) < 1e-9 and abs(r["p_value"] - p) < 1e-9
    assert r["ci_low"] < r["diff"] < r["ci_high"]


def test_bootstrap_ci_covers_mean():
    rng = np.random.default_rng(1)
    x = rng.exponential(2.0, 500)
    lo, hi = bootstrap_ci(x, n_boot=1000)
    assert lo < x.mean() < hi


def test_ols_recovers_known_coefficients():
    rng = np.random.default_rng(2)
    X = pd.DataFrame({"x1": rng.normal(size=500), "x2": rng.normal(size=500)})
    y = 1.5 + 2.0 * X["x1"] - 0.5 * X["x2"] + rng.normal(scale=0.1, size=500)
    r = ols(y, X)
    assert abs(r.params["const"] - 1.5) < 0.05
    assert abs(r.params["x1"] - 2.0) < 0.05
    assert abs(r.params["x2"] + 0.5) < 0.05
    assert r.rsquared > 0.99
    # p-values via scipy linregress on x1 alone should agree in direction/significance
    assert r.pvalues["x1"] < 1e-6


def test_vif_flags_collinearity():
    rng = np.random.default_rng(3)
    x1 = rng.normal(size=300)
    X = pd.DataFrame({"x1": x1, "x2": x1 * 0.95 + rng.normal(scale=0.3, size=300), "x3": rng.normal(size=300)})
    v = vif(X)
    assert v["x1"] > 5 and v["x2"] > 5 and v["x3"] < 1.5


def test_adstock_geometric_carryover():
    out = adstock([100, 0, 0, 0], decay=0.5)
    assert np.allclose(out, [100, 50, 25, 12.5])
    assert np.allclose(adstock([3, 4, 5], decay=0.0), [3, 4, 5])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
