"""
Small, dependency-light statistics helpers used across the notebooks.

Everything here is implemented with numpy/scipy so the notebooks run without
statsmodels. Where statsmodels offers the same thing (OLS summary, proportions
z-test) the function name says which one it mirrors, so the results can be
cross-checked.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------- proportions

@dataclass
class ProportionTest:
    p_control: float
    p_treatment: float
    abs_lift: float
    rel_lift: float
    z: float
    p_value: float
    ci_low: float
    ci_high: float

    def summary(self) -> str:
        return (f"control {self.p_control:.3%} vs treatment {self.p_treatment:.3%} | "
                f"abs lift {self.abs_lift:+.3%} (rel {self.rel_lift:+.1%}) | "
                f"95% CI [{self.ci_low:+.3%}, {self.ci_high:+.3%}] | z = {self.z:.2f}, p = {self.p_value:.4f}")


def two_proportion_ztest(x_c: int, n_c: int, x_t: int, n_t: int, alpha: float = 0.05) -> ProportionTest:
    """Pooled two-sided z-test for p_t - p_c (mirrors statsmodels proportions_ztest),
    with an unpooled Wald confidence interval for the difference."""
    p_c, p_t = x_c / n_c, x_t / n_t
    p_pool = (x_c + x_t) / (n_c + n_t)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    z = (p_t - p_c) / se_pool
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    se_diff = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    zcrit = stats.norm.ppf(1 - alpha / 2)
    diff = p_t - p_c
    return ProportionTest(p_c, p_t, diff, diff / p_c, z, p_value, diff - zcrit * se_diff, diff + zcrit * se_diff)


def sample_size_two_proportions(p_base: float, mde_rel: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Per-group sample size to detect a relative lift `mde_rel` on baseline rate `p_base`."""
    p2 = p_base * (1 + mde_rel)
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    p_bar = (p_base + p2) / 2
    num = (z_a * np.sqrt(2 * p_bar * (1 - p_bar)) + z_b * np.sqrt(p_base * (1 - p_base) + p2 * (1 - p2))) ** 2
    return int(np.ceil(num / (p2 - p_base) ** 2))


def srm_check(counts: dict[str, int], expected_ratio: dict[str, float] | None = None) -> tuple[float, float]:
    """Sample-ratio-mismatch chi-square test. Returns (chi2, p_value)."""
    keys = list(counts)
    obs = np.array([counts[k] for k in keys], dtype=float)
    if expected_ratio is None:
        exp = np.full_like(obs, obs.sum() / len(obs))
    else:
        exp = obs.sum() * np.array([expected_ratio[k] for k in keys])
    chi2, p = stats.chisquare(obs, exp)
    return float(chi2), float(p)


# --------------------------------------------------------------------------- means

def welch_ttest(a, b) -> dict:
    """Welch's t-test with Cohen's d and a 95% CI for the difference in means."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    df = se ** 4 / ((a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1) + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1))
    tcrit = stats.t.ppf(0.975, df)
    pooled_sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return {"mean_a": a.mean(), "mean_b": b.mean(), "diff": diff, "ci_low": diff - tcrit * se, "ci_high": diff + tcrit * se,
            "t": t, "df": df, "p_value": p, "cohens_d": diff / pooled_sd}


def bootstrap_ci(x, stat=np.mean, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    boots = np.array([stat(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)])
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def bootstrap_diff_ci(a, b, stat=np.mean, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    boots = np.array([stat(rng.choice(a, len(a))) - stat(rng.choice(b, len(b))) for _ in range(n_boot)])
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def pairwise_welch_bonferroni(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """All pairwise Welch t-tests with Bonferroni-adjusted p-values (post-hoc after ANOVA)."""
    groups = sorted(df[group_col].unique())
    rows = []
    for i, g1 in enumerate(groups):
        for g2 in groups[i + 1:]:
            r = welch_ttest(df.loc[df[group_col] == g1, value_col], df.loc[df[group_col] == g2, value_col])
            rows.append({"group_1": g1, "group_2": g2, "diff": r["diff"], "p_raw": r["p_value"]})
    out = pd.DataFrame(rows)
    out["p_bonferroni"] = (out["p_raw"] * len(out)).clip(upper=1.0)
    out["significant_5pct"] = out["p_bonferroni"] < 0.05
    return out


# --------------------------------------------------------------------------- regression

@dataclass
class OLSResult:
    params: pd.Series
    bse: pd.Series
    tvalues: pd.Series
    pvalues: pd.Series
    conf_int: pd.DataFrame
    rsquared: float
    rsquared_adj: float
    fitted: np.ndarray
    resid: np.ndarray
    nobs: int
    df_resid: int

    def summary(self) -> pd.DataFrame:
        tbl = pd.DataFrame({"coef": self.params, "std_err": self.bse, "t": self.tvalues, "p_value": self.pvalues,
                            "ci_2.5%": self.conf_int.iloc[:, 0], "ci_97.5%": self.conf_int.iloc[:, 1]})
        return tbl


def ols(y, X: pd.DataFrame, add_const: bool = True) -> OLSResult:
    """Ordinary least squares with classical standard errors (mirrors statsmodels.OLS)."""
    X = X.copy()
    if add_const:
        X.insert(0, "const", 1.0)
    Xm = X.to_numpy(float)
    yv = np.asarray(y, float)
    n, k = Xm.shape
    beta, *_ = np.linalg.lstsq(Xm, yv, rcond=None)
    fitted = Xm @ beta
    resid = yv - fitted
    df_resid = n - k
    sigma2 = resid @ resid / df_resid
    cov = sigma2 * np.linalg.inv(Xm.T @ Xm)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), df_resid))
    tcrit = stats.t.ppf(0.975, df_resid)
    ss_tot = ((yv - yv.mean()) ** 2).sum()
    r2 = 1 - (resid @ resid) / ss_tot
    r2_adj = 1 - (1 - r2) * (n - 1) / df_resid
    idx = X.columns
    return OLSResult(pd.Series(beta, idx), pd.Series(se, idx), pd.Series(t, idx), pd.Series(p, idx),
                     pd.DataFrame({"low": beta - tcrit * se, "high": beta + tcrit * se}, index=idx),
                     r2, r2_adj, fitted, resid, n, df_resid)


def vif(X: pd.DataFrame) -> pd.Series:
    """Variance inflation factor per column (no constant in X)."""
    out = {}
    for col in X.columns:
        others = X.drop(columns=col)
        r = ols(X[col], others)
        out[col] = 1 / (1 - r.rsquared) if r.rsquared < 1 else np.inf
    return pd.Series(out, name="VIF")


def adstock(x, decay: float) -> np.ndarray:
    """Geometric ad-stock: carried-over effect of past spend."""
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    carry = 0.0
    for i, v in enumerate(x):
        carry = v + decay * carry
        out[i] = carry
    return out
