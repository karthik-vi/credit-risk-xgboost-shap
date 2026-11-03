"""Training, evaluation, SHAP and fairness utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

from .data import COST_FN, COST_FP, PROTECTED

NUMERIC = ["duration_months", "credit_amount", "installment_rate_pct", "residence_since", "age_years", "existing_credits", "num_dependents"]
# Monotone constraints (risk direction a credit officer would defend): longer duration / larger amount /
# higher instalment rate -> risk must not decrease. 0 = unconstrained.
MONOTONE = {"duration_months": 1, "credit_amount": 1, "installment_rate_pct": 1}


def features(df: pd.DataFrame, drop_protected: bool) -> pd.DataFrame:
    X = df.drop(columns=["default", "sex", "age_group"])
    if drop_protected:
        X = X.drop(columns=PROTECTED)
    return X


def ks_stat(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def expected_cost(y, p, thr) -> float:
    pred_bad = p >= thr
    fn = np.sum((y == 1) & ~pred_bad)  # bad applicant approved
    fp = np.sum((y == 0) & pred_bad)   # good applicant rejected
    return float((COST_FN * fn + COST_FP * fp) / len(y))


def best_threshold(y, p) -> float:
    grid = np.linspace(0.05, 0.95, 91)
    return float(grid[np.argmin([expected_cost(y, p, t) for t in grid])])


def metrics(y, p, thr) -> dict:
    return {"roc_auc": round(roc_auc_score(y, p), 4), "pr_auc": round(average_precision_score(y, p), 4),
            "brier": round(brier_score_loss(y, p), 4), "ks": round(ks_stat(y, p), 4),
            "threshold": round(thr, 3), "expected_cost_per_applicant": round(expected_cost(y, p, thr), 4),
            "approval_rate": round(float(np.mean(p < thr)), 4)}


def bootstrap_auc(y, p, n=1000, seed=0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def fairness(df_test: pd.DataFrame, p, thr) -> dict:
    """Approval-rate parity and equal opportunity (approval rate among truly good applicants) per group."""
    out = {}
    approved = p < thr
    good = df_test["default"].values == 0
    for attr in ("sex", "age_group", "foreign_worker"):
        g = df_test[attr].astype(str).values
        rows = {}
        for v in sorted(set(g)):
            m = g == v
            rows[v] = {"n": int(m.sum()), "approval_rate": round(float(approved[m].mean()), 4),
                       "approval_rate_if_good": round(float(approved[m & good].mean()), 4) if (m & good).any() else None,
                       "actual_default_rate": round(float((~good[m]).mean()), 4)}
        ar = [r["approval_rate"] for r in rows.values()]
        tg = [r["approval_rate_if_good"] for r in rows.values() if r["approval_rate_if_good"] is not None]
        out[attr] = {"groups": rows, "approval_rate_ratio_min_max": round(min(ar) / max(ar), 4) if max(ar) else None,
                     "equal_opportunity_gap": round(max(tg) - min(tg), 4) if tg else None}
    return out
