"""Train/evaluate credit-risk models on German Credit, with SHAP explanations and fairness checks.

Protocol: 5 repeated stratified 70/30 splits (seeds 0-4). Hyper-parameters and the decision threshold are
chosen by 5-fold CV on the training part only (threshold minimises the dataset's 5:1 cost matrix).
Seed 0 is used for SHAP plots, reason codes and the fairness table.

  python scripts/train_eval.py --out results/metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from credit.data import download, load  # noqa: E402
from credit.model import MONOTONE, best_threshold, bootstrap_auc, fairness, features, metrics  # noqa: E402

GRID = [dict(max_depth=d, learning_rate=lr, n_estimators=n, min_child_weight=mcw, subsample=0.8, colsample_bytree=0.8)
        for d in (2, 3, 4) for lr in (0.03, 0.1) for n in (200, 400) for mcw in (1, 5)]


def make_xgb(params, X, monotone, seed):
    mono = tuple(MONOTONE.get(c, 0) for c in X.columns) if monotone else None
    return XGBClassifier(**params, enable_categorical=True, tree_method="hist", eval_metric="logloss",
                         random_state=seed, n_jobs=2, **({"monotone_constraints": mono} if mono else {}))


def make_lr(X):
    cat = [c for c in X.columns if str(X[c].dtype) == "category"]
    num = [c for c in X.columns if c not in cat]
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat), ("num", StandardScaler(), num)])
    return make_pipeline(pre, LogisticRegression(max_iter=2000, C=0.3))


def cv_oof(builder, X, y, seed):
    """Out-of-fold predictions on the training set (for model selection and threshold choice)."""
    oof = np.zeros(len(y))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m = builder()
        m.fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return oof


def run_split(df, seed, out_dir=None):
    from sklearn.metrics import roc_auc_score

    tr_df, te_df = train_test_split(df, test_size=0.3, stratify=df["default"], random_state=seed)
    ytr, yte = tr_df["default"].values, te_df["default"].values
    res = {}
    variants = {
        "logreg_baseline": (True, "lr", False),
        "xgb_default": (True, "xgb_default", False),
        "xgb_tuned": (True, "xgb_tuned", False),
        "xgb_tuned_monotone": (True, "xgb_tuned", True),
        "xgb_tuned_with_protected_attrs": (False, "xgb_tuned", False),
    }
    tuned_cache = {}
    models = {}
    for name, (drop_prot, kind, mono) in variants.items():
        Xtr, Xte = features(tr_df, drop_prot), features(te_df, drop_prot)
        if kind == "lr":
            builder = lambda: make_lr(Xtr)  # noqa: E731
        elif kind == "xgb_default":
            builder = lambda: make_xgb(dict(n_estimators=100), Xtr, False, seed)  # noqa: E731
        else:
            key = (drop_prot, mono)
            if key not in tuned_cache:
                scores = []
                for g in GRID:
                    oof = cv_oof(lambda: make_xgb(g, Xtr, mono, seed), Xtr, ytr, seed)
                    scores.append(roc_auc_score(ytr, oof))
                tuned_cache[key] = GRID[int(np.argmax(scores))]
            params = tuned_cache[key]
            builder = lambda: make_xgb(params, Xtr, mono, seed)  # noqa: E731
        oof = cv_oof(builder, Xtr, ytr, seed)
        thr = best_threshold(ytr, oof)
        m = builder().fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        r = metrics(yte, p, thr)
        r["cv_oof_auc"] = round(roc_auc_score(ytr, oof), 4)
        if kind == "xgb_tuned":
            r["params"] = tuned_cache[(drop_prot, mono)]
        r["fairness"] = fairness(te_df, p, thr)
        res[name] = r
        models[name] = (m, Xte, p, thr)
    return res, models, te_df


def shap_report(model, Xte, p, thr, te_df, out_dir: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    expl = shap.TreeExplainer(model)
    sv = expl(Xte)
    # global importance
    imp = pd.Series(np.abs(sv.values).mean(0), index=Xte.columns).sort_values(ascending=False)
    plt.figure()
    shap.plots.bar(sv, max_display=12, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_global_bar.png", dpi=120)
    plt.close("all")
    plt.figure()
    shap.plots.beeswarm(sv, max_display=12, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_beeswarm.png", dpi=120)
    plt.close("all")
    # local: reason codes for 3 declined applicants (top features pushing risk UP)
    declined = np.where(p >= thr)[0][:3]
    local = []
    for j, i in enumerate(declined):
        contrib = pd.Series(sv.values[i], index=Xte.columns).sort_values(ascending=False)
        local.append({"test_row": int(i), "p_default": round(float(p[i]), 4), "actual_default": int(te_df["default"].iloc[i]),
                      "base_value": round(float(sv.base_values[i]), 4),
                      "top_reasons_increasing_risk": [{"feature": f, "value": str(Xte.iloc[i][f]), "shap_logodds": round(float(v), 4)}
                                                      for f, v in contrib.head(4).items()]})
        plt.figure()
        shap.plots.waterfall(sv[i], max_display=10, show=False)
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_local_{j}.png", dpi=110)
        plt.close("all")
    # additivity check: base + sum(shap) == model margin
    margin = model.predict(Xte, output_margin=True)
    add_err = float(np.max(np.abs(sv.base_values + sv.values.sum(1) - margin)))
    return {"global_mean_abs_shap": {k: round(float(v), 4) for k, v in imp.items()}, "local_examples": local,
            "additivity_max_abs_error": add_err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("results/metrics.json"))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    a = ap.parse_args()
    df = load(download(a.data_dir))
    seeds = [int(s) for s in a.seeds.split(",")]
    per_seed, report = {}, {"dataset": "UCI Statlog German Credit (id 144)", "n": len(df), "default_rate": round(df["default"].mean(), 4)}
    for s in seeds:
        res, models, te_df = run_split(df, s)
        per_seed[s] = res
        print(s, {k: (v["roc_auc"], v["expected_cost_per_applicant"]) for k, v in res.items()}, flush=True)
        if s == seeds[0]:
            m, Xte, p, thr = models["xgb_tuned_monotone"]
            fig = Path("docs/figures")
            fig.mkdir(parents=True, exist_ok=True)
            report["shap_seed0_xgb_tuned_monotone"] = shap_report(m, Xte, p, thr, te_df, fig)
            lo, hi = bootstrap_auc(te_df["default"].values, p)
            report["seed0_auc_bootstrap_95ci_xgb_tuned_monotone"] = [round(lo, 4), round(hi, 4)]
            report["fairness_seed0"] = {k: v["fairness"] for k, v in res.items()}
    summary = {}
    for name in per_seed[seeds[0]]:
        summary[name] = {}
        for metric in ("roc_auc", "pr_auc", "brier", "ks", "expected_cost_per_applicant", "approval_rate", "cv_oof_auc"):
            v = [per_seed[s][name][metric] for s in seeds]
            summary[name][metric] = {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)}
    report["summary_over_seeds"] = summary
    report["seeds"] = seeds
    report["per_seed"] = {s: {k: {kk: vv for kk, vv in v.items() if kk != "fairness"} for k, v in r.items()} for s, r in per_seed.items()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, default=str))
    for k, v in summary.items():
        print(f"{k:34s} AUC {v['roc_auc']['mean']:.3f}±{v['roc_auc']['std']:.3f}  cost {v['expected_cost_per_applicant']['mean']:.3f}  brier {v['brier']['mean']:.3f}")


if __name__ == "__main__":
    main()
