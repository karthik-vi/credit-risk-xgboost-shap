# Experiment log (newest first)

German Credit (UCI 144), 1,000 rows. 5 repeated stratified 70/30 splits (seeds 0–4); hyper-parameters by 5-fold CV AUC on train;
decision threshold minimises the 5:1 cost matrix on out-of-fold train predictions. 2026-11-03. Raw: `results/metrics.json`, `results/train_log.txt`.

## EXP-002 — Model comparison (test, mean ± std over 5 splits)

| Model | ROC AUC | PR AUC | KS | Brier | Expected cost / applicant | Approval rate |
|---|---|---|---|---|---|---|
| Logistic regression (challenger) | 0.792 ± 0.011 | 0.604 | 0.471 | 0.164 | 0.533 | 0.319 |
| XGBoost default (100 trees) | 0.762 ± 0.006 | 0.584 | 0.437 | 0.197 | 0.558 | 0.430 |
| XGBoost tuned | **0.797 ± 0.010** | **0.634** | **0.483** | **0.160** | **0.511** | 0.337 |
| **XGBoost tuned + monotone constraints (reported model)** | 0.793 ± 0.008 | 0.621 | 0.467 | 0.163 | 0.531 | 0.301 |
| XGBoost tuned *with* protected attributes | 0.789 ± 0.017 | 0.619 | 0.466 | 0.166 | 0.539 | 0.313 |

- Seed-0 bootstrap 95% CI for the reported model's AUC: [0.746, 0.855]. All AUC differences between tuned models are inside that interval.
- Tuning matters (default XGBoost is the worst model); XGBoost vs logistic regression is a near-tie on 1,000 rows.
- Monotone constraints cost ~0.004 AUC and ~0.02 cost/applicant. Kept, because they make every explanation directionally defensible.
- Adding protected attributes did not improve AUC (0.789 vs 0.797) and made fairness gaps larger (EXP-001).
- Tuned params (seed 0, monotone): max_depth 3, learning_rate 0.03, 200 trees, min_child_weight 1, subsample 0.8, colsample 0.8.

## EXP-001 — SHAP and fairness (seed 0 test split, n=300)

- **SHAP global (mean |SHAP|, log-odds):** checking_status 0.565, purpose 0.337, duration_months 0.322, credit_history 0.302,
  employment_since 0.228, savings_status 0.217. Additivity check: max |base + Σ SHAP − margin| = 2.9e-6.
- **Local reason codes:** top-4 risk-increasing features for declined applicants, e.g. test row 0 (p_default 0.122, declined because the cost-optimal
  threshold is low): credit_amount 10,144 DM (+0.83), duration 60 months (+0.33). Plots: `docs/figures/shap_local_*.png`.
- **Fairness (approval-rate ratio min/max; four-fifths rule reference = 0.80):**

| Model | sex (F vs M) | age (<25 vs ≥25) | foreign worker (yes vs no) | equal-opportunity gap (sex / age / foreign) |
|---|---|---|---|---|
| Logistic regression | 0.65 | 0.40 | 0.59 | 0.10 / 0.13 / 0.11 |
| XGB monotone, protected attrs **dropped** | 0.73 | 0.61 | 0.61 | 0.07 / 0.08 / 0.14 |
| XGB with protected attrs | 0.75 | 0.46 | 0.45 | 0.13 / 0.23 / 0.40 |

- **Failure / finding:** every model fails the four-fifths reference on every attribute, **including the models that never see the attribute** —
  proxies (duration, savings, employment length, housing) carry age and status information. Dropping protected attributes reduced but did not remove disparity.
  Group sizes are small (e.g. 37 foreign-worker "no" in the full data), so these ratios are noisy.
- **Surprise:** the cost-optimal threshold approves only ~30% of applicants; with a 5:1 cost and 30% base default rate, rejecting is cheap.

## Resume-claim comparison
The resume bullet ("XGBoost credit risk model with SHAP Explainable AI to meet regulatory validation requirements") has no metric.
Reproduced: an XGBoost model with SHAP global/local explanations and a model-card style validation (`docs/MODEL_CARD.md`). "Meeting regulatory
requirements" cannot be reproduced; the model card lists what a validator would still ask for.
