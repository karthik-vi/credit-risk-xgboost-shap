# Architecture Decision Records

## Period freeze

| | |
|---|---|
| Work period reconstructed | Jocata, ML Engineer, Jan 2024 – Jun 2025 |
| Cutoff | Jun 2025 |
| Python | 3.11 |
| Libraries | XGBoost 2.1.4 (PyPI 2025-02-06), SHAP 0.47.2 (2025-04-17), scikit-learn 1.6.1 (2025-01-10), pandas 2.2.3 (2024-09-20), NumPy 2.2.5 (2025-04-19), matplotlib 3.10.1 (2025-02-27) |
| Transitive deps | `requirements.txt` compiled with `uv pip compile --exclude-newer 2025-06-30` |

No exceptions.

## ADR-001 Data: UCI German Credit — 2026-11-03
- Public, CC BY 4.0, includes a cost matrix and sensitive attributes, which makes it suitable for a regulator-style validation write-up.
- Alternatives: Home Credit (Kaggle login, competition rules), Give Me Some Credit (Kaggle), Taiwan default (UCI, CC BY 4.0, 30k rows).
- Would change my mind: if sample size mattered more than documented attributes → Taiwan default dataset.

## ADR-002 Model: XGBoost with monotone constraints, logistic regression as challenger — 2026-11-03
- XGBoost (native categorical handling) is what the reconstructed work used. A regulator will ask for a simpler challenger model:
  logistic regression on one-hot features is reported side by side.
- Monotone constraints on duration, amount and instalment rate: risk may not decrease when these increase. This makes explanations
  defensible ("a larger loan never made you look safer") at a measured cost in AUC (EXPERIMENTS).

## ADR-003 Explainability: SHAP TreeExplainer — 2026-11-03
- Exact Shapley values for trees, additive in log-odds (additivity error checked and reported).
- Global: mean |SHAP| ranking and beeswarm. Local: waterfall plots and top-4 "reasons increasing risk" for declined applicants
  (the basis for adverse-action reason codes).

## ADR-004 Evaluation and threshold — 2026-11-03
- 5 repeated stratified 70/30 splits; hyper-parameters by 5-fold CV AUC on the training part; threshold minimises the dataset's 5:1 cost
  on out-of-fold training predictions, then applied to test.
- Metrics: ROC AUC, PR AUC, Brier, KS, expected cost per applicant, approval rate; bootstrap 95% CI for AUC.
- Fairness: approval-rate ratio (four-fifths rule reference) and equal-opportunity gap (approval rate among truly good applicants)
  by sex, age group (<25) and foreign-worker status.

## ADR-005 Deployment — 2026-11-03
- Batch scoring script + saved model is enough for this reconstruction; no API. Model card in `docs/MODEL_CARD.md`.
