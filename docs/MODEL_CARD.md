# Model card — German Credit XGBoost (monotone) — reconstruction

**Purpose.** Demonstrate an explainable credit-default model and its validation package. Not for real lending decisions.

| | |
|---|---|
| Model | XGBoost 2.1.4, 200 trees, depth 3, lr 0.03, native categoricals, monotone ↑ risk in duration, amount, instalment rate |
| Inputs | 17 application attributes; **excluded:** personal_status_sex, age_years, foreign_worker |
| Output | P(default); decision threshold minimises expected cost with 5:1 (bad approved : good rejected) |
| Training data | UCI German Credit, 1,000 applicants (1994, DM), 30% bad; see DATA.md |
| Performance (5 splits) | AUC 0.793 ± 0.008, PR AUC 0.621, KS 0.467, Brier 0.163, cost 0.531/applicant |

## Validation questions a reviewer would ask — and current answers

| Question | Status |
|---|---|
| Is performance stable across samples? | 5 splits, std 0.008 AUC; seed-0 bootstrap CI [0.746, 0.855]. Small data → wide CI. |
| Does it beat a simple, interpretable challenger? | Barely: logistic regression AUC 0.792. A validator may prefer the challenger. |
| Are explanations faithful? | SHAP TreeExplainer, exact; additivity error 2.9e-6. Monotone constraints prevent counter-intuitive directions for 3 variables. |
| Can a declined applicant get reasons? | Top-4 risk-increasing features per decision (`shap_report` local examples). Wording for customers not written. |
| Is it calibrated? | Brier 0.163; no calibration curve or recalibration step yet. **Open.** |
| Disparate impact? | **Fails** four-fifths reference for sex (0.73), age <25 (0.61), foreign workers (0.61) even without those inputs. **Open — blocking for real use.** |
| Proxy variables? | Not analysed beyond the observation above. **Open.** |
| Population stability / drift monitoring? | Not implemented. **Open.** |
| Reject inference? | Not addressed (data contains only booked loans). **Open.** |
| Data lineage? | Public UCI dataset, fixed URL, deterministic preprocessing. |

## Limitations
1,000 rows; 1990s German data; derived "sex" field mixes marital status; no time dimension for out-of-time validation.
