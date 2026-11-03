# Data provenance

| Item | Value |
|---|---|
| Dataset | Statlog (German Credit Data), UCI Machine Learning Repository id 144 (Hofmann, 1994) |
| URL | https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data (zip: `/static/public/144/statlog+german+credit+data.zip`, 29,607 bytes, downloaded 2026-11-03) |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0), as stated on the UCI dataset page (checked 2026-11-03) |
| Size | 1,000 applicants, 20 attributes (13 categorical, 7 numeric), label good (700) / bad (300) |
| How obtained | `scripts/train_eval.py` downloads it to `data/` on first run (`credit.data.download`) |
| Committed | Only `tests/fixture_german_20rows.data` (first 20 rows, CC BY 4.0, for unit tests) |

**This is a public research dataset from 1994, not customer data from Jocata or any lender.** Currency is Deutsche Mark.

## Preprocessing

- Attribute codes (A11, A12, …) are decoded to readable categories from `german.doc` (`credit.data.CODES`).
- Label: `default = 1` if the original class is 2 ("bad").
- Derived columns **for fairness checks only**: `sex` (from attribute 9 "personal status and sex"), `age_group` (<25 / ≥25).
- The production-style models drop all three protected/sensitive attributes (`personal_status_sex`, `age_years`, `foreign_worker`).
  One variant keeps them, to measure what including them changes.
- Cost matrix from `german.doc`: approving a bad applicant costs 5, rejecting a good one costs 1. Thresholds minimise this cost.

## Limitations

- 1,000 rows: test sets of 300 → AUC has a ±0.05-wide 95% CI (bootstrap in EXPERIMENTS). Five repeated splits are reported.
- Attribute 9 mixes sex and marital status; there is no "single female" in the data that can be separated from other statuses cleanly
  (A95 has no rows), so sex-based fairness numbers are coarse.
- 1970s–90s German lending practice; class balance (30% bad) is artificial (the sample was stratified), so approval rates
  and costs are not representative of any real portfolio.
