"""UCI Statlog (German Credit Data), id 144, CC BY 4.0. 1,000 applicants, 20 attributes, label good/bad.
Source: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data . Historical (1994) public data,
NOT customer data from any employer."""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"
COLS = ["checking_status", "duration_months", "credit_history", "purpose", "credit_amount", "savings_status",
        "employment_since", "installment_rate_pct", "personal_status_sex", "other_debtors", "residence_since",
        "property", "age_years", "other_installment_plans", "housing", "existing_credits", "job",
        "num_dependents", "telephone", "foreign_worker", "target"]
# Code meanings from german.doc (shortened)
CODES = {
    "checking_status": {"A11": "<0 DM", "A12": "0-200 DM", "A13": ">=200 DM", "A14": "no account"},
    "credit_history": {"A30": "no credits/all paid", "A31": "all paid here", "A32": "existing paid duly", "A33": "past delays", "A34": "critical/other credits"},
    "purpose": {"A40": "car new", "A41": "car used", "A42": "furniture", "A43": "radio/tv", "A44": "appliances", "A45": "repairs",
                "A46": "education", "A47": "vacation", "A48": "retraining", "A49": "business", "A410": "other"},
    "savings_status": {"A61": "<100 DM", "A62": "100-500 DM", "A63": "500-1000 DM", "A64": ">=1000 DM", "A65": "unknown/none"},
    "employment_since": {"A71": "unemployed", "A72": "<1y", "A73": "1-4y", "A74": "4-7y", "A75": ">=7y"},
    "personal_status_sex": {"A91": "male div/sep", "A92": "female div/sep/mar", "A93": "male single", "A94": "male mar/wid", "A95": "female single"},
    "other_debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    "property": {"A121": "real estate", "A122": "savings/insurance", "A123": "car/other", "A124": "unknown/none"},
    "other_installment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "free"},
    "job": {"A171": "unemployed/unskilled nonres", "A172": "unskilled res", "A173": "skilled", "A174": "management/self-emp"},
    "telephone": {"A191": "none", "A192": "yes"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
}
PROTECTED = ["personal_status_sex", "age_years", "foreign_worker"]
# Cost matrix from german.doc: predicting good for a bad applicant costs 5, the reverse costs 1.
COST_FN, COST_FP = 5.0, 1.0


def download(dest: Path) -> Path:
    f = dest / "german.data"
    if not f.exists():
        dest.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=120) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(dest)
    return f


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS)
    for c, m in CODES.items():
        df[c] = df[c].map(m).astype("category")
    df["default"] = (df.pop("target") == 2).astype(int)  # 1 = bad credit risk
    df["sex"] = df["personal_status_sex"].astype(str).str.split().str[0]  # derived, for fairness checks only
    df["age_group"] = (df["age_years"] < 25).map({True: "<25", False: ">=25"})
    return df
