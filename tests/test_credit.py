from pathlib import Path

import numpy as np
import pandas as pd

from credit.data import COLS, load
from credit.model import best_threshold, expected_cost, fairness, features, ks_stat

FIXTURE = Path(__file__).parent / "fixture_german_20rows.data"


def test_load_decodes_codes_and_target():
    df = load(FIXTURE)
    assert len(df) == 20 and set(df["default"].unique()) <= {0, 1}
    assert "no account" in set(df["checking_status"].astype(str)) or "<0 DM" in set(df["checking_status"].astype(str))
    assert set(df["age_group"]) <= {"<25", ">=25"}


def test_protected_attributes_can_be_dropped():
    df = load(FIXTURE)
    X = features(df, drop_protected=True)
    assert not {"personal_status_sex", "age_years", "foreign_worker", "sex", "age_group", "default"} & set(X.columns)


def test_cost_and_threshold():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert expected_cost(y, p, 0.5) == 0.0
    assert expected_cost(y, p, 0.95) == 5.0 * 2 / 4  # both bads approved
    assert 0.2 < best_threshold(y, p) <= 0.8
    assert ks_stat(y, p) == 1.0


def test_fairness_table_shape():
    df = load(FIXTURE)
    p = np.linspace(0, 1, len(df))
    f = fairness(df, p, 0.5)
    assert set(f) == {"sex", "age_group", "foreign_worker"}
    assert all("approval_rate_ratio_min_max" in v for v in f.values())
