"""Undersampling: keeps positives, hits the ratio, is deterministic, rejects test rows."""

import numpy as np
import pandas as pd
import pytest

from config import TRAIN_FOLDS
from undersample import undersample_train


def make_frame(n_pos=100, n_neg=1000, fold=10):
    n = n_pos + n_neg
    return pd.DataFrame(
        {
            "encounter_id": np.arange(n),
            "patient_nbr": np.arange(n) % 500,
            "target": [1] * n_pos + [0] * n_neg,
            "fold": [fold] * n,
        }
    )


def test_keeps_every_positive():
    df = make_frame()
    out, _ = undersample_train(df, df["fold"])
    assert out["target"].sum() == df["target"].sum()


def test_hits_the_ratio():
    df = make_frame()
    out, _ = undersample_train(df, df["fold"])
    n_pos = int((out["target"] == 1).sum())
    n_neg = int((out["target"] == 0).sum())
    assert abs(n_neg / n_pos - 1.857) < 0.01


def test_r_is_the_sampling_fraction():
    df = make_frame()
    out, r = undersample_train(df, df["fold"])
    n_neg = int((out["target"] == 0).sum())
    assert r == pytest.approx(n_neg / 1000)


def test_deterministic():
    df = make_frame()
    a, ra = undersample_train(df, df["fold"])
    b, rb = undersample_train(df, df["fold"])
    assert a["encounter_id"].tolist() == b["encounter_id"].tolist()
    assert ra == rb


def test_rejects_test_rows():
    df = make_frame(fold=0)  # fold 0 is test
    assert 0 not in TRAIN_FOLDS
    with pytest.raises(AssertionError):
        undersample_train(df, df["fold"])


def test_rejects_mixed_folds():
    df = make_frame()
    df.loc[:10, "fold"] = 3  # fold 3 is validation
    with pytest.raises(AssertionError):
        undersample_train(df, df["fold"])


def test_no_duplicate_rows():
    df = make_frame()
    out, _ = undersample_train(df, df["fold"])
    assert not out["encounter_id"].duplicated().any()
    assert len(out) == len(df[df.target == 1]) + int(round(1.857 * 100))


def test_accepts_fold_as_array():
    df = make_frame()
    out, r = undersample_train(df, df["fold"].to_numpy())
    assert out["target"].sum() == 100
    assert 0 < r < 1
