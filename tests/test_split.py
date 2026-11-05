"""Checks on data/processed/split_assignment.csv."""

import pandas as pd
import pytest

from config import COHORT_PATH, N_SPLITS, SPLIT_PATH, fold_to_split


@pytest.fixture(scope="module")
def assignment():
    assert SPLIT_PATH.exists(), "run src/split.py first"
    return pd.read_csv(SPLIT_PATH)


@pytest.fixture(scope="module")
def cohort():
    return pd.read_parquet(COHORT_PATH, columns=["encounter_id", "patient_nbr", "target"])


def test_columns(assignment):
    assert list(assignment.columns) == ["patient_nbr", "fold"]


def test_one_row_per_patient(assignment):
    assert assignment["patient_nbr"].is_unique
    assert len(assignment) == 69_987


def test_fold_values(assignment):
    assert assignment["fold"].between(0, N_SPLITS - 1).all()
    assert assignment["fold"].nunique() == N_SPLITS


def test_every_admission_has_a_fold(assignment, cohort):
    merged = cohort.merge(assignment, on="patient_nbr", how="left")
    assert merged["fold"].notna().all()
    assert len(merged) == 99_340


def test_no_patient_in_two_folds(assignment, cohort):
    merged = cohort.merge(assignment, on="patient_nbr", how="left")
    per_patient = merged.groupby("patient_nbr")["fold"].nunique()
    assert (per_patient == 1).all()


def test_held_out_share(assignment):
    total = len(assignment)
    test_share = assignment["fold"].map(lambda f: fold_to_split(f) == "test").mean()
    val_share = assignment["fold"].map(lambda f: fold_to_split(f) == "val").mean()
    assert 0.14 <= test_share <= 0.16
    assert 0.14 <= val_share <= 0.16


def test_prevalence_per_fold(assignment, cohort):
    merged = cohort.merge(assignment, on="patient_nbr", how="left")
    prev = merged.groupby("fold")["target"].mean()
    assert (prev - 11_314 / 99_340).abs().max() <= 0.01


def test_split_shares_are_right(assignment):
    counts = assignment["fold"].map(fold_to_split).value_counts()
    assert counts["test"] == len(assignment[assignment.fold.isin([0, 1, 2])])
    assert counts["val"] == len(assignment[assignment.fold.isin([3, 4, 5])])
    assert sum(counts.values) == 69_987
