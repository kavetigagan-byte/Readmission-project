
"""
Automated tests for the Diabetic Readmission Cohort.
Author: Role A (Data Owner)
"""

from pathlib import Path

import pandas as pd
import pytest


DATA_PATH = Path("data/processed/cohort_v1.parquet")


@pytest.fixture(scope="module")
def df():
    """Load the cleaned cohort once for all tests."""

    assert DATA_PATH.exists(), (
        "Cleaned cohort Parquet file does not exist."
    )

    return pd.read_parquet(DATA_PATH)


# 1. Dataset existence
def test_dataset_exists():
    assert DATA_PATH.exists()


# 2. Expected row count
def test_expected_row_count(df):
    assert len(df) == 69987


# 3. Expected positive readmission cases
def test_expected_positive_cases(df):
    assert df["target"].sum() == 6285


# 4. Binary target validation
def test_target_is_binary(df):
    assert set(df["target"].unique()) == {0, 1}


# 5. Unique patient numbers
def test_patient_numbers_are_unique(df):
    assert df["patient_nbr"].is_unique


# 6. Valid gender values
def test_gender_values_are_valid(df):
    assert set(df["gender"].astype(str).unique()) == {
        "Female",
        "Male",
    }


# 7. Valid age midpoint values
def test_age_values_are_valid(df):

    valid_ages = {
        5, 15, 25, 35, 45,
        55, 65, 75, 85, 95,
    }

    assert set(df["age"].unique()).issubset(valid_ages)


# 8. Diagnosis group columns
def test_diagnosis_groups_exist(df):

    required_columns = [
        "diag_1_group",
        "diag_2_group",
        "diag_3_group",
    ]

    for column in required_columns:
        assert column in df.columns


# 9. Admission mapping description columns
def test_mapping_descriptions_exist(df):

    required_columns = [
        "admission_type_id_desc",
        "discharge_disposition_id_desc",
        "admission_source_id_desc",
    ]

    for column in required_columns:
        assert column in df.columns


# 10. Unique encounter IDs
def test_no_duplicate_encounters(df):
    assert df["encounter_id"].is_unique
