"""Shared constants for the Role C modelling pipeline.

Everything here is copied from docs/experiment-matrix-v2.md.
Do not change a value without changing the spec first.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

COHORT_PATH = DATA_PROCESSED / "cohort_v2.parquet"
SPLIT_PATH = DATA_PROCESSED / "split_assignment.csv"

# ---------------------------------------------------------------- attributes
# 15 model attributes, chosen before looking at the outcome.
MODEL_ATTRIBUTES = [
    "age",
    "gender",
    "race",
    "time_in_hospital",
    "number_inpatient",
    "number_emergency",
    "number_outpatient",
    "num_lab_procedures",
    "num_medications",
    "number_diagnoses",
    "admission_type_id",
    "discharge_disposition_id",
    "diag_1_group",
    "A1Cresult",
    "insulin",
]

# Never used as features. n_stays_total counts future admissions and leaks.
AUXILIARY_COLUMNS = [
    "encounter_id",
    "patient_nbr",
    "target",
    "is_first_encounter",
    "n_stays_total",
]

NUMERIC_ATTRIBUTES = [
    "time_in_hospital",
    "number_inpatient",
    "number_emergency",
    "number_outpatient",
    "num_lab_procedures",
    "num_medications",
    "number_diagnoses",
]

CATEGORICAL_ATTRIBUTES = [
    c for c in MODEL_ATTRIBUTES if c not in NUMERIC_ATTRIBUTES
]

# ---------------------------------------------------------------------- split
N_SPLITS = 20
TEST_FOLDS = frozenset({0, 1, 2})
VAL_FOLDS = frozenset({3, 4, 5})
TRAIN_FOLDS = frozenset(range(6, 20))


def fold_to_split(fold):
    """Map a fold number to 'train' / 'val' / 'test'."""
    if fold in TEST_FOLDS:
        return "test"
    if fold in VAL_FOLDS:
        return "val"
    if fold in TRAIN_FOLDS:
        return "train"
    raise ValueError(f"unknown fold: {fold!r}")


# --------------------------------------------------------------- class ratio
UNDERSAMPLE_RATIO = 1.857  # negatives per positive in the training fold
RANDOM_SEED = 42
EXPECTED_PREVALENCE = 11_314 / 99_340  # 0.11393..., natural prevalence
