"""Build the patient-level split, once.

    python src/split.py

Output
------
data/processed/split_assignment.csv   patient_nbr, fold
results/split_summary.json            counts and prevalence per fold

Rule (docs/experiment-matrix-v2.md §4)
--------------------------------------
StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42) on patients,
stratified by each patient's FIRST-encounter label.

    folds 0-2   -> test
    folds 3-5   -> val
    folds 6-19  -> train

Every admission of a patient inherits that patient's fold.
"""

import json
import sys

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from config import (
    COHORT_PATH,
    DATA_PROCESSED,
    EXPECTED_PREVALENCE,
    N_SPLITS,
    RANDOM_SEED,
    RESULTS,
    SPLIT_PATH,
    fold_to_split,
)

SPLIT_NAME = {0: "test", 1: "val", 2: "train"}


def load_patients(path=COHORT_PATH):
    """One row per patient: the earliest encounter and its label."""
    df = pd.read_parquet(
        path, columns=["encounter_id", "patient_nbr", "target"]
    )
    first = (
        df.sort_values("encounter_id")
        .drop_duplicates("patient_nbr", keep="first")[["patient_nbr", "target"]]
        .rename(columns={"target": "first_label"})
        .reset_index(drop=True)
    )
    return df, first


def assign_folds(patients):
    """StratifiedGroupKFold on patients; one row per patient means one group each."""
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED
    )
    fold = pd.Series(-1, index=patients.index, dtype="int64")
    for k, (_, held_out) in enumerate(
        splitter.split(patients, patients["first_label"], groups=patients["patient_nbr"])
    ):
        fold.iloc[held_out] = k
    return fold


def check(df, patients, fold):
    """Hard checks. Raise before a bad split is ever written to disk."""
    assert len(patients) == 69_987, f"expected 69,987 patients, got {len(patients):,}"
    assert (fold >= 0).all(), "some patients got no fold"
    assert fold.nunique() == N_SPLITS, f"expected {N_SPLITS} folds, got {fold.nunique()}"
    assert patients["patient_nbr"].is_unique, "duplicate patient_nbr in patient table"

    assignment = dict(zip(patients["patient_nbr"], fold))
    row_fold = df["patient_nbr"].map(assignment)
    assert row_fold.notna().all(), "some admissions inherit no fold"
    assert row_fold.nunique() == N_SPLITS

    # no patient in two folds - guaranteed by the dict mapping, asserted anyway
    per_patient_folds = (
        pd.DataFrame({"patient_nbr": df["patient_nbr"], "fold": row_fold})
        .drop_duplicates()
        .groupby("patient_nbr")
        .size()
    )
    assert (per_patient_folds == 1).all(), "a patient appears in more than one fold"

    # held-out share of patients
    test_share = (fold.isin({0, 1, 2})).mean()
    val_share = (fold.isin({3, 4, 5})).mean()
    assert 0.14 <= test_share <= 0.16, f"test holds {test_share:.2%} of patients"
    assert 0.14 <= val_share <= 0.16, f"val holds {val_share:.2%} of patients"

    # prevalence per fold within 1 percentage point
    prev = (
        pd.DataFrame({"fold": row_fold, "target": df["target"]})
        .groupby("fold")["target"]
        .mean()
    )
    worst = (prev - EXPECTED_PREVALENCE).abs().max()
    assert worst <= 0.01, f"a fold is {worst:.2%} away from {EXPECTED_PREVALENCE:.2%}"

    return row_fold, prev


def summarise(df, row_fold):
    rows = []
    for fold in range(N_SPLITS):
        mask = row_fold == fold
        rows.append(
            {
                "fold": fold,
                "split": fold_to_split(fold),
                "patients": int(df.loc[mask, "patient_nbr"].nunique()),
                "rows": int(mask.sum()),
                "positives": int(df.loc[mask, "target"].sum()),
                "prevalence": round(float(df.loc[mask, "target"].mean()), 6),
            }
        )
    return pd.DataFrame(rows)


def main():
    if not COHORT_PATH.exists():
        sys.exit(f"missing cohort: {COHORT_PATH}")

    df, patients = load_patients()
    fold = assign_folds(patients)
    row_fold, _ = check(df, patients, fold)

    assignment = (
        pd.DataFrame({"patient_nbr": patients["patient_nbr"], "fold": fold})
        .sort_values("patient_nbr")
        .reset_index(drop=True)
    )
    assignment.to_csv(SPLIT_PATH, index=False)

    table = summarise(df, row_fold)
    split_totals = (
        table.groupby("split", sort=False)[["patients", "rows", "positives"]]
        .sum()
        .reindex(["train", "val", "test"])
    )
    split_totals["prevalence"] = (
        split_totals["positives"] / split_totals["rows"]
    ).round(6)

    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = {
        "rule": {
            "splitter": "StratifiedGroupKFold",
            "n_splits": N_SPLITS,
            "random_state": RANDOM_SEED,
            "stratified_on": "first-encounter label",
            "test_folds": [0, 1, 2],
            "val_folds": [3, 4, 5],
            "train_folds": list(range(6, 20)),
        },
        "totals": split_totals.reset_index().to_dict(orient="records"),
        "folds": table.to_dict(orient="records"),
    }
    (RESULTS / "split_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"patients: {len(patients):,}   admissions: {len(df):,}")
    print()
    print(split_totals.to_string())
    print()
    print(f"fold sizes (patients): {table['patients'].min()} - {table['patients'].max()}")
    print(
        f"prevalence per fold:  {table['prevalence'].min():.4f} - "
        f"{table['prevalence'].max():.4f}  (cohort {EXPECTED_PREVALENCE:.4f})"
    )
    print()
    print(f"wrote {SPLIT_PATH.relative_to(DATA_PROCESSED.parent.parent)}")
    print("wrote results/split_summary.json")


if __name__ == "__main__":
    main()
