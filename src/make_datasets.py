"""Materialise the train / val / test CSVs from the cohort and the split.

    python src/make_datasets.py

Reads
-----
data/processed/cohort/cohort_v2.parquet
data/processed/split/split_assignment.csv

Writes
------
data/processed/train/train.csv                  natural prevalence, 11.39%
data/processed/train/train_undersampled.csv     35% positive, training only
data/processed/val/val.csv                      natural prevalence
data/processed/test/test.csv                    natural prevalence

The three split files are just the cohort with the patient's fold applied, so
they can always be rebuilt. They carry the same caveat as the cohort CSV:
read them with keep_default_na=False, or pandas turns A1Cresult = "None" into NaN.
"""

import sys

import numpy as np
import pandas as pd

from config import (
    COHORT_PATH,
    SPLIT_PATH,
    TEST_CSV_PATH,
    TEST_FOLDS,
    TRAIN_BALANCED_CSV_PATH,
    TRAIN_CSV_PATH,
    TRAIN_FOLDS,
    VAL_CSV_PATH,
    VAL_FOLDS,
)
from undersample import undersample_train


def main():
    if not COHORT_PATH.exists() or not SPLIT_PATH.exists():
        sys.exit("run src/split.py first")

    cohort = pd.read_parquet(COHORT_PATH)
    assignment = pd.read_csv(SPLIT_PATH)

    df = cohort.merge(
        assignment, on="patient_nbr", how="left", validate="many_to_one"
    )
    if df["fold"].isna().any():
        sys.exit("some admissions have no fold - rebuild the split")

    pieces = {
        "train": TRAIN_FOLDS,
        "val": VAL_FOLDS,
        "test": TEST_FOLDS,
    }
    paths = {
        "train": TRAIN_CSV_PATH,
        "val": VAL_CSV_PATH,
        "test": TEST_CSV_PATH,
    }

    written = {}
    for name, folds in pieces.items():
        part = df[df["fold"].isin(folds)].drop(columns="fold")
        part.to_csv(paths[name], index=False)
        written[name] = part

    train = df[df["fold"].isin(TRAIN_FOLDS)]
    balanced, r = undersample_train(train, train["fold"])
    balanced.drop(columns="fold").to_csv(TRAIN_BALANCED_CSV_PATH, index=False)

    # checks
    total = sum(len(p) for p in written.values())
    assert total == 99_340, f"splits cover {total:,} rows, expected 99,340"
    assert len(balanced) < len(written["train"])
    assert written["train"]["target"].sum() == balanced["target"].sum()
    for name, part in written.items():
        assert abs(part["target"].mean() - 0.1139) < 0.01, name

    print(f"{'file':<44}{'rows':>8}{'positives':>11}{'rate':>9}")
    for name in ("train", "val", "test"):
        p = written[name]
        print(
            f"{str(paths[name]).split('processed/')[-1]:<44}"
            f"{len(p):>8,}{int(p.target.sum()):>11,}{p.target.mean():>9.2%}"
        )
    print(
        f"{str(TRAIN_BALANCED_CSV_PATH).split('processed/')[-1]:<44}"
        f"{len(balanced):>8,}{int(balanced.target.sum()):>11,}"
        f"{balanced.target.mean():>9.2%}"
    )
    print()
    print(f"r = {r:.6f}   log(r) = {np.log(r):.6f}")
    print("all four files written")


if __name__ == "__main__":
    main()
