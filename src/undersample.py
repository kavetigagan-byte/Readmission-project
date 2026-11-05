"""Undersample the TRAINING set only.

    python src/undersample.py        # dry run on the real data, writes a report

Rule (docs/experiment-matrix-v2.md §5)
--------------------------------------
Keep every positive row. Sample negative rows without replacement so that
    n_neg = round(ratio * n_pos)          ratio = 1.857  ->  35% positive
    r     = n_neg_kept / n_neg_available

`r` is the sampling fraction. The model's probabilities are shifted by it and
must be corrected before any probability, Brier score or calibration curve is
reported:

    p_true = sigmoid( logit(p_model) + log(r) )

Never pass validation or test rows to this function. It raises if you do.
"""

import json
import sys

import numpy as np
import pandas as pd

from config import (
    COHORT_PATH,
    RANDOM_SEED,
    RESULTS,
    SPLIT_PATH,
    TRAIN_FOLDS,
    UNDERSAMPLE_RATIO,
    fold_to_split,
)


def undersample_train(
    df,
    fold,
    ratio=UNDERSAMPLE_RATIO,
    seed=RANDOM_SEED,
    train_folds=TRAIN_FOLDS,
):
    """Return (df_balanced, r) using training rows only.

    df    : training rows only (a DataFrame with `target` and `encounter_id`)
    fold  : fold number for each row of df, aligned by index or positional
    ratio : negatives kept per positive
    seed  : sampling seed

    Raises AssertionError if any row is outside the training folds.
    """
    if "target" not in df.columns:
        raise ValueError("df must contain a 'target' column")

    fold = _as_series(fold, df.index)
    outside = ~fold.isin(train_folds)
    if outside.any():
        bad = sorted(fold[outside].unique().tolist())[:10]
        raise AssertionError(
            f"undersample_train received {int(outside.sum()):,} rows outside the "
            f"training folds (folds seen: {bad}). "
            "Validation and test rows must never be undersampled."
        )

    pos = df[df["target"] == 1]
    neg_available = df[df["target"] == 0]

    if len(pos) == 0:
        raise ValueError("no positive rows in the training frame")

    n_neg = int(round(ratio * len(pos)))
    if n_neg > len(neg_available):
        raise ValueError(
            f"ratio {ratio} needs {n_neg:,} negatives but only "
            f"{len(neg_available):,} are available"
        )

    neg = neg_available.sample(n=n_neg, random_state=seed, replace=False)
    r = n_neg / len(neg_available)

    out = (
        pd.concat([pos, neg])
        .sort_values("encounter_id")
        .reset_index(drop=True)
    )

    assert len(out) == len(pos) + n_neg
    assert out["target"].sum() == len(pos), "positives were lost"
    assert not out["encounter_id"].duplicated().any(), "duplicate rows"

    return out, r


def _as_series(fold, index):
    if isinstance(fold, pd.Series):
        if not fold.index.equals(index):
            fold = fold.reindex(index)
        if fold.isna().any():
            raise AssertionError("fold is missing for some rows")
        return fold
    values = np.asarray(fold)
    if len(values) != len(index):
        raise AssertionError(
            f"fold has {len(values)} entries for {len(index)} rows"
        )
    return pd.Series(values, index=index)


def build_report(df, out, r, ratio, seed):
    """Everything needed to reproduce and to correct the probabilities."""
    n_pos = int(out["target"].sum())
    n_neg = int((out["target"] == 0).sum())
    return {
        "ratio": ratio,
        "seed": seed,
        "n_pos": n_pos,
        "n_neg_kept": n_neg,
        "n_neg_available": int(round(n_neg / r)) if r else 0,
        "r": r,
        "log_r": float(np.log(r)) if r else None,
        "train_rows_before": int(len(df)),
        "train_rows_after": int(len(out)),
        "positive_rate_after": n_pos / len(out),
    }


def main():
    if not COHORT_PATH.exists() or not SPLIT_PATH.exists():
        sys.exit("run src/split.py first")

    cohort = pd.read_parquet(
        COHORT_PATH, columns=["encounter_id", "patient_nbr", "target"]
    )
    assignment = pd.read_csv(SPLIT_PATH)
    df = cohort.merge(assignment, on="patient_nbr", how="left", validate="many_to_one")
    if df["fold"].isna().any():
        sys.exit("some admissions have no fold - rebuild the split")

    train = df[df["fold"].isin(TRAIN_FOLDS)].reset_index(drop=True)
    out, r = undersample_train(train, train["fold"])

    report = build_report(train, out, r, UNDERSAMPLE_RATIO, RANDOM_SEED)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "undersample.json").write_text(json.dumps(report, indent=2))

    print(f"train rows before : {len(train):,}  ({int(train.target.sum()):,} positive)")
    print(f"train rows after  : {len(out):,}  ({report['n_pos']:,} positive)")
    print(f"negatives kept    : {report['n_neg_kept']:,} of {report['n_neg_available']:,}")
    print(f"positive rate     : {report['positive_rate_after']:.2%}")
    print(f"r                 : {r:.6f}   log(r) = {report['log_r']:.6f}")
    print("wrote results/undersample.json")


if __name__ == "__main__":
    main()
