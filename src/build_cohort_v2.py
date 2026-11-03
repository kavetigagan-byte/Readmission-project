
"""
Diabetes Hospital Readmission — Cohort Construction v2
Role A: Data Owner

Run from project root:
    python src/build_cohort_v2.py

Outputs:
    data/processed/cohort_v2.parquet
    data/processed/cohort_v2.sha256

The v1 pipeline and dataset are never overwritten.
"""

import hashlib
from pathlib import Path

import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

RAW = Path("data/raw/diabetic_data.csv")
V1 = Path("data/processed/cohort_v1.parquet")
OUT = Path("data/processed/cohort_v2.parquet")
HASH = Path("data/processed/cohort_v2.sha256")

AGE_MAP = {
    f"[{i}-{i+10})": i + 5
    for i in range(0, 100, 10)
}

EXPIRED_CODES = {"11", "13", "14", "19", "20", "21"}

CONSTANT_DROP = ["examide", "citoglipton"]

RARE_DROP = [
    "acetohexamide",
    "glimepiride-pioglitazone",
    "metformin-pioglitazone",
    "metformin-rosiglitazone",
    "troglitazone",
    "glipizide-metformin",
    "tolbutamide",
    "miglitol",
    "tolazamide",
    "chlorpropamide",
]

EXPLICIT_DROP = ["weight", "payer_code"]

DROP_COLUMNS = CONSTANT_DROP + RARE_DROP + EXPLICIT_DROP

# Exactly 15 model input attributes.
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

# Exactly 5 auxiliary columns.
AUXILIARY_COLUMNS = [
    "encounter_id",
    "patient_nbr",
    "target",
    "is_first_encounter",
    "n_stays_total",
]

FINAL_COLUMNS = MODEL_ATTRIBUTES + AUXILIARY_COLUMNS

EXPECTED = {
    "rows": 99_340,
    "positives": 11_314,
    "patients": 69_987,
    "single_stay": 53_646,
    "repeat_stay": 16_341,
    "repeat_rows": 45_694,
    "max_stays": 40,
    "first_rows": 69_987,
    "first_positives": 6_285,
}


# ============================================================
# 2. DIAGNOSIS GROUPING
# Matches the original v1 implementation exactly.
# ============================================================

def group_diag(code):

    if code in ("?", "", None):
        return "Missing"

    try:
        value = float(code)
    except (ValueError, TypeError):
        return "Other"

    if int(value) == 250:
        return "Diabetes"

    if (390 <= value <= 459) or int(value) == 785:
        return "Circulatory"

    if (460 <= value <= 519) or int(value) == 786:
        return "Respiratory"

    if (520 <= value <= 579) or int(value) == 787:
        return "Digestive"

    if (580 <= value <= 629) or int(value) == 788:
        return "Genitourinary"

    if 140 <= value <= 239:
        return "Neoplasms"

    if 710 <= value <= 739:
        return "Musculoskeletal"

    if 800 <= value <= 999:
        return "Injury"

    return "Other"


# ============================================================
# 3. BUILD V2 DATASET
# ============================================================

def build_cohort():

    # Load original dataset without automatic type conversion.
    if not RAW.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW}")

    df = pd.read_csv(
        RAW,
        dtype=str,
        keep_default_na=False,
        na_values=[]
    )

    print("Raw rows:", len(df))

    # Step 1: Remove invalid gender.
    df = df.loc[
        df["gender"] != "Unknown/Invalid"
    ].copy()

    # Step 2: Remove expired and hospice discharges.
    df = df.loc[
        ~df["discharge_disposition_id"].isin(EXPIRED_CODES)
    ].copy()

    print("Rows after required filters:", len(df))

    # Step 3: Create binary target.
    df["target"] = (
        df["readmitted"] == "<30"
    ).astype("int64")

    df = df.drop(columns=["readmitted"])

    # Step 4: Convert patient identifiers.
    for col in ["encounter_id", "patient_nbr"]:
        df[col] = pd.to_numeric(
            df[col], errors="raise"
        ).astype("int64")

    # Step 5: Count all retained encounters per patient.
    df = df.sort_values(
        ["patient_nbr", "encounter_id"]
    ).reset_index(drop=True)

    df["n_stays_total"] = (
        df.groupby("patient_nbr")["encounter_id"]
        .transform("size")
        .astype("int64")
    )

    # Mark the earliest retained encounter for each patient.
    df["is_first_encounter"] = (
        ~df.duplicated(
            subset=["patient_nbr"],
            keep="first"
        )
    ).astype("int64")

    # Step 6: Validate the documented 14 dropped columns.
    missing = set(DROP_COLUMNS) - set(df.columns)

    assert not missing, (
        f"Missing columns for pruning: {missing}"
    )

    # Two constant columns.
    for col in CONSTANT_DROP:
        assert df[col].nunique(dropna=False) == 1, (
            f"{col} is not constant"
        )

    # Ten low-frequency medication columns.
    for col in RARE_DROP:
        count = int((df[col] != "No").sum())

        assert count < 100, (
            f"{col} has {count} non-'No' values"
        )

    df = df.drop(columns=DROP_COLUMNS)

    print("Dropped columns:", len(DROP_COLUMNS))

    # Step 7: Convert age bands to numeric midpoints.
    df["age"] = df["age"].map(AGE_MAP)

    assert df["age"].notna().all(), (
        "Unmapped age values found"
    )

    df["age"] = df["age"].astype("int64")

    # Step 8: Group primary diagnosis.
    df["diag_1_group"] = df["diag_1"].apply(group_diag)

    df = df.drop(
        columns=["diag_1", "diag_2", "diag_3"]
    )

    # Step 9: Clean race missing indicators.
    df["race"] = df["race"].replace(
        {"?": "Unknown", "": "Unknown"}
    )

    # Keep A1Cresult and insulin source values unchanged.

    # Step 10: Select exactly the required 20 columns.
    missing = set(FINAL_COLUMNS) - set(df.columns)

    assert not missing, (
        f"Missing final columns: {missing}"
    )

    df = df.loc[:, FINAL_COLUMNS].copy()

    # Step 11: Set numeric data types.
    integer_cols = [
        "age",
        "time_in_hospital",
        "number_inpatient",
        "number_emergency",
        "number_outpatient",
        "num_lab_procedures",
        "num_medications",
        "number_diagnoses",
        "admission_type_id",
        "discharge_disposition_id",
        "encounter_id",
        "patient_nbr",
        "target",
        "is_first_encounter",
        "n_stays_total",
    ]

    for col in integer_cols:
        df[col] = pd.to_numeric(
            df[col], errors="raise"
        ).astype("int64")

    # Step 12: Convert categorical columns.
    category_cols = [
        "gender",
        "race",
        "diag_1_group",
        "A1Cresult",
        "insulin",
    ]

    for col in category_cols:
        df[col] = df[col].astype("category")

    # Step 13: Deterministic final ordering.
    df = df.sort_values(
        "encounter_id"
    ).reset_index(drop=True)

    return df


# ============================================================
# 4. V2 ACCEPTANCE VALIDATION
# ============================================================

def validate(df):

    patients = df.groupby(
        "patient_nbr", observed=True
    ).size()

    first = df.loc[
        df["is_first_encounter"] == 1
    ]

    # Final schema and dataset size.
    assert df.shape == (EXPECTED["rows"], 20), (
        f"Unexpected shape: {df.shape}"
    )

    assert list(df.columns) == FINAL_COLUMNS

    # Target checks.
    assert int(df["target"].sum()) == EXPECTED["positives"]
    assert set(df["target"].unique()) == {0, 1}

    # Patient checks.
    assert (
        df["patient_nbr"].nunique()
        == EXPECTED["patients"]
    )

    assert df["encounter_id"].is_unique

    assert int(
        (patients == 1).sum()
    ) == EXPECTED["single_stay"]

    assert int(
        (patients >= 2).sum()
    ) == EXPECTED["repeat_stay"]

    assert int(patients.max()) == EXPECTED["max_stays"]

    # Repeat-patient encounter checks.
    repeat_rows = int(
        (df["n_stays_total"] > 1).sum()
    )

    assert repeat_rows == EXPECTED["repeat_rows"]

    actual = patients.sort_index()

    recorded = (
        df.groupby(
            "patient_nbr", observed=True
        )["n_stays_total"]
        .first()
        .sort_index()
    )

    assert actual.equals(recorded), (
        "n_stays_total does not match actual retained encounters"
    )

    # First-encounter checks.
    assert len(first) == EXPECTED["first_rows"]

    assert int(
        first["target"].sum()
    ) == EXPECTED["first_positives"]

    assert int(
        df["is_first_encounter"].sum()
    ) == EXPECTED["patients"]

    assert set(
        df["is_first_encounter"].unique()
    ) == {0, 1}

    # Required filters.
    assert not (
        df["gender"].astype(str) == "Unknown/Invalid"
    ).any()

    assert not (
        df["discharge_disposition_id"]
        .astype(str)
        .isin(EXPIRED_CODES)
    ).any()

    # Explicit feature/auxiliary separation.
    assert len(MODEL_ATTRIBUTES) == 15
    assert len(AUXILIARY_COLUMNS) == 5
    assert not set(MODEL_ATTRIBUTES).intersection(
        AUXILIARY_COLUMNS
    )

    print("\nAll v2 acceptance checks passed.")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Positive cases:", int(df["target"].sum()))
    print(
        "Positive rate:",
        f"{df['target'].mean() * 100:.4f}%"
    )
    print("Unique patients:", df["patient_nbr"].nunique())
    print(
        "Single-stay patients:",
        int((patients == 1).sum())
    )
    print(
        "Repeat-stay patients:",
        int((patients >= 2).sum())
    )
    print("Repeat-patient rows:", repeat_rows)
    print("Maximum stays:", int(patients.max()))
    print("First-encounter rows:", len(first))
    print(
        "First-encounter positives:",
        int(first["target"].sum())
    )


# ============================================================
# 5. COMPARE V2 FIRST ENCOUNTERS AGAINST V1
# ============================================================

def compare_v1(df):

    if not V1.exists():
        raise FileNotFoundError(
            "cohort_v1.parquet is required for comparison"
        )

    old = pd.read_parquet(V1)

    first = df.loc[
        df["is_first_encounter"] == 1
    ].copy()

    common = [
        col for col in MODEL_ATTRIBUTES + [
            "target",
            "encounter_id",
            "patient_nbr"
        ]
        if col in old.columns and col in first.columns
    ]

    old = (
        old[common]
        .sort_values("encounter_id")
        .reset_index(drop=True)
    )

    first = (
        first[common]
        .sort_values("encounter_id")
        .reset_index(drop=True)
    )

    assert len(old) == len(first), (
        "v1 and v2 first-encounter row counts differ"
    )

    errors = []

    for col in common:

        a = old[col].astype("string").fillna("<NA>")
        b = first[col].astype("string").fillna("<NA>")

        mismatch = a != b

        if mismatch.any():

            errors.append(col)

            print(
                f"\nMismatch in {col}: "
                f"{int(mismatch.sum())} rows"
            )

            examples = pd.DataFrame({
                "encounter_id": old.loc[
                    mismatch, "encounter_id"
                ],
                "v1_value": a[mismatch],
                "v2_value": b[mismatch],
            })

            print(
                examples.head(10).to_string(index=False)
            )

    assert not errors, (
        f"v1 comparison failed in: {errors}"
    )

    print(
        f"\nv1 first-encounter comparison passed "
        f"for {len(common)} columns."
    )


# ============================================================
# 6. SAVE V2 PARQUET AND SHA256
# ============================================================

def save_cohort(df):

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_parquet(
        OUT,
        index=False
    )

    digest = hashlib.sha256()

    with OUT.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    checksum = digest.hexdigest()

    HASH.write_text(
        f"{checksum}  {OUT.name}\n",
        encoding="utf-8"
    )

    print("\nSaved Parquet:", OUT)
    print("Saved SHA256:", HASH)
    print("Fingerprint:", checksum)


# ============================================================
# 7. MAIN
# ============================================================

if __name__ == "__main__":

    cohort = build_cohort()

    validate(cohort)

    compare_v1(cohort)

    save_cohort(cohort)

    print("\nCohort v2 pipeline completed successfully.")

    print("\nROLE C FEATURE GUIDANCE")
    print("X = df[MODEL_ATTRIBUTES]")
    print("y = df['target']")
    print(
        "Never use encounter_id, patient_nbr, "
        "is_first_encounter or n_stays_total as predictors."
    )
    print(
        "Optional sample weighting: "
        "1 / n_stays_total (subject to team approval)."
    )
