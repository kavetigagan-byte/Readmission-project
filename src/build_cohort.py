"""
Data Cleaning and Cohort Construction Pipeline (Phases 1 & 2)
Author: Role A (Data Owner)

Usage:
    python src/build_cohort.py
"""

import csv
import hashlib
import os
import pandas as pd

RAW_DATA_PATH = "data/raw/diabetic_data.csv"
IDS_MAPPING_PATH = "data/raw/IDS_mapping.csv"
OUTPUT_PARQUET_PATH = "data/processed/cohort_v1.parquet"
OUTPUT_HASH_PATH = "data/processed/cohort_v1.sha256"

AGE_MAPPING = {
    "[0-10)": 5,
    "[10-20)": 15,
    "[20-30)": 25,
    "[30-40)": 35,
    "[40-50)": 45,
    "[50-60)": 55,
    "[60-70)": 65,
    "[70-80)": 75,
    "[80-90)": 85,
    "[90-100)": 95,
}

EXPIRED_HOSPICE_CODES = {"11", "13", "14", "19", "20", "21"}

INT_COLUMNS = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "age",
    "target",
]


def load_raw(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Reads raw CSV strictly as string to prevent silent type conversion bugs."""
    print(f"Loading raw data from: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])
    print(f"Raw records loaded: {len(df)}")
    return df


def drop_invalid_gender(df: pd.DataFrame) -> pd.DataFrame:
    """Step 1: Drop invalid gender values (Unknown/Invalid)."""
    before = len(df)
    df_clean = df[df["gender"] != "Unknown/Invalid"].copy()
    print(f"after gender filter: {len(df_clean)} (removed {before - len(df_clean)})")
    return df_clean


def drop_dead_and_hospice(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2: Drop expired and hospice discharge dispositions."""
    before = len(df)
    df_clean = df[~df["discharge_disposition_id"].isin(EXPIRED_HOSPICE_CODES)].copy()
    print(f"after dead/hospice: {len(df_clean)} (removed {before - len(df_clean)})")
    return df_clean


def keep_first_encounter(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3: Keep earliest encounter per patient using smallest encounter_id proxy."""
    before = len(df)
    df_sorted = df.copy()
    df_sorted["_eid"] = df_sorted["encounter_id"].astype("int64")
    df_clean = (
        df_sorted.sort_values("_eid")
        .drop_duplicates("patient_nbr", keep="first")
        .drop(columns="_eid")
    )
    print(f"after first encounter per patient: {len(df_clean)} (removed {before - len(df_clean)})")
    return df_clean


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Step 4: Create binary readmission target (<30 days = 1, otherwise = 0)."""
    df_out = df.copy()
    df_out["target"] = (df_out["readmitted"] == "<30").astype(int)
    positives = df_out["target"].sum()
    pct = (positives / len(df_out)) * 100
    print(f"after target build: {len(df_out)} total, {positives} positives ({pct:.2f}%)")
    return df_out.drop(columns="readmitted")


def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Step 5: Drop high-missingness (>40%), constant, and near-constant (<100 non-No) columns."""
    df_out = df.copy()

    explicit_drops = ["weight", "payer_code"]
    cols_to_drop = [c for c in explicit_drops if c in df_out.columns]

    # Detect constant columns after cohort filtering
    for col in df_out.columns:
        if col in cols_to_drop:
            continue
        if df_out[col].nunique() <= 1:
            cols_to_drop.append(col)

    # Detect medication columns with < 100 non-'No' values
    all_med_candidates = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "examide", "citoglipton", "insulin", "glyburide-metformin",
        "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
        "metformin-pioglitazone",
    ]

    for med in all_med_candidates:
        if med in df_out.columns and med not in cols_to_drop:
            non_no_count = (df_out[med] != "No").sum()
            if non_no_count < 100:
                cols_to_drop.append(med)

    print(f"after column pruning: dropped {len(cols_to_drop)} columns: {sorted(cols_to_drop)}")
    return df_out.drop(columns=cols_to_drop)


def fix_age(df: pd.DataFrame) -> pd.DataFrame:
    """Step 6: Convert age bands into numeric midpoints."""
    df_out = df.copy()
    df_out["age"] = df_out["age"].map(AGE_MAPPING)
    print("after age midpoint recoding: age transformed to integer midpoint")
    return df_out


def group_diag(code: str) -> str:
    """Step 7 helper: Categorize raw ICD-9 codes into 9 clinical categories + Missing."""
    if code in ("?", "", None):
        return "Missing"
    try:
        v = float(code)
    except ValueError:
        return "Other"

    if int(v) == 250:
        return "Diabetes"
    if (390 <= v <= 459) or (int(v) == 785):
        return "Circulatory"
    if (460 <= v <= 519) or (int(v) == 786):
        return "Respiratory"
    if (520 <= v <= 579) or (int(v) == 787):
        return "Digestive"
    if (580 <= v <= 629) or (int(v) == 788):
        return "Genitourinary"
    if 140 <= v <= 239:
        return "Neoplasms"
    if 710 <= v <= 739:
        return "Musculoskeletal"
    if 800 <= v <= 999:
        return "Injury"
    return "Other"


def group_diagnoses(df: pd.DataFrame) -> pd.DataFrame:
    """Step 7: Group diag_1, diag_2, diag_3 into categories and drop originals."""
    df_out = df.copy()
    for col in ["diag_1", "diag_2", "diag_3"]:
        df_out[f"{col}_group"] = df_out[col].apply(group_diag)
    df_out = df_out.drop(columns=["diag_1", "diag_2", "diag_3"])
    print("after diagnosis grouping: diag_1/2/3 mapped to clinical groups and dropped")
    return df_out


def parse_ids_mapping_csv(mapping_path: str = IDS_MAPPING_PATH) -> dict:
    """Correction 1: Parses IDS_mapping.csv using standard csv.reader."""
    lookups = {
        "admission_type_id": {},
        "discharge_disposition_id": {},
        "admission_source_id": {},
    }

    if not os.path.exists(mapping_path):
        print(f"Warning: {mapping_path} not found. Skipping ID descriptions.")
        return lookups

    current_key = None
    with open(mapping_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or not any(field.strip() for field in row):
                continue

            header_candidate = row[0].strip().lower()
            if "admission_type_id" in header_candidate:
                current_key = "admission_type_id"
                continue
            elif "discharge_disposition_id" in header_candidate:
                current_key = "discharge_disposition_id"
                continue
            elif "admission_source_id" in header_candidate:
                current_key = "admission_source_id"
                continue

            if current_key and len(row) >= 2:
                code_id = row[0].strip()
                desc = row[1].strip()
                if code_id.isdigit():
                    lookups[current_key][code_id] = desc

    return lookups


def add_admission_descriptions(df: pd.DataFrame, mapping_path: str = IDS_MAPPING_PATH) -> pd.DataFrame:
    """Adds *_desc columns from IDS_mapping.csv for charts."""
    lookups = parse_ids_mapping_csv(mapping_path)
    df_out = df.copy()

    for col in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        desc_col = f"{col}_desc"
        lookup = lookups.get(col, {})
        df_out[desc_col] = df_out[col].astype(str).map(lookup).fillna("Unknown")

    print("after admission descriptions: added *_desc columns via csv.reader")
    return df_out


def clean_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Step 8: Handle missing indicators and collapse rare specialties."""
    df_out = df.copy()

    if "race" in df_out.columns:
        df_out["race"] = df_out["race"].replace("?", "Unknown")

    if "medical_specialty" in df_out.columns:
        df_out["medical_specialty"] = df_out["medical_specialty"].replace("?", "Unknown")
        counts = df_out["medical_specialty"].value_counts()
        rare = counts[counts < 50].index
        df_out["medical_specialty"] = df_out["medical_specialty"].replace(rare, "Other")

    print("after categorical cleanup: race and medical_specialty normalized")
    return df_out


def set_final_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Step 9: Cast columns to canonical types (int64 for keys/metrics, category for text)."""
    df_out = df.copy()

    for key in ["encounter_id", "patient_nbr"]:
        if key in df_out.columns:
            df_out[key] = df_out[key].astype("int64")

    for col in INT_COLUMNS:
        if col in df_out.columns:
            df_out[col] = df_out[col].astype("int64")

    object_cols = df_out.select_dtypes(include=["object"]).columns
    for col in object_cols:
        df_out[col] = df_out[col].astype("category")

    print("after final dtypes: applied int64 to metrics/keys and category to text")
    return df_out


def summarise(df: pd.DataFrame) -> None:
    """Correction 2: Prints exact cohort metrics formatted for docs/cleaning_log.md."""
    print("\n" + "=" * 65)
    print("COHORT CLEANING SUMMARY (Paste into docs/cleaning_log.md)")
    print("=" * 65)
    print(f"Final Total Rows:             {len(df):,}")
    print(f"Unique Patients:              {df['patient_nbr'].nunique():,}")
    print(f"Target Positives (<30 days):  {df['target'].sum():,} ({df['target'].mean() * 100:.2f}%)")
    print(f"Target Negatives (0):         {(df['target'] == 0).sum():,} ({(df['target'] == 0).mean() * 100:.2f}%)")
    print(f"Total Columns:                {len(df.columns)}")

    print("\nRace Breakdown:")
    for race, count in df["race"].value_counts().items():
        print(f"  - {race}: {count:,}")

    print("\nPrimary Diagnosis Group (diag_1_group):")
    for diag, count in df["diag_1_group"].value_counts().items():
        print(f"  - {diag}: {count:,}")

    print("\nAge Midpoint Distribution:")
    for age, count in df["age"].value_counts().sort_index().items():
        print(f"  - Age {age}: {count:,}")
    print("=" * 65 + "\n")


def validate_expected_statistics(df: pd.DataFrame) -> None:
    """Correction 3: Strengthened invariant validation against expected_stats.txt."""
    print("Executing comprehensive validation checks against expected_stats.txt...")

    # 1. Total volume and patient cardinality
    assert len(df) == 69_987, f"Row count error: got {len(df)}, expected 69,987"
    assert df["patient_nbr"].is_unique, "Duplicate patient_nbr detected: not 1 row per patient"

    # 2. Target validation
    assert df["target"].sum() == 6_285, f"Positive target count error: got {df['target'].sum()}, expected 6,285"
    assert (df["target"] == 0).sum() == 63_702, "Negative target count mismatch (expected 63,702)"
    assert set(df["target"].unique()) == {0, 1}, "Target column contains non-binary values"

    # 3. Discharges (ensure all expired and hospice codes are removed)
    forbidden_discharges = {"11", "13", "14", "19", "20", "21"}
    present_discharges = set(df["discharge_disposition_id"].astype(str).unique())
    inter = present_discharges.intersection(forbidden_discharges)
    assert not inter, f"Found forbidden discharge codes in cohort: {inter}"

    # 4. Gender
    gender_counts = df["gender"].value_counts().to_dict()
    assert set(gender_counts.keys()) == {"Female", "Male"}, f"Invalid genders present: {gender_counts.keys()}"
    assert "Unknown/Invalid" not in gender_counts, "Unknown/Invalid gender was not dropped"

    # 5. Exact Age Midpoint Distribution
    expected_age_counts = {
        5: 153, 15: 534, 25: 1121, 35: 2692, 45: 6828,
        55: 12351, 65: 15688, 75: 17749, 85: 11110, 95: 1761,
    }
    actual_age_counts = df["age"].value_counts().to_dict()
    for age_mid, exp_count in expected_age_counts.items():
        act_count = actual_age_counts.get(age_mid, 0)
        assert act_count == exp_count, f"Age {age_mid} count mismatch: got {act_count}, expected {exp_count}"

    # 6. Exact diag_1_group Distribution
    expected_diag_1 = {
        "Circulatory": 21_389,
        "Other": 12_124,
        "Respiratory": 9_491,
        "Digestive": 6_488,
        "Diabetes": 5_748,
        "Injury": 4_694,
        "Musculoskeletal": 4_064,
        "Genitourinary": 3_441,
        "Neoplasms": 2_538,
        "Missing": 10,
    }
    actual_diag_1 = df["diag_1_group"].value_counts().to_dict()
    for category, exp_cnt in expected_diag_1.items():
        act_cnt = actual_diag_1.get(category, 0)
        assert act_cnt == exp_cnt, f"diag_1_group '{category}' count mismatch: got {act_cnt}, expected {exp_cnt}"

    # 7. Exact Race Distribution
    expected_race = {
        "Caucasian": 52_305,
        "AfricanAmerican": 12_627,
        "Unknown": 1_917,
        "Hispanic": 1_501,
        "Other": 1_149,
        "Asian": 488,
    }
    actual_race = df["race"].value_counts().to_dict()
    for r_cat, exp_r_cnt in expected_race.items():
        act_r_cnt = actual_race.get(r_cat, 0)
        assert act_r_cnt == exp_r_cnt, f"Race '{r_cat}' count mismatch: got {act_r_cnt}, expected {exp_r_cnt}"

    # 8. Specialty Category Levels & Missing Count
    assert df["medical_specialty"].nunique() <= 35, f"Specialty levels not collapsed: {df['medical_specialty'].nunique()}"
    assert (df["medical_specialty"] == "Unknown").sum() == 33_652, "Specialty Unknown count mismatch (expected 33,652)"

    # 9. Verify All 12 Dropped Columns Are Absent
    dropped_cols = [
        "weight", "payer_code", "examide", "citoglipton", "glimepiride-pioglitazone",
        "acetohexamide", "metformin-pioglitazone", "metformin-rosiglitazone",
        "troglitazone", "glipizide-metformin", "tolbutamide", "miglitol",
        "tolazamide", "chlorpropamide",
    ]
    for col in dropped_cols:
        assert col not in df.columns, f"Expected dropped column still present: {col}"

    # 10. Check that raw missing symbol '?' does not exist anywhere in textual data
    for col in df.select_dtypes(include=["object", "category"]).columns:
        assert not (df[col].astype(str) == "?").any(), f"Column {col} still contains unhandled '?' values"

    print("✔ All comprehensive validation checks passed with 100% fidelity.")


def save_and_fingerprint(df: pd.DataFrame, out_path: str = OUTPUT_PARQUET_PATH, hash_path: str = OUTPUT_HASH_PATH) -> None:
    """Step 10: Sort deterministically, write Parquet, and compute SHA256."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df_sorted = df.sort_values("encounter_id").reset_index(drop=True)
    df_sorted.to_parquet(out_path, index=False)

    hasher = hashlib.sha256()
    with open(out_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    file_hash = hasher.hexdigest()

    with open(hash_path, "w") as f:
        f.write(f"{file_hash}  {os.path.basename(out_path)}\n")

    print(f"Saved parquet: {out_path}")
    print(f"SHA256 Fingerprint: {file_hash}")


def main():
    df = load_raw()
    df = drop_invalid_gender(df)
    df = drop_dead_and_hospice(df)
    df = keep_first_encounter(df)
    df = build_target(df)
    df = drop_useless_columns(df)
    df = fix_age(df)
    df = group_diagnoses(df)
    df = add_admission_descriptions(df)
    df = clean_categories(df)
    df = set_final_dtypes(df)
    summarise(df)
    validate_expected_statistics(df)
    save_and_fingerprint(df)
    print("Cohort pipeline completed successfully.")


if __name__ == "__main__":
    main()