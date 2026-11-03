# Data Cleaning and Cohort Construction Log

**Project:** Diabetes Hospital Readmission Prediction  
**Role:** Role A — Data Owner  
**Dataset:** Diabetic Hospital Readmission Dataset

---

# 1. Raw Dataset Overview

The original dataset contains 101,766 hospital encounters and 50 columns.

Raw dataset path:

`data/raw/diabetic_data.csv`

The dataset was loaded using:

```python
pd.read_csv(
    RAW,
    dtype=str,
    keep_default_na=False,
    na_values=[]
)
```

Reading all values as strings prevents unwanted automatic type conversion, particularly for identifiers and diagnosis codes.

# 2. Cohort Construction — Version 1

Version 1 retains only the first eligible encounter for each patient.

## 2.1 Row Filtering

| Step | Cleaning operation | Rows before | Rows after | Removed |
|---|---|---:|---:|---:|
| 1 | Raw dataset | — | 101,766 | — |
| 2 | Remove invalid gender | 101,766 | 101,763 | 3 |
| 3 | Remove expired/hospice discharges | 101,763 | 99,340 | 2,423 |
| 4 | Keep first encounter per patient | 99,340 | 69,987 | 29,353 |

### Gender filtering

Records with `gender = Unknown/Invalid` were removed because they contain invalid demographic information.

### Expired and hospice filtering

The following discharge disposition codes were excluded:

`11, 13, 14, 19, 20, 21`

These represent expired or hospice-related discharge outcomes.

These encounters were excluded because a patient who has died cannot experience a subsequent readmission.

### First encounter selection

Only the earliest retained encounter for each patient was selected.

The smallest `encounter_id` was used as the chronological proxy because the dataset does not provide an admission date.

The resulting v1 cohort contains 69,987 unique patients.

## 2.2 Version 1 Target Distribution

| Target | Meaning | Records |
|---|---|---:|
| 1 | Readmitted within 30 days | 6,285 |
| 0 | Not readmitted within 30 days | 63,702 |
| **Total** | | **69,987** |

Positive rate: **8.98%** (approximately).

# 3. Cohort Construction — Version 2

Version 2 changes the encounter-selection strategy.

Instead of keeping only the first encounter, it retains every eligible encounter after the gender and expired/hospice filters.

The purpose is to preserve repeat hospital visits for patient readmission analysis.

## 3.1 Row Filtering

| Step | Cleaning operation | Rows before | Rows after | Removed |
|---|---|---:|---:|---:|
| 1 | Raw dataset | — | 101,766 | — |
| 2 | Remove invalid gender | 101,766 | 101,763 | 3 |
| 3 | Remove expired/hospice discharges | 101,763 | 99,340 | 2,423 |
| 4 | First-encounter filtering | Not applied | 99,340 | 0 |

**Important:** Version 2 does not remove repeat encounters.

## 3.2 Target Construction

The binary target was created from the original `readmitted` column.

```python
df["target"] = (
    df["readmitted"] == "<30"
).astype("int64")
```

| Original readmission value | Target |
|---|---:|
| `<30` | 1 |
| `NO` | 0 |
| `>30` | 0 |

The original `readmitted` column was removed after target creation.

## 3.3 Patient Encounter History

Two additional columns were created.

| Column | Description |
|---|---|
| `n_stays_total` | Total number of retained encounters for each patient |
| `is_first_encounter` | 1 for the earliest retained encounter and 0 for subsequent encounters |

The earliest encounter is identified using the smallest `encounter_id` for each patient.

These columns allow repeat-encounter analysis while preserving the ability to reproduce the v1 first-encounter cohort.

## 3.4 Patient-Level Statistics

| Metric | Result |
|---|---:|
| Total encounters | 99,340 |
| Unique patients | 69,987 |
| Patients with one stay | 53,646 |
| Patients with two or more stays | 16,341 |
| Rows belonging to repeat-stay patients | 45,694 |
| Maximum stays per patient | 40 |
| First-encounter subset | 69,987 |

Approximately 23.35% of patients have multiple retained encounters.

Approximately 46% of all encounter rows belong to patients with repeat stays.

## 3.5 Version 2 Target Distribution

| Target | Meaning | Records |
|---|---|---:|
| 1 | Readmitted within 30 days | 11,314 |
| 0 | Not readmitted within 30 days | 88,026 |
| **Total** | | **99,340** |

Positive rate: **11.3892%**.

## 3.6 First-Encounter Validation

The v2 dataset was filtered using:

`is_first_encounter == 1`

The resulting subset was compared with the preserved v1 Parquet dataset.

| Comparison | Result |
|---|---:|
| First-encounter rows | 69,987 |
| First-encounter positives | 6,285 |
| Common columns compared | 18 |
| Comparison result | Passed |

The comparison confirms that the first-encounter subset matches v1 across the 18 common columns.

# 4. Column Removal Policy — Version 2

A total of 14 columns were removed according to the documented v2 pruning policy.

| Category | Columns | Count |
|---|---|---:|
| Constant columns | `examide`, `citoglipton` | 2 |
| Low-frequency medication columns | `acetohexamide`, `glimepiride-pioglitazone`, `metformin-pioglitazone`, `metformin-rosiglitazone`, `troglitazone`, `glipizide-metformin`, `tolbutamide`, `miglitol`, `tolazamide`, `chlorpropamide` | 10 |
| Explicitly removed | `weight`, `payer_code` | 2 |
| **Total** | | **14** |

The pipeline verifies that the constant columns have only one distinct value and that each listed low-frequency medication column has fewer than 100 non-`No` records before removal.

The final schema does not include `medical_specialty`.

# 5. Age Transformation

The original age intervals were converted into numeric midpoints.

| Original age band | Numeric value |
|---|---:|
| [0-10) | 5 |
| [10-20) | 15 |
| [20-30) | 25 |
| [30-40) | 35 |
| [40-50) | 45 |
| [50-60) | 55 |
| [60-70) | 65 |
| [70-80) | 75 |
| [80-90) | 85 |
| [90-100) | 95 |

This transformation provides a numeric representation of age for downstream analysis.

# 6. Primary Diagnosis Grouping

The original `diag_1` ICD-9 diagnosis code was grouped into broader clinical categories.

The resulting `diag_1_group` column contains:

| Diagnosis group | Records |
|---|---:|
| Circulatory | 29,680 |
| Other | 17,793 |
| Respiratory | 13,934 |
| Digestive | 9,333 |
| Diabetes | 8,661 |
| Injury | 6,851 |
| Genitourinary | 5,002 |
| Musculoskeletal | 4,935 |
| Neoplasms | 3,131 |
| Missing | 20 |
| **Total** | **99,340** |

The original `diag_1`, `diag_2`, and `diag_3` columns were removed from the final v2 dataset.

The 20 missing primary diagnosis codes are represented by the explicit `Missing` category.

# 7. Missing Value Analysis

Missing-value checks were performed on the final v2 Parquet dataset.

| Check | Count |
|---|---:|
| Literal question marks (`?`) | 0 |
| Empty strings | 0 |
| Actual null values | 0 |

Missing source indicators in `race` were replaced with `Unknown`.

The original categories of `A1Cresult` and `insulin` were retained.

The following are explicit categories rather than null values:

- `race`: Unknown
- `A1Cresult`: None
- `diag_1_group`: Missing

Therefore, zero detected null values in the processed dataset does not mean the original raw dataset contained no missing information.

# 8. Final Version 2 Dataset Schema

The final v2 dataset contains exactly 20 columns.

| Category | Number of columns |
|---|---:|
| Model attributes | 15 |
| Auxiliary columns | 5 |
| **Total** | **20** |

The dataset is saved as:

`data/processed/cohort_v2.parquet`

Its checksum is saved separately:

`data/processed/cohort_v2.sha256`

# 9. Data Leakage Prevention and Role C Handover

The 15 approved model attributes are defined in `MODEL_ATTRIBUTES`.

The target is:

`target`

The following auxiliary columns must not be used as predictive features:

- `encounter_id`
- `patient_nbr`
- `is_first_encounter`
- `n_stays_total`

**Important leakage warning:** `n_stays_total` uses all retained encounters for a patient, including future encounters relative to an individual stay. It must not be included in the predictive feature matrix.

If approved by the team, it may be used only for sample weighting:

```python
sample_weight = 1.0 / df["n_stays_total"]
```

Role C should construct the feature matrix explicitly:

```python
X = df[MODEL_ATTRIBUTES]
y = df["target"]
```

# 10. Final Validation Summary

The v2 pipeline completed all acceptance checks.

| Validation | Expected | Actual | Status |
|---|---:|---:|---|
| Dataset rows | 99,340 | 99,340 | Passed |
| Dataset columns | 20 | 20 | Passed |
| Positive cases | 11,314 | 11,314 | Passed |
| Unique patients | 69,987 | 69,987 | Passed |
| Single-stay patients | 53,646 | 53,646 | Passed |
| Repeat-stay patients | 16,341 | 16,341 | Passed |
| Repeat-patient rows | 45,694 | 45,694 | Passed |
| Maximum stays | 40 | 40 | Passed |
| First-encounter rows | 69,987 | 69,987 | Passed |
| First-encounter positives | 6,285 | 6,285 | Passed |
| v1 comparison | 18 common columns | 18 passed | Passed |

## 11. Conclusion

The v2 pipeline successfully preserves all 99,340 eligible hospital encounters and adds patient-level encounter history information.

The final dataset contains 20 columns, including 15 model attributes and 5 auxiliary columns. The first-encounter subset was verified against the preserved v1 dataset.

The pipeline also generates a separate Parquet file and SHA256 checksum, ensuring that v1 remains unchanged.

The resulting dataset is ready for the next stages of exploratory data analysis and model development, subject to the documented feature leakage restrictions.