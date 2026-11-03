# Data Dictionary — Diabetes Hospital Readmission Dataset (v2)

## 1. Dataset Overview

The v2 dataset contains 99,340 hospital encounters belonging to 69,987 unique patients. Unlike v1, which retained only the first encounter per patient, v2 preserves every encounter after the required gender and discharge-disposition filters.

The dataset contains exactly 20 columns:
- 15 model attributes
- 5 auxiliary columns

**Dataset file:** `data/processed/cohort_v2.parquet`

**Checksum file:** `data/processed/cohort_v2.sha256`

## 2. Dataset Summary

| Metric | Value |
|---|---:|
| Total encounters | 99,340 |
| Unique patients | 69,987 |
| Positive readmissions | 11,314 |
| Negative readmissions | 88,026 |
| Positive rate | 11.3892% |
| Patients with one stay | 53,646 |
| Patients with multiple stays | 16,341 |
| Rows from repeat-stay patients | 45,694 |
| Maximum stays per patient | 40 |
| First-encounter records | 69,987 |
| First-encounter positives | 6,285 |

## 3. Model Attributes (15 Columns)

These columns are the permitted model input attributes for Role C.

| Column | Data Type | Description |
|---|---|---|
| age | int64 | Patient age represented by the midpoint of the original age band. |
| gender | category | Patient gender. |
| race | category | Patient race; missing source markers are represented as Unknown. |
| time_in_hospital | int64 | Number of days the patient stayed in the hospital. |
| number_inpatient | int64 | Number of inpatient visits recorded in the preceding period. |
| number_emergency | int64 | Number of emergency visits recorded in the preceding period. |
| number_outpatient | int64 | Number of outpatient visits recorded in the preceding period. |
| num_lab_procedures | int64 | Number of laboratory procedures during the encounter. |
| num_medications | int64 | Number of medications administered or prescribed. |
| number_diagnoses | int64 | Number of diagnoses recorded for the encounter. |
| admission_type_id | int64 | Encoded hospital admission type. |
| discharge_disposition_id | int64 | Encoded discharge disposition. |
| diag_1_group | category | Group derived from the primary diagnosis code. |
| A1Cresult | category | HbA1c test result category. |
| insulin | category | Insulin treatment status or change during the encounter. |

## 4. Auxiliary Columns (5 Columns)

These columns support identification, target definition, encounter analysis and weighting. They must not be included in the model feature matrix.

| Column | Data Type | Description | Usage |
|---|---|---|---|
| encounter_id | int64 | Unique identifier for a hospital encounter. | Identifier only |
| patient_nbr | int64 | Patient identifier shared across encounters. | Identifier only |
| target | int64 | Binary readmission target: 1 for readmission within 30 days, otherwise 0. | Prediction label |
| is_first_encounter | int64 | Indicates the earliest retained encounter for each patient (1 = first, 0 = subsequent). | Slicing and comparison only |
| n_stays_total | int64 | Total number of retained encounters for the patient in this dataset. | Optional sample weighting only |

**Important leakage warning:** `n_stays_total` is calculated using all retained encounters, including future encounters relative to a given encounter. It contains future-informed information and must never be used as a predictive feature. If approved by the team, it may be used only for sample weighting:

`sample_weight = 1 / n_stays_total`

`is_first_encounter`, `encounter_id`, and `patient_nbr` must also be excluded from model features.

## 5. Categorical Value Distributions

### 5.1 Race

| Category | Records |
|---|---:|
| Caucasian | 74,220 |
| AfricanAmerican | 18,772 |
| Unknown | 2,232 |
| Hispanic | 2,017 |
| Other | 1,471 |
| Asian | 628 |
| **Total** | **99,340** |

The original missing indicators (`?` and empty strings) are converted to `Unknown`.

### 5.2 A1Cresult

| Category | Records |
|---|---:|
| None | 82,506 |
| >8 | 8,137 |
| Norm | 4,922 |
| >7 | 3,775 |
| **Total** | **99,340** |

`None` is retained as an original dataset category. It is not a pandas null value.

### 5.3 Insulin

| Category | Records |
|---|---:|
| No | 46,376 |
| Steady | 30,069 |
| Down | 11,908 |
| Up | 10,987 |
| **Total** | **99,340** |

### 5.4 Primary Diagnosis Groups

| Category | Records |
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

The 20 missing primary diagnosis codes are represented by the explicit `Missing` category.

## 6. Missing Value Analysis

The final v2 Parquet dataset was checked for literal question marks, empty strings and pandas null values.

| Check | Count |
|---|---:|
| Question mark markers (`?`) | 0 |
| Empty strings | 0 |
| Actual null values | 0 |

All 99,340 records have zero detected missing values in the final 20-column dataset.

This does not imply that the original raw dataset contained no missing information. Missing indicators were handled during preprocessing, and certain unavailable values remain represented by explicit categories such as `Unknown`, `None`, and `Missing`.

## 7. Column Removal Policy

The v2 preprocessing pipeline removes 14 documented columns:

| Removal Category | Columns | Count |
|---|---|---:|
| Constant columns | examide, citoglipton | 2 |
| Low-frequency medication columns | acetohexamide, glimepiride-pioglitazone, metformin-pioglitazone, metformin-rosiglitazone, troglitazone, glipizide-metformin, tolbutamide, miglitol, tolazamide, chlorpropamide | 10 |
| Explicit removal | weight, payer_code | 2 |
| **Total** | | **14** |

The pipeline validates the constant-column and low-frequency rules before removing these columns.

The `medical_specialty` column is not included in the final 20-column schema.

## 8. Role C Feature Selection

The approved model feature matrix must use only the 15 model attributes.

```python
X = df[MODEL_ATTRIBUTES]
y = df["target"]
```

The following columns must never be included in the predictor matrix:

- encounter_id
- patient_nbr
- target
- is_first_encounter
- n_stays_total

The target is the prediction label, not an input feature.

## 9. Dataset Versioning

| Version | Description | Output |
|---|---|---|
| v1 | First retained encounter per patient | `cohort_v1.parquet` |
| v2 | All retained encounters per patient | `cohort_v2.parquet` |

The v2 pipeline preserves the v1 output and verifies that the first-encounter subset matches v1 across 18 common columns.

The SHA256 checksum for v2 is stored separately in:

`data/processed/cohort_v2.sha256`

## 10. Conclusion

The v2 dataset preserves all 99,340 eligible encounters and provides patient-level encounter history through `n_stays_total` and `is_first_encounter`.

The final dataset contains 20 attributes, with a clear distinction between model inputs and auxiliary information. Its acceptance statistics and first-encounter comparison against v1 passed successfully.

This version provides the prepared dataset for subsequent readmission analysis and model development while documenting important target-leakage restrictions for downstream users.