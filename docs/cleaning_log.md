# Data Cleaning Log

## 1. Project Information

| Item | Details |
|---|---|
| Project | Diabetes Hospital Readmission Prediction |
| Dataset | UCI Diabetes 130-US Hospitals Dataset |
| Role | Role A – Data Owner |
| Processing script | `src/build_cohort.py` |
| Output dataset | `data/processed/cohort_v1.parquet` |

## 2. Source Files

- `diabetic_data.csv` – original hospital encounter records.
- `IDS_mapping.csv` – descriptions for admission type, discharge disposition and admission source codes.

## 3. Cleaning Operations

The following table records the main cleaning operations specified for the cohort.

| # | Cleaning rule | Rows before | Rows after | Removed | Reason |
|---|---|---:|---:|---:|---|
| 1 | Remove records with `Unknown/Invalid` gender | 101,766 | 101,763 | 3 | Remove invalid gender records |
| 2 | Remove expired and hospice-related discharges | 101,763 | 99,340 | 2,423 | These encounters cannot represent a subsequent readmission |
| 3 | Retain the first encounter per patient | 99,340 | 69,987 | 29,353 | Keep one encounter per patient and avoid repeated patient records |

### Column-level cleaning

| # | Operation | Result | Reason |
|---|---|---|---|
| 4 | Remove constant columns | 3 columns removed | Constant columns provide no useful variation |
| 5 | Remove low-frequency medication columns | 9 columns removed | Fewer than 100 records had a value other than `No` |

### Other transformations

1. Created a binary target variable:
   - `1` – readmitted within 30 days (`<30`).
   - `0` – not readmitted within 30 days.
2. Converted age ranges into numerical midpoint values.
3. Grouped diagnosis codes into broader diagnostic categories.
4. Normalized race and medical specialty categories.
5. Added admission, discharge and admission-source descriptions using `IDS_mapping.csv`.
6. Converted columns into appropriate data types.
7. Sorted the final cohort and saved it as a Parquet file.

## 4. Final Cohort Summary

| Metric | Result |
|---|---:|
| Final records | 69,987 |
| Positive cases (`target = 1`) | 6,285 |
| Negative cases (`target = 0`) | 63,702 |
| Positive rate | 8.98% |

## 5. Assumptions and Decisions

- **First encounter:** The smallest `encounter_id` is used as a proxy for the first encounter because a reliable encounter date is not available.
- **Target definition:** The `<30` readmission category is treated as positive. The other readmission categories are treated as negative.
- **Medical specialty:** Missing medical specialty values are represented as `Unknown`.
- **Expired and hospice discharges:** These records are excluded because they cannot undergo a subsequent readmission.
- **Missing values:** Missing values are handled according to the defined cleaning rules. Model-specific imputation is not performed in this phase.

## 6. Output and Reproducibility

The cleaned dataset is saved at:

`data/processed/cohort_v1.parquet`

The SHA-256 fingerprint is saved at:

`data/processed/cohort_v1.sha256`

The processing script is available at:

`src/build_cohort.py`

## 7. Validation

The processed cohort was checked using automated tests.

| Validation | Result |
|---|---|
| Total automated tests | 10 |
| Passed | 10 |
| Failed | 0 |

The tests verify the expected cohort size, target distribution, unique patient records, valid gender values, age values, diagnosis groups and mapping-description columns.

## 8. Conclusion

The cleaning process produced a cohort of 69,987 patient records containing 6,285 positive readmission cases and 63,702 negative cases. The processed dataset passed all 10 automated validation tests and is ready for exploratory data analysis.
