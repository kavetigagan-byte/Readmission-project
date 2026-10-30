# Data Dictionary

## 1. Introduction

This document describes all 39 columns in the processed dataset used for the Diabetes Hospital Readmission Prediction project.

The dataset is derived from the UCI Diabetes 130-US Hospitals Dataset. It contains patient encounter information, demographic details, admission information, medical procedures, medication records, diagnosis groups and the target variable.

The dictionary documents each column's meaning, data type, missing-value percentage, cleaning decision and justification.

## 2. Dataset Overview

| Attribute | Value |
|---|---|
| Dataset | UCI Diabetes 130-US Hospitals Dataset |
| Total records | 69,987 |
| Total columns | 39 |
| Integer columns | 13 |
| Categorical columns | 26 |
| Missing values detected | 0 |
| Output format | Parquet |

**Note:** Missing percentages are calculated using pandas `isna()`. Zero means no pandas-recognized null values remain; it does not necessarily mean that every value represents a known or observed measurement.

## 3. Column Descriptions

### 3.1 Patient Identification and Demographic Information

| Column | Meaning | Type | Missing | Decision and justification |
|---|---|---|---:|---|
| `encounter_id` | Unique identifier assigned to a hospital encounter | int64 | 0% | Kept as an encounter identifier for record tracking |
| `patient_nbr` | Unique identifier assigned to a patient | int64 | 0% | Kept to identify patients and verify one record per patient |
| `race` | Recorded race category of the patient | category | 0% | Kept; missing or unspecified categories are normalized |
| `gender` | Gender recorded for the patient | category | 0% | Kept after removing invalid gender records |
| `age` | Patient age represented by the midpoint of the original age range | int64 | 0% | Transformed into numerical age midpoint values |

### 3.2 Admission and Hospital Information

| Column | Meaning | Type | Missing | Decision and justification |
|---|---|---|---:|---|
| `admission_type_id` | Coded type of hospital admission | category | 0% | Kept as a categorical admission variable |
| `discharge_disposition_id` | Coded destination or status at hospital discharge | category | 0% | Kept; expired and hospice-related encounters were excluded |
| `admission_source_id` | Coded source from which the patient was admitted | category | 0% | Kept as a categorical admission variable |
| `time_in_hospital` | Number of days the patient stayed in the hospital | int64 | 0% | Kept as a measure of hospital stay duration |
| `medical_specialty` | Medical specialty associated with the hospital encounter | category | 0% | Kept; missing and low-frequency categories are normalized |

### 3.3 Medical Procedures and Previous Healthcare Utilization

| Column | Meaning | Type | Missing | Decision and justification |
|---|---|---|---:|---|
| `num_lab_procedures` | Number of laboratory procedures performed during the encounter | int64 | 0% | Kept as a measure of laboratory activity |
| `num_procedures` | Number of non-laboratory procedures performed during the encounter | int64 | 0% | Kept as a measure of treatment activity |
| `num_medications` | Number of distinct medications administered during the encounter | int64 | 0% | Kept as a measure of medication usage |
| `number_outpatient` | Number of outpatient visits in the year preceding the encounter | int64 | 0% | Kept to represent previous outpatient utilization |
| `number_emergency` | Number of emergency visits in the year preceding the encounter | int64 | 0% | Kept to represent previous emergency utilization |
| `number_inpatient` | Number of inpatient visits in the year preceding the encounter | int64 | 0% | Kept as a measure of previous hospitalization |
| `number_diagnoses` | Total number of diagnoses recorded for the encounter | int64 | 0% | Kept as a measure of patient diagnostic complexity |

### 3.4 Laboratory Test Results

| Column | Meaning | Type | Missing | Decision and justification |
|---|---|---|---:|---|
| `max_glu_serum` | Maximum glucose serum test result category | category | 0% | Kept as a categorical laboratory result |
| `A1Cresult` | HbA1c test result category | category | 0% | Kept to represent blood glucose control information |

### 3.5 Diabetes Medication Information

These columns describe diabetes medications recorded during the hospital encounter. Their values represent medication status or dosage changes.

| Column | Meaning | Type | Missing | Decision and justification |
|---|---|---|---:|---|
| `metformin` | Metformin medication status | category | 0% | Kept as a medication feature |
| `repaglinide` | Repaglinide medication status | category | 0% | Kept as a medication feature |
| `nateglinide` | Nateglinide medication status | category | 0% | Kept as a medication feature |
| `glimepiride` | Glimepiride medication status | category | 0% | Kept as a medication feature |
| `glipizide` | Glipizide medication status | category | 0% | Kept as a medication feature |
| `glyburide` | Glyburide medication status | category | 0% | Kept as a medication feature |
| `pioglitazone` | Pioglitazone medication status | category | 0% | Kept as a medication feature |
| `rosiglitazone` | Rosiglitazone medication status | category | 0% | Kept as a medication feature |
| `acarbose` | Acarbose medication status | category | 0% | Kept as a medication feature |
| `insulin` | Insulin medication status | category | 0% | Kept as a medication feature |
| `glyburide-metformin` | Combined glyburide-metformin medication status | category | 0% | Kept as a medication feature |
| `change` | Indicates whether diabetes medication dosage or type was changed | category | 0% | Kept as a medication-change indicator |
| `diabetesMed` | Indicates whether diabetes medication was prescribed | category | 0% | Kept as an overall diabetes medication indicator |

**Medication value interpretation:**

- `No`: Medication was not prescribed.
- `Steady`: Medication dosage remained unchanged.
- `Up`: Medication dosage was increased.
- `Down`: Medication dosage was decreased.

The `change` and `diabetesMed` variables use their own categorical values rather than the medication dosage categories.

### 3.6 Target Variable

| Column | Meaning | Type | Missing | Decision and justification |
|---|---|---|---:|---|
| `target` | Binary indicator of readmission within 30 days | int64 | 0% | Created from the original `readmitted` column; 1 represents `<30`, while 0 represents `NO` or `>30` |

### 3.7 Diagnosis Group Columns

The original diagnosis codes are grouped into broader clinical categories to make the dataset easier to interpret.

| Column | Meaning | Type | Missing | Decision and justification |
|---|---|---|---:|---|
| `diag_1_group` | Clinical category of the primary diagnosis | category | 0% | Transformed from the primary diagnosis code |
| `diag_2_group` | Clinical category of the secondary diagnosis | category | 0% | Transformed from the secondary diagnosis code |
| `diag_3_group` | Clinical category of the additional diagnosis | category | 0% | Transformed from the third diagnosis code |

The diagnosis groups include categories such as:

- Circulatory
- Respiratory
- Digestive
- Diabetes
- Injury
- Musculoskeletal
- Genitourinary
- Neoplasms
- Other
- Missing

### 3.8 Admission and Discharge Description Columns

These columns provide readable descriptions for the coded admission and discharge variables. Their descriptions are derived from `IDS_mapping.csv`.

| Column | Meaning | Type | Missing | Decision and justification |
|---|---|---|---:|---|
| `admission_type_id_desc` | Text description of the admission type code | category | 0% | Added using the admission type mapping |
| `discharge_disposition_id_desc` | Text description of the discharge disposition code | category | 0% | Added using the discharge disposition mapping |
| `admission_source_id_desc` | Text description of the admission source code | category | 0% | Added using the admission source mapping |

## 4. Data Type Summary

| Data type | Number of columns | Description |
|---|---:|---|
| int64 | 13 | Numeric identifiers, counts, age and target |
| category | 26 | Categorical demographic, medical and diagnosis information |
| **Total** | **39** | **All processed columns** |

## 5. Missing Value Analysis

The missing-value analysis was performed using pandas.

| Metric | Result |
|---|---:|
| Total columns | 39 |
| Columns with pandas-detected missing values | 0 |
| Columns without pandas-detected missing values | 39 |
| Total missing percentage | 0% |

Missing or unspecified categorical information may be represented by explicit categories such as `Unknown` or `Missing`, rather than pandas null values.

## 6. Conclusion

The processed dataset contains 39 columns covering patient demographics, hospital admission details, medical procedures, medication information, diagnosis groups and readmission outcomes.

All 39 columns have zero pandas-detected missing values. The dataset uses 13 integer columns and 26 categorical columns.

This data dictionary provides a reference for understanding the processed cohort and supports subsequent exploratory data analysis and model development.
