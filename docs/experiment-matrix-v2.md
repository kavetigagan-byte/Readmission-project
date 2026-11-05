# Experiment specification — readmission model

**Revised 2 Nov 2026, before any model is fitted.** Supersedes `experiment-matrix.md`.

**Scope:** all admissions only. One split, grouped by patient. 15 attributes. Two models compared.

---

## 1. Task and why it is classification

**Binary classification.**

- **Class 1** — readmitted within 30 days (`readmitted == "<30"`): 11,314 admissions.
- **Class 0** — everything else (`NO` or `>30`): 88,026 admissions.

Logistic regression is used **as a classifier**, despite the word "regression" in its name. How that works:

1. It computes a weighted sum of the features: `z = b + w₁x₁ + w₂x₂ + …`
2. It squashes `z` through the logistic (sigmoid) function into a number between 0 and 1: `p = 1 / (1 + e^(−z))`
3. `p` is the probability of class 1. A threshold (0.5, or anything chosen on validation) turns that probability into a class.

The name comes from step 1–2: it fits a **linear model to the log-odds**, which is a continuous quantity, so historically it sits in the "regression" family. But the output is a bounded probability used to assign a class, so the *task* is classification. Two practical proofs:

- `model.predict_proba(X)` gives a probability; `model.predict(X)` gives `0` or `1` from the same object. A regression model has no `.predict()` class equivalent.
- Without the threshold it is a **risk score**, not a decision. The classifier is the score plus the cut-off.

Consequence for us: accuracy is the wrong metric (predicting "no" always scores 89%), the threshold is a choice we must justify, and the *quality of the probability* (calibration) matters as much as the ranking.

**Why not regression:** there is no continuous outcome. `readmitted` has three categories, and the dataset has no dates, so "days until readmission" does not exist. With dates, the right method would be survival analysis, not regression.

---

## 2. The design

| Run | Cohort | Rows | Model |
|---|---|---|---|
| **M1** | all admissions | 99,340 | logistic regression |
| **M2** | all admissions | 99,340 | gradient boosting |

No weights, no first-stay cohort, no extra variants.

The one question this answers: **is the signal linear?** If M2 beats M1 clearly, non-linearity and interactions matter. If the gap is small, the linear model is the honest choice — simpler, and explainable coefficient by coefficient.

Both runs use the identical split, the identical 15 attributes, and the identical training rows.

---

## 3. The 15 attributes

`age` · `gender` · `race` · `time_in_hospital` · `number_inpatient` · `number_emergency` · `number_outpatient` · `num_lab_procedures` · `num_medications` · `number_diagnoses` · `admission_type_id` · `discharge_disposition_id` · `diag_1_group` · `A1Cresult` · `insulin`

Same 15 for both models and for all rows. Chosen from clinical reasoning and availability at discharge, **before** checking their association with the target — outcome-driven selection over the whole file lets test rows decide which features exist, which inflates the reported score. A's univariate table is evidence the choice was sensible; it is not a filter.

Never features: `encounter_id`, `patient_nbr`, `target`, `is_first_encounter`, `n_stays_total`. The last counts future admissions and would leak the answer.

---

## 4. Split — grouped by patient

`data/processed/split_assignment.csv` (`patient_nbr`, `fold`), built once:

- `StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42)`, stratified on the first-encounter label
- folds 0–2 → test (~15% of patients), 3–5 → val, 6–19 → train
- every row of a patient inherits that patient's fold
- assertion: no `patient_nbr` appears in two folds

**Why grouped:** 16,341 patients have more than one admission, supplying 45,694 of the 99,340 rows. A random row-level split puts the same person in train and test, and the model then recognises patients rather than learning patterns. Grouping removes that entirely.

Expected: ~10,498 patients per val/test fold, ~14,900 rows per val/test fold (~1,697 positives each), ~69,500 training rows (~7,920 positives).

---

## 5. Class balance: 35% positive, 65% negative

The raw ratio is 11,314 : 88,026 — about 1 : 7.8. We sample class 0 down to **35% of the training rows**, i.e. **1.857 negatives per positive**:

| | Before | After sampling |
|---|---|---|
| Class 1 | 11,314 | 11,314 (all kept) |
| Class 0 | 88,026 | **21,012** |
| Total | 99,340 | **32,326** |
| Positive rate | 11.39% | **35.00%** |
| Ratio | 1 : 7.8 | **1 : 1.86** |

Measured effect of the ratio on the reported metrics (same split, same 15 attributes, `docs/reference/model-screening.md`): **none**. Logistic PR-AUC 0.1973 / 0.1984 / 0.1985 / 0.1995 at 1.0 / 1.86 / 2.65 / 4.0 negatives per positive. The ratio is therefore chosen for convenience, not performance — 35:65 is the setting we state.

### Train only

The sampling is applied to **training rows only, after the split**. Val and test keep the natural 11.39%.

If the whole file were sampled, the test set would also be 27.4% positive, and then PR-AUC, calibration and prevalence all describe a population that does not exist. Every number would be uninterpretable and incomparable to published work on this dataset.

```python
def undersample_train(df_train, ratio=1.857, seed=42):
    pos = df_train[df_train.target == 1]
    neg = df_train[df_train.target == 0].sample(
        n=round(ratio * len(pos)), random_state=seed
    )
    r = len(neg) / len(df_train[df_train.target == 0])
    return pd.concat([pos, neg]).sample(frac=1, random_state=seed), r
```

Training fold: ~7,920 positives → ~14,710 negatives → ~22,630 rows. `r ≈ 0.240`.

### The correction that cannot be skipped

Sampling negatives multiplies their odds by `1/r`, so the fitted intercept is shifted. Map every probability back before reporting it:

```
p_true = sigmoid( logit(p_model) + log(r) )        log(r) ≈ −1.427
```

PR-AUC and ROC-AUC are rank-based and unaffected. **Brier score, calibration curve and any quoted probability are wrong without this correction** — an average patient would be reported at ~35% risk instead of ~11%. `r` is stored alongside every fitted model.

### Do not also use `class_weight`

The 35:65 sampling *is* the imbalance correction. Adding `class_weight="balanced"` corrects a second time and the intercept shift becomes wrong twice over. One or the other.

### No per-patient truncation — keep every admission

**Decision: no patient's rows are deleted, capped or truncated, and no outlier rows are removed.**

The only rows removed anywhere are random **negative** rows in the training fold, for the class ratio. There is no patient-level rule.

Measured, same split and attributes:

| Option | Rows | Positives | Test PR-AUC |
|---|---|---|---|
| **Keep everything (chosen)** | **99,340** | **11,314** | **0.1984** |
| Cap each patient's positive rows at 10 (train only) | −39 train rows | −39 train positives | 0.1982 |
| Keep first 10 admissions per patient | 98,883 | 11,121 | 0.1892 |
| Drop patients with >10 admissions | 97,933 | 10,740 | 0.1807 |

The difference between keeping everything and the most aggressive deletion is **0.018 PR-AUC**, and the deletion removes the highest-risk patients — those with more than 10 admissions are readmitted at **40.8%** versus 11.4% overall, and hold 5.07% of all positives.

One patient contributes 40 admissions and 23 of the 11,314 positives. That is handled by the grouped split (they land in exactly one fold) and by clustered standard errors (B), not by deleting their record. State it in the limitations, once.

### The one diagnostic worth keeping


Fit M1 once without any sampling, using `class_weight="balanced"` on the full training set, and report its test PR-AUC as a single line. That measures what undersampling cost. If the cost is small, the line proves it; if it is large, you know the price. A check, not a variant.

---

## 6. Models and preprocessing

- Preprocessing inside a `Pipeline`, fitted on training data only: median imputation (numeric), most-frequent (categorical), one-hot for nominal, standardisation for the linear model.
- **M1:** `LogisticRegression(C` from `{0.01, 0.1, 1}`, chosen on validation `)`. L2 penalty. No class weights.
- **M2:** `HistGradientBoostingClassifier(max_iter=300, early_stopping=True, random_state=42)`. No class weights.
- `random_state=42` everywhere. Identical settings across both runs.

---

## 7. Metrics

Validation first; test **once**, at the end, on natural-prevalence rows.

**Primary:** PR-AUC (average precision). Baseline to beat: **0.1139**. Report the dummy classifier beside it, so "above baseline" is visible rather than claimed.

**Secondary:** ROC-AUC · **capacity table** — precision, recall and lift when flagging the top 2% / 5% / 10% / 20% of admissions by predicted risk (threshold-free, so it does not depend on the sampling correction) · Brier score after the `r` correction · calibration slope and intercept · reliability curve for the headline model · train − val PR-AUC gap.

*Why capacity and not "recall at precision ≥ 0.30":* after 35:65 undersampling the fitted probabilities sit on a 35% scale, so any fixed threshold is meaningless until the `+log(r)` correction is applied. A capacity table asks the operational question directly — *if the clinic can follow up 10% of discharges, how many readmissions do we catch?* — and needs no threshold at all.

**Subgroups:** patients with a single admission vs repeat patients; patients with ≥ 3 readmissions, pooled over val+test, labelled descriptive.

**Uncertainty:** paired bootstrap over `patient_nbr`, 2,000 resamples, all of a patient's rows moving together. Row-level resampling understates variance because repeat rows are correlated.

---

## 8. Decision rule

Ranked on **validation PR-AUC**.

1. Headline = higher of M1 and M2 on validation.
2. If the gap is within **0.005** → headline = **M1** (logistic). The simpler model is the stronger claim, and it doubles as B's explanatory model.
3. Memorisation gate: train − val PR-AUC gap > **0.15** → demote that run.
4. Calibration gate: Brier worse than the base-rate predictor → neither is reported as usable; that is the finding.
5. Both are reported on test regardless. The rule only decides which one is called "our model".

---

## 9. What each outcome would mean

| Result | Reading |
|---|---|
| M2 ≫ M1 | Real non-linearity and interactions. Report both, lead with boosting, keep M1 for explanation. |
| M2 ≈ M1 | The signal is largely additive. A legitimate finding, not a failure — and the linear model can then be defended as sufficient. |
| Both near the 0.1139 baseline | The 15 attributes do not carry enough signal. Report it, then test whether the cut to 15 was too aggressive. |
| Undersampled worse than full + `class_weight` | The imbalance fix cost data. Report the price; prefer the full-data fit. |
| Train ≫ val | Memorisation — most likely via repeat patients. Check the grouped split first, then model complexity. |

---

## 10. Checks before any number is trusted

- [ ] split file: 69,987 patients, no patient in two folds, val/test 14–16% of patients
- [ ] positive rate per fold within ±1 pp of 11.39%
- [ ] `cohort_v2_min` has exactly 20 columns; the 15 attributes match §3 exactly
- [ ] `n_stays_total` and `is_first_encounter` absent from the feature matrix (assert, don't eyeball)
- [ ] undersampled train: all positives kept, negatives at 1.86:1 ±1%, seed-deterministic
- [ ] **no val or test row in the undersampled frame** — asserted in code, raises on violation
- [ ] `r` stored with each model; Brier and calibration computed after the logit correction
- [ ] test mean predicted risk ≈ 11.4% after correction (within ~1 pp)
- [ ] identical hyperparameters and preprocessing for M1 and M2
- [ ] test evaluated exactly once; count recorded

---

## 11. Change log

| Date | Change | Reason |
|---|---|---|
| 2 Nov 2026 | v1: four weighting variants (unweighted, `1/k`, first-stay, `2/k` for ≥3 readmissions) | Testing whether repeat-patient duplication distorts the fit |
| 2 Nov 2026 | Dropped all custom weights; 15 attributes; cohort × model comparison | `1/k` and `2/k` depend on information correlated with the outcome, and `2/k` had no principled value |
| 2 Nov 2026 | **All admissions only; 35:65 undersampling (11,314 positives : 21,012 negatives); logistic vs boosting** | First-stay cohort dropped as unnecessary. Class ratio set to 35% positive; a sweep at 1.0 / 1.86 / 2.65 / 4.0 negatives per positive moved PR-AUC by less than 0.002, so the ratio is a convenience, not a modelling choice. |

Any further change is dated here with its reason.
