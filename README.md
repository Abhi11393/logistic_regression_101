# Logistic Regression 101: Credit Approval Prediction from Bureau + Bank Data

A from-scratch, fully-explained logistic regression build: predicting whether a loan applicant
gets Approved or Rejected, using real CIBIL bureau data combined with a bank's own internal
records. Built as a teaching project -- **every calculation is shown and reasoned about, nothing
is assumed** -- broken into 8 small, independently-runnable scripts so each step can be read and
understood on its own.

**A companion project,
[`logistic_regression_102`](https://github.com/Abhi11393/logistic_regression_102), rebuilds this
exact same problem using the industry-standard credit-scoring toolkit instead (WOE/IV/PDO
scorecard scaling) -- worth reading alongside this one to see where and why the two approaches
diverge.**

## The data

**Not included in this repo** (see `.gitignore`) -- the source `.xlsx` files are excluded because
of their size and unclear redistribution licensing for a public repo. To run this project
yourself, place the three files below into `data/raw/` first. Three files:

- **`External_Cibil_Dataset.xlsx`** -- 51,336 applicants x 62 columns. This is the "bureau data":
  a credit report pulled from CIBIL (India's credit bureau) covering each applicant's credit
  history across *every* lender, not just this one -- delinquency history, enquiry counts, bureau
  scores, and demographics.
- **`Internal_Bank_Dataset.xlsx`** -- the same 51,336 applicants x 26 columns, but this time it's
  the bank's *own* records: how many loans/trade lines the applicant has with this specific bank,
  broken down by type and age.
- **`Unseen_Dataset.xlsx`** -- 100 additional applicants, no ID, no known outcome, and only 41 of
  the ~87 combined columns. This stands in for "100 brand new people just applied" -- used only in
  the final step, to actually score new applicants the way the finished model would be used for
  real.

**Target**: `Approved_Flag`, originally 4 risk-priority classes (P1 best, P4 worst). This project
collapses it to a binary target -- `Approve` (P1+P2, 74.0% of applicants) vs. `Reject` (P3+P4,
26.0%) -- specifically so logistic regression could be learned in its simplest, foundational form
(one sigmoid, one decision boundary) before anything more complex.

## The 8 steps

Each script in `scripts/` loads the previous script's saved output from `data/`, does one clearly
scoped thing, and saves its own output for the next script -- run them in order with:

```
.venv\Scripts\python.exe scripts\01_load_and_merge.py
.venv\Scripts\python.exe scripts\02_explore_data.py
.venv\Scripts\python.exe scripts\03_clean_data.py
.venv\Scripts\python.exe scripts\04_feature_selection.py
.venv\Scripts\python.exe scripts\05_encode_scale_split.py
.venv\Scripts\python.exe scripts\06_train_model.py
.venv\Scripts\python.exe scripts\07_evaluate_model.py
.venv\Scripts\python.exe scripts\08_predict_unseen.py
```

(One-time setup, if you haven't already: `python -m venv .venv` then
`.venv\Scripts\python.exe -m pip install -r requirements.txt`.)

| # | Script | What it does | What you learn |
|---|---|---|---|
| 1 | `01_load_and_merge.py` | Load both source files, verify the join key lines up, merge into one table. | How to check a merge key is trustworthy *before* trusting a merge. |
| 2 | `02_explore_data.py` | Look for problems: target balance, the `-99999` sentinel value, income outliers -- no cleaning yet. | Diagnose before you treat. |
| 3 | `03_clean_data.py` | Turn `-99999` into real `NaN`; drop columns that are mostly missing, impute the rest with the median (+ a `_was_missing` flag); fix income outliers; build the binary target. Saves the exact numbers used (medians, drop list) to a "cleaning recipe" JSON for reuse in Step 8. | Imputation, and why *which* columns to drop vs. keep depends on how much is actually missing. |
| 4 | `04_feature_selection.py` | Chi-square test (categorical vs. target), t-test (numeric vs. target), then VIF to remove redundant numeric features. Also restricts candidates to columns that actually exist in `Unseen_Dataset` -- a feature that can't be obtained for a new applicant can't be used, no matter how significant it looks historically. | Statistical significance testing, p-values, and multicollinearity. |
| 5 | `05_encode_scale_split.py` | One-hot encode categoricals, split into train/test (stratified), scale numeric features -- fitting the scaler on the training data only. | The dummy variable trap, and *data leakage* (the single most important rule in this project). |
| 6 | `06_train_model.py` | Walks through the sigmoid function and log-odds by hand on toy numbers, then fits the real model and reads its coefficients as odds ratios. | What logistic regression actually computes, not just how to call `.fit()`. |
| 7 | `07_evaluate_model.py` | Confusion matrix computed BY HAND, then accuracy/precision/recall/F1 derived from it, confirmed against scikit-learn; ROC curve and AUC; comparison against a naive baseline. | Every core classification metric, from first principles, with a sanity check against blindly trusting one number. |
| 8 | `08_predict_unseen.py` | Applies the *exact* Step 3/5 transformations (saved medians, saved scaler -- never recomputed) to the 100 unseen applicants and predicts. | Why production scoring must reuse training-time statistics, not recompute new ones. |

## Real results (from this actual run)

| Metric | Model | Naive "always Approve" baseline |
|---|---|---|
| Accuracy | **85.3%** | 74.0% |
| Precision | 86.8% | 74.0% |
| Recall | 94.6% | 100% (meaningless here -- see `07_evaluate_model.py`'s explanation) |
| F1 | 90.5% | -- |
| AUC | **0.877** | 0.50 (random) |

On the 100 unseen applicants: **80 predicted Approve, 20 predicted Reject** (an 80% approve rate,
close to but somewhat above the 74% seen in training -- see Known Limitations below).

**What mattered most** (by odds-ratio magnitude, from `output/06_coefficients.csv`):
`enq_L3m` (recent credit enquiries in the last 3 months -- more enquiries strongly *lowers* approval
odds, a classic "credit-hungry" signal), `Age_Oldest_TL` (older credit history *raises* approval
odds), and `num_std_12mts` (a bureau-reported standard/healthy account count, also raising odds).
**What didn't survive Step 4's significance testing**: `GENDER` (p=0.115) and, more surprisingly,
`NETMONTHLYINCOME` (p=0.153) -- once bureau behavior is accounted for, raw income alone did not
separate Approved from Rejected applicants in this dataset.

## Known limitations / next steps

1. **The binary collapse (P1+P2 vs P3+P4) throws away information** the original 4-class target
   had -- a P1 and a P2 applicant are treated identically, though the source data distinguished
   them. A natural next step: revisit this as a multinomial (multi-class) logistic regression, or
   train a separate model per class boundary, once the binary fundamentals here feel solid.
2. **The 0.5 decision threshold was never tuned.** It's the natural default, but Step 7 computes
   the full ROC curve -- a real deployment would pick a threshold based on the actual cost of a
   False Positive (bad risk approved) vs. a False Negative (good applicant rejected) to this
   specific business, which likely isn't symmetric.
3. **No cross-validation.** Results come from one train/test split (with a fixed seed for
   reproducibility). A more rigorous evaluation would repeat this across several splits (or use
   k-fold cross-validation) to confirm these metrics aren't an artifact of one particular split.
4. **Feature availability was constrained to `Unseen_Dataset`'s schema** (see Step 4), which meant
   giving up `AGE` and several other potentially-predictive columns. A real team would push to
   capture those fields for new applicants rather than permanently excluding them.
5. **The 80% vs. 74% approve-rate gap on unseen data** (Step 8) is exactly the kind of shift a
   real deployment would want to monitor over time (see the PSI drift-monitoring approach built in
   the `vehicle_loan_default_risk` project) -- one batch of 100 isn't enough to say whether this is
   meaningful drift or just small-sample noise.
