"""
STEP 8 of 8: Score 100 brand-new applicants the model has never seen.

WHAT this step does:
    Take Unseen_Dataset.xlsx (100 applicants with no known outcome), clean it EXACTLY the way the
    training data was cleaned, encode and scale it EXACTLY the way the training data was, and ask
    the trained model for a predicted probability of Approval for each one.

WHY this step exists, explained from zero:
    This is the actual point of building a model at all -- everything from Step 1 onward was in
    service of this moment: a new person applies, and we need a real answer. The critical rule
    for this step: every transformation applied here must be the SAME transformation, using the
    SAME learned numbers (medians, the scaler's mean/std, the exact column list), as Steps 3-5
    used on the training data. We do NOT recompute a new median from these 100 rows, and we do NOT
    re-fit a new scaler on them -- doing either would silently change what the model is looking at
    compared to what it was actually trained on, invalidating its learned coefficients.

WHAT you'll learn in this step:
    - Why "apply the training-time transformation" and "recompute a fresh transformation" are two
      completely different things, and only one of them is correct here.
    - How to turn a model's probability output into a business decision.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

UNSEEN_PATH = RAW_DIR / "Unseen_Dataset.xlsx"
RECIPE_PATH = DATA_DIR / "03_cleaning_recipe.json"
SELECTED_FEATURES_PATH = DATA_DIR / "04_selected_features.csv"
FINAL_COLUMNS_PATH = DATA_DIR / "05_final_feature_columns.json"
SCALER_PATH = DATA_DIR / "05_scaler.joblib"
MODEL_PATH = OUTPUT_DIR / "06_logistic_regression_model.joblib"
PREDICTIONS_OUT_PATH = OUTPUT_DIR / "08_unseen_predictions.csv"

THRESHOLD = 0.5  # the same threshold used for the headline metrics in Step 7

print("=" * 90)
print("STEP 8: Predict on unseen (brand-new) applicants")
print("=" * 90)

unseen = pd.read_excel(UNSEEN_PATH, sheet_name="Sheet1")
print(f"\nLoaded {len(unseen)} new applicants from Unseen_Dataset.xlsx, with {unseen.shape[1]} raw columns.")
print("Notice: there is no PROSPECTID and no target column here -- these are genuinely new people,")
print("not a hidden piece of our training or test data.")

with open(RECIPE_PATH) as f:
    recipe = json.load(f)
selected = pd.read_csv(SELECTED_FEATURES_PATH)
numeric_features = selected.loc[selected["type"] == "numeric", "feature"].tolist()
categorical_features = selected.loc[selected["type"] == "categorical", "feature"].tolist()
with open(FINAL_COLUMNS_PATH) as f:
    final_columns_info = json.load(f)
final_feature_columns = final_columns_info["final_feature_columns"]
scaled_columns = final_columns_info["scaled_columns"]
scaler = joblib.load(SCALER_PATH)
model = joblib.load(MODEL_PATH)

# ---------------------------------------------------------------------------------------------
# 8a. Replace the -99999 sentinel with NaN -- identical to Step 3a
# ---------------------------------------------------------------------------------------------
sentinel = recipe["sentinel_value"]
numeric_cols_in_unseen = unseen.select_dtypes(include="number").columns.tolist()
n_sentinel = 0
for col in numeric_cols_in_unseen:
    mask = unseen[col] == sentinel
    n_sentinel += mask.sum()
    unseen.loc[mask, col] = np.nan
print(f"\nReplaced {n_sentinel} sentinel ({sentinel}) cells with NaN.")

# ---------------------------------------------------------------------------------------------
# 8b. Impute missing values using the SAVED training medians -- not new medians from this data
# ---------------------------------------------------------------------------------------------
# Using training medians instead of recomputing from these 100 rows matters for two reasons:
#   1. Consistency: the model's coefficients were learned assuming values were filled with the
#      TRAINING median. Filling with a different number at prediction time changes what those
#      coefficients are effectively being applied to.
#   2. Reliability: 100 rows is a small sample -- a median computed from just these 100 applicants
#      could be a poor, noisy estimate compared to the ~38,500 training rows the saved median came
#      from.
print("\nImputing missing values using the medians SAVED from the training data (Step 3):")
for col, median_value in recipe["impute_medians"].items():
    if col not in unseen.columns:
        continue  # this column wasn't selected as a feature (or wasn't part of the servable schema)
    flag_col = f"{col}_was_missing"
    unseen[flag_col] = unseen[col].isna().astype(int)
    n_missing_here = unseen[flag_col].sum()
    unseen[col] = unseen[col].fillna(median_value)
    if n_missing_here > 0:
        print(f"  {col:<32} {n_missing_here} missing -> filled with training median {median_value:.4f}")

# ---------------------------------------------------------------------------------------------
# 8c. Fix income using the SAVED cutoff, median, and cap -- not recomputed
# ---------------------------------------------------------------------------------------------
if "NETMONTHLYINCOME" in unseen.columns:
    low_cutoff = recipe["income_low_cutoff"]
    income_median = recipe["income_median"]
    income_cap = recipe["income_cap_p99"]
    low_mask = unseen["NETMONTHLYINCOME"] < low_cutoff
    print(f"\nApplicants with implausibly low income (< {low_cutoff}): {low_mask.sum()} "
          f"-> treated as missing, filled with training median {income_median:,.0f}")
    unseen.loc[low_mask, "NETMONTHLYINCOME"] = np.nan
    unseen["NETMONTHLYINCOME"] = unseen["NETMONTHLYINCOME"].fillna(income_median)
    n_capped = (unseen["NETMONTHLYINCOME"] > income_cap).sum()
    unseen["NETMONTHLYINCOME"] = unseen["NETMONTHLYINCOME"].clip(upper=income_cap)
    print(f"Applicants capped at the training-set 99th percentile ({income_cap:,.0f}): {n_capped}")

# ---------------------------------------------------------------------------------------------
# 8d. Subset to the selected features, one-hot encode, and align columns to match training
# ---------------------------------------------------------------------------------------------
model_input = unseen[numeric_features + categorical_features].copy()
model_input = pd.get_dummies(model_input, columns=categorical_features, drop_first=True)
bool_cols = [c for c in model_input.columns if model_input[c].dtype == bool]
model_input[bool_cols] = model_input[bool_cols].astype(int)

# CRITICAL step: reindex to the EXACT column list (and order) the model was trained on.
#   - If Unseen_Dataset simply doesn't contain a category that appeared in training (e.g. no
#     "PROFESSIONAL" education in these particular 100 rows), reindex adds that dummy column back
#     as all 0s -- correctly saying "none of these applicants are in that category."
#   - If get_dummies happened to produce an unexpected extra column, reindex drops it, since the
#     model has no coefficient for a column it never saw during training.
missing_from_unseen = set(final_feature_columns) - set(model_input.columns)
extra_in_unseen = set(model_input.columns) - set(final_feature_columns)
print(f"\nColumns present in training but not naturally created here (added back as all-0): "
      f"{sorted(missing_from_unseen) if missing_from_unseen else 'none'}")
print(f"Columns created here but not part of the trained model (dropped): "
      f"{sorted(extra_in_unseen) if extra_in_unseen else 'none'}")

model_input = model_input.reindex(columns=final_feature_columns, fill_value=0)

# ---------------------------------------------------------------------------------------------
# 8e. Scale numeric columns using the SAVED, already-fitted scaler -- .transform(), not .fit()
# ---------------------------------------------------------------------------------------------
model_input[scaled_columns] = scaler.transform(model_input[scaled_columns])
print(f"\nScaled {len(scaled_columns)} numeric columns using the scaler fitted on training data in Step 5.")

# ---------------------------------------------------------------------------------------------
# 8f. Predict
# ---------------------------------------------------------------------------------------------
predicted_probability = model.predict_proba(model_input)[:, 1]
predicted_decision = np.where(predicted_probability >= THRESHOLD, "Approve", "Reject")

results = pd.DataFrame({
    "applicant_row": range(len(unseen)),
    "predicted_probability_of_approval": predicted_probability.round(4),
    "predicted_decision": predicted_decision,
})

print(f"\nPredictions for all {len(results)} new applicants (first 15 shown):")
print(results.head(15).to_string(index=False))

print(f"\nOverall: {(predicted_decision == 'Approve').sum()} predicted Approve, "
      f"{(predicted_decision == 'Reject').sum()} predicted Reject "
      f"({(predicted_decision == 'Approve').mean():.1%} approve rate on this new batch -- compare "
      f"this to the ~74% Approve rate seen in the training data; a very different rate here would "
      f"be worth investigating rather than assuming the model is automatically right).")

results.to_csv(PREDICTIONS_OUT_PATH, index=False)
print(f"\nSaved predictions to: {PREDICTIONS_OUT_PATH}")
print("\nDone. This was the last of the 8 steps -- see README.md for a full recap.")
