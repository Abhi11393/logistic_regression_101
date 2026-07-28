"""
STEP 5 of 8: Turn the selected features into pure numbers a model can use, then split into
train/test sets.

WHAT this step does:
    1. One-hot encode the categorical columns (turn text categories into 0/1 columns).
    2. Split the data into a TRAINING set and a TEST set.
    3. Scale the numeric columns (put them all on a comparable numeric range), fitting the scaler
       on the training data only.

WHY this step exists, explained from zero:
    A logistic regression model is, underneath, a mathematical formula that multiplies each
    feature by a learned number (a coefficient) and adds them up. That formula has no idea what
    "Married" or "Single" means -- it can only multiply actual numbers. So every text column has
    to become numeric columns first (one-hot encoding). Separately, the model also needs to be
    tested honestly on data it never saw during training -- that's the train/test split -- and
    every one of its features needs to be on a comparable numeric scale, or a large-magnitude
    column like income (tens of thousands) will dominate a small-magnitude column like a 0-to-1
    ratio, not because it's more important, but purely because of the units it happens to be
    measured in.

WHAT you'll learn in this step:
    - One-hot encoding, and the "dummy variable trap" (why we drop one category per column).
    - Train/test splitting, and why we stratify it.
    - Feature scaling (standardization), and the single most important rule about it: fit the
      scaler on the training data ONLY, never on the full dataset -- this section explains why in
      detail, because getting this wrong is a very common, very serious real-world mistake called
      DATA LEAKAGE.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CLEANED_PATH = DATA_DIR / "03_cleaned.csv"
SELECTED_FEATURES_PATH = DATA_DIR / "04_selected_features.csv"
TRAIN_OUT_PATH = DATA_DIR / "05_train.csv"
TEST_OUT_PATH = DATA_DIR / "05_test.csv"
SCALER_OUT_PATH = DATA_DIR / "05_scaler.joblib"
FINAL_COLUMNS_OUT_PATH = DATA_DIR / "05_final_feature_columns.json"

SEED = 42  # fixed so the split and every downstream result is exactly reproducible on re-run

print("=" * 90)
print("STEP 5: Encode, split, and scale")
print("=" * 90)

df = pd.read_csv(CLEANED_PATH)
selected = pd.read_csv(SELECTED_FEATURES_PATH)
numeric_features = selected.loc[selected["type"] == "numeric", "feature"].tolist()
categorical_features = selected.loc[selected["type"] == "categorical", "feature"].tolist()
target = "Approved_Binary"

print(f"\nUsing {len(numeric_features)} numeric features and {len(categorical_features)} "
      f"categorical features chosen in Step 4.")

model_df = df[numeric_features + categorical_features + [target]].copy()

# ---------------------------------------------------------------------------------------------
# 5a. One-hot encode the categorical columns
# ---------------------------------------------------------------------------------------------
# pd.get_dummies turns one text column with, say, 3 categories ("Married", "Single", "Divorced")
# into separate 0/1 columns, one per category, with a 1 marking which category that row actually
# had.
#
# drop_first=True deliberately leaves OUT one category per column (the "reference" category) --
# this is not an oversight, it's required. If we kept a dummy column for EVERY category, they
# would always add up to exactly 1 for every row (every applicant has EXACTLY one marital status),
# which means the columns become perfectly predictable from each other -- this is called the
# "dummy variable trap," and it's a special, guaranteed case of the exact multicollinearity
# problem Step 4's VIF check was hunting for. Dropping one category avoids it entirely: that
# dropped category's effect is still captured, just indirectly, folded into the model's intercept.
print("\nOne-hot encoding categorical columns (dropping one reference category per column)...")
before_cols = model_df.shape[1]
model_df = pd.get_dummies(model_df, columns=categorical_features, drop_first=True)
# pd.get_dummies produces True/False columns in recent pandas versions -- convert to 0/1 integers,
# which is what the model actually expects and is easier to read in printed output.
new_dummy_cols = [c for c in model_df.columns if model_df[c].dtype == bool]
model_df[new_dummy_cols] = model_df[new_dummy_cols].astype(int)
print(f"Columns before encoding: {before_cols}  ->  after encoding: {model_df.shape[1]} "
      f"(each category, minus one reference per column, became its own 0/1 column)")

# ---------------------------------------------------------------------------------------------
# 5b. Split into training data and test data
# ---------------------------------------------------------------------------------------------
# We hold out a TEST set -- data the model will NEVER see while being fit -- so that when we
# measure its accuracy later (Step 7), we're measuring how it performs on genuinely new
# applicants, not how well it memorized the data it was trained on. Evaluating on the same data
# a model was trained on is one of the most common beginner mistakes in modeling: it always makes
# the model look better than it actually is.
#
# stratify=y ensures both the training set and the test set keep the SAME Approve/Reject ratio
# (roughly 74%/26%, from Step 2) as the full dataset. Without this, a plain random split could by
# chance put unusually few Rejected examples in the test set, making the evaluation in Step 7
# noisier and less trustworthy than it needs to be.
X = model_df.drop(columns=[target])
y = model_df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=SEED, stratify=y
)
print(f"\nTrain set: {X_train.shape[0]} rows   Test set: {X_test.shape[0]} rows "
      f"({X_test.shape[0] / len(model_df):.0%} held out)")
print(f"Approve rate in train set: {y_train.mean():.1%}   in test set: {y_test.mean():.1%}  "
      f"(should be nearly identical, because of stratify=y)")

# ---------------------------------------------------------------------------------------------
# 5c. Scale the numeric columns -- fit on TRAIN ONLY (this is the important part)
# ---------------------------------------------------------------------------------------------
# StandardScaler transforms a column so it has mean 0 and standard deviation 1 (this is called
# "standardization"). WHY logistic regression needs this: it fits its coefficients using an
# optimization process that's sensitive to the scale of each input; a column measured in tens of
# thousands (income) next to a column measured in single digits (a count) can make that
# optimization slower and, worse, makes the resulting coefficients impossible to compare to judge
# which feature matters more -- scaling puts every numeric feature on the same footing first.
#
# We only scale the genuinely continuous numeric columns -- NOT the 0/1 flag and dummy columns
# created in Steps 3 and 5a. Those are already on a clean, directly-interpretable 0-to-1 scale;
# "scaling" a column that's already just 0s and 1s would only relabel them as confusing decimal
# numbers without adding anything useful.
#
# THE CRITICAL RULE: .fit() is called on X_train ONLY, never on the full dataset and never
# including X_test. WHY this matters so much: StandardScaler's "mean" and "standard deviation" are
# themselves just statistics computed from data. If we computed them using the test set too, then
# information about the test set (even just its average) would leak into the numbers the model is
# trained on -- the model would be tested on data that subtly influenced its own preparation. This
# is called DATA LEAKAGE, and it makes a model's test performance look better than it will actually
# be on truly new, real-world applicants who were never part of any "let's compute a mean" step at
# all. .transform() (not .fit_transform()) is then applied to X_test, reusing the training set's
# already-learned mean and standard deviation, unchanged.
cols_to_scale = [c for c in numeric_features]  # the categorical dummy/flag columns are excluded by construction

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])   # fit AND transform on train
X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])          # transform ONLY on test

print(f"\nScaled {len(cols_to_scale)} numeric columns.")
if "NETMONTHLYINCOME" in cols_to_scale:
    print("Example -- NETMONTHLYINCOME before/after scaling (first 3 training rows):")
    print("  before:", X_train["NETMONTHLYINCOME"].head(3).tolist())
    print("  after: ", X_train_scaled["NETMONTHLYINCOME"].head(3).round(3).tolist())
else:
    print("(NETMONTHLYINCOME wasn't in the selected feature list, so no example to show here.)")

# ---------------------------------------------------------------------------------------------
# 5d. Save everything the next steps need
# ---------------------------------------------------------------------------------------------
train_out = X_train_scaled.copy()
train_out[target] = y_train.values
test_out = X_test_scaled.copy()
test_out[target] = y_test.values

train_out.to_csv(TRAIN_OUT_PATH, index=False)
test_out.to_csv(TEST_OUT_PATH, index=False)
joblib.dump(scaler, SCALER_OUT_PATH)

# The exact list and ORDER of final columns matters for Step 8 -- the unseen data must be encoded
# into a table with these exact columns, in this exact order, or the model will silently apply the
# wrong coefficient to the wrong column.
final_columns = X_train_scaled.columns.tolist()
with open(FINAL_COLUMNS_OUT_PATH, "w") as f:
    json.dump({"final_feature_columns": final_columns, "scaled_columns": cols_to_scale}, f, indent=2)

print(f"\nSaved: {TRAIN_OUT_PATH.name}, {TEST_OUT_PATH.name}, {SCALER_OUT_PATH.name}, "
      f"{FINAL_COLUMNS_OUT_PATH.name}")
print("\nDone with Step 5. Next: 06_train_model.py")
