"""
STEP 3 of 8: Clean the data, using exactly the problems found in Step 2.

WHAT this step does:
    1. Turn the -99999 sentinel into real, honest missing values (NaN).
    2. Decide, column by column, whether to DROP a column (too much of it is missing to trust) or
       KEEP it and fill the gaps (impute).
    3. Fix the income outliers found in Step 2.
    4. Build the binary target column we'll actually model: Approved_Binary.

WHY this step exists, explained from zero:
    "Missing data" isn't one problem with one fix -- a column that's 2% missing and a column
    that's 93% missing need completely different treatment, and treating them the same way (e.g.
    "just fill everything with the average") would be a mistake in both directions: for the 2%
    column, imputing is safe and barely changes anything; for the 93% column, imputing would mean
    inventing values for almost the entire column, which creates a column that looks complete but
    contains almost no real information.

WHAT you'll learn in this step:
    - The concept of imputation (filling in a missing value with a stand-in, usually the median).
    - Why we track WHICH rows were imputed with a separate "_was_missing" flag column, instead of
      just silently filling the gap and moving on.
    - Winsorization (capping extreme values instead of deleting the rows).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MERGED_PATH = DATA_DIR / "01_merged.csv"
CLEANED_OUT_PATH = DATA_DIR / "03_cleaned.csv"
RECIPE_OUT_PATH = DATA_DIR / "03_cleaning_recipe.json"

# We will fill this dictionary in as we make each decision below, then save it to a JSON file at
# the very end. WHY: Step 8 of this project scores 100 brand-new applicants who were never part of
# training. Those new applicants must be cleaned using the EXACT SAME numbers (the same medians,
# the same drop list, the same income cap) that we compute here from the training data -- NOT
# freshly recomputed medians from the new 100 rows, which could easily be a different, misleading
# set of numbers (100 rows is a small, possibly unrepresentative sample). Saving this "recipe" now
# is what makes that later step consistent and honest instead of accidentally cheating.
cleaning_recipe = {}

print("=" * 90)
print("STEP 3: Clean the data")
print("=" * 90)

df = pd.read_csv(MERGED_PATH)
n_rows = len(df)

# ---------------------------------------------------------------------------------------------
# 3a. Replace the -99999 sentinel with real NaN, everywhere it appears
# ---------------------------------------------------------------------------------------------
# Once every -99999 becomes NaN, pandas' normal missing-value tools (.isna(), .fillna(), etc.)
# will finally see these cells as missing -- before this line, pandas treated -99999 as a real,
# valid number, which is exactly the danger Step 2 flagged.
SENTINEL = -99999
numeric_cols = df.select_dtypes(include="number").columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "PROSPECTID"]  # an ID column, not a real feature

n_sentinel_replaced = 0
for col in numeric_cols:
    mask = df[col] == SENTINEL
    n_sentinel_replaced += mask.sum()
    df.loc[mask, col] = np.nan

print(f"\nReplaced {n_sentinel_replaced} sentinel ({SENTINEL}) cells with NaN across {len(numeric_cols)} numeric columns.")

# ---------------------------------------------------------------------------------------------
# 3b. Decide, per column, whether to drop or impute -- using the actual missing percentage
# ---------------------------------------------------------------------------------------------
# The rule we're using, stated explicitly so it can be questioned/changed later:
#   - More than 70% missing  -> DROP the column. Imputing a value for 70%+ of a column means the
#     column would mostly consist of a single made-up number -- at that point it can't meaningfully
#     help the model separate approved from rejected applicants, and keeping it just adds noise
#     (and a false impression of completeness).
#   - 70% or less missing    -> KEEP the column. Add a "<column>_was_missing" flag (1/0) BEFORE
#     filling the gap, so the model can still use "this value was originally unknown" as its own
#     signal if that itself turns out to matter (exactly the same "is missingness informative"
#     check done in the vehicle_loan_default_risk project) -- then fill the gap with the column's
#     MEDIAN (not the mean, since a few extreme outliers -- see Step 2's income findings -- can
#     drag a mean far from what's "typical," while the median is resistant to that).
DROP_THRESHOLD = 0.70

missing_pct = df[numeric_cols].isna().mean().sort_values(ascending=False)
cols_to_drop = missing_pct[missing_pct > DROP_THRESHOLD].index.tolist()
cols_to_impute = missing_pct[(missing_pct > 0) & (missing_pct <= DROP_THRESHOLD)].index.tolist()

print(f"\nColumns to DROP (>{DROP_THRESHOLD:.0%} missing): {len(cols_to_drop)}")
for col in cols_to_drop:
    print(f"  {col:<32} {missing_pct[col]:.1%} missing")

print(f"\nColumns to IMPUTE (some missing, but <= {DROP_THRESHOLD:.0%}): {len(cols_to_impute)}")
for col in cols_to_impute:
    print(f"  {col:<32} {missing_pct[col]:.1%} missing")

df = df.drop(columns=cols_to_drop)
cleaning_recipe["sentinel_value"] = SENTINEL
cleaning_recipe["columns_dropped"] = cols_to_drop
cleaning_recipe["impute_medians"] = {}  # filled in below, one column at a time

for col in cols_to_impute:
    flag_col = f"{col}_was_missing"
    df[flag_col] = df[col].isna().astype(int)
    median_value = df[col].median()
    df[col] = df[col].fillna(median_value)
    cleaning_recipe["impute_medians"][col] = median_value

print(f"\nAfter dropping and imputing: {df.shape[1]} columns remain "
      f"(added {len(cols_to_impute)} new '_was_missing' flag columns).")

# ---------------------------------------------------------------------------------------------
# 3c. Fix the income outliers found in Step 2
# ---------------------------------------------------------------------------------------------
# Two separate problems, two separate fixes:
#   1. Implausibly LOW income (< 1000) -- almost certainly a data-entry error, not a real
#      applicant earning a few hundred rupees a month while applying for credit. We treat these
#      as missing and impute the median, same logic as 3b.
#   2. Extremely HIGH income (top 1%) -- these might be genuine (some applicants really do earn a
#      lot), but a handful of extreme values can disproportionately swing a logistic regression's
#      fitted line, since it's fitting a straight line in log-odds space across ALL the data. We
#      WINSORIZE instead of deleting: cap any value above the 99th percentile AT the 99th
#      percentile, rather than removing those rows entirely -- this keeps the applicant in the
#      dataset (their other features are still useful) while limiting how much leverage one
#      extreme number can have.
low_income_mask = df["NETMONTHLYINCOME"] < 1000
print(f"\nTreating {low_income_mask.sum()} rows with implausibly low income (< 1000) as missing.")
df.loc[low_income_mask, "NETMONTHLYINCOME"] = np.nan
income_median = df["NETMONTHLYINCOME"].median()
df["NETMONTHLYINCOME"] = df["NETMONTHLYINCOME"].fillna(income_median)
print(f"Imputed with the median income: {income_median:,.0f}")

income_cap = df["NETMONTHLYINCOME"].quantile(0.99)
n_capped = (df["NETMONTHLYINCOME"] > income_cap).sum()
df["NETMONTHLYINCOME"] = df["NETMONTHLYINCOME"].clip(upper=income_cap)
print(f"Winsorized {n_capped} rows with income above the 99th percentile ({income_cap:,.0f}) -- "
      f"capped, not removed.")
cleaning_recipe["income_low_cutoff"] = 1000
cleaning_recipe["income_median"] = income_median
cleaning_recipe["income_cap_p99"] = income_cap

# ---------------------------------------------------------------------------------------------
# 3d. Build the target we'll actually model: Approved_Binary
# ---------------------------------------------------------------------------------------------
# We're choosing Approved = 1, Rejected = 0. This choice is arbitrary (we could have picked the
# opposite) but it MUST be explicit and consistent for every later step, because every coefficient
# and every probability the model produces will be "the probability of the class labeled 1" --
# get this backwards later and every interpretation flips.
df["Approved_Binary"] = df["Approved_Flag"].isin(["P1", "P2"]).astype(int)
print(f"\nApproved_Binary target created: 1 = Approve (was P1/P2), 0 = Reject (was P3/P4).")
print(df["Approved_Binary"].value_counts())

# We can now drop the original 4-class column -- Approved_Binary is what every remaining script
# will use as the target.
df = df.drop(columns=["Approved_Flag"])

# ---------------------------------------------------------------------------------------------
# 3e. Save
# ---------------------------------------------------------------------------------------------
df.to_csv(CLEANED_OUT_PATH, index=False)
print(f"\nSaved cleaned data to: {CLEANED_OUT_PATH}  (shape: {df.shape})")


def _to_plain_python(value):
    """JSON doesn't know how to write numpy's number types (np.float64, etc.) -- this converts
    every value in the recipe to a plain Python float/int/str first so json.dump doesn't error."""
    if isinstance(value, dict):
        return {k: _to_plain_python(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain_python(v) for v in value]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


with open(RECIPE_OUT_PATH, "w") as f:
    json.dump(_to_plain_python(cleaning_recipe), f, indent=2)
print(f"Saved the cleaning recipe (drop list + medians + income cap) to: {RECIPE_OUT_PATH}")
print("-> Step 8 will load this exact file to clean the 100 unseen applicants the same way.")

print("\nDone with Step 3. Next: 04_feature_selection.py")
