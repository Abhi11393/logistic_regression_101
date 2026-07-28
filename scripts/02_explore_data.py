"""
STEP 2 of 8: Explore the merged data BEFORE touching it.

WHAT this step does:
    Look carefully at the merged table -- its shape, its target column, its missing values, its
    outliers -- and write down exactly what's wrong with it, with real numbers. No cleaning
    happens in this script; this step is purely diagnostic.

WHY this step exists, explained from zero:
    A very common beginner mistake is to jump straight to "clean the data" using generic defaults
    (fill every gap with 0, drop every questionable row) without first checking what's actually
    THERE. Every cleaning decision made in Step 3 will be justified using a specific number
    discovered in THIS step -- "we're doing X because Y% of this column is Z" -- rather than a
    guess. This mirrors exactly how the vehicle_loan_default_risk and customer_segmentation
    projects were built: check first, decide second.

WHAT you'll learn in this step:
    - df.info(), df.describe(), and value_counts() as your first three moves on any new dataset.
    - How to detect a "sentinel" value (a fake placeholder number standing in for "missing," like
      -99999) -- these do NOT show up as normal missing values, and pandas will not warn you about
      them on its own.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MERGED_PATH = DATA_DIR / "01_merged.csv"

print("=" * 90)
print("STEP 2: Explore the data")
print("=" * 90)

df = pd.read_csv(MERGED_PATH)

# ---------------------------------------------------------------------------------------------
# 2a. The basics: how big is this, and what does each column look like?
# ---------------------------------------------------------------------------------------------
print(f"\nShape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nColumn data types (a quick way to see numeric vs text columns at a glance):")
print(df.dtypes.value_counts())
# .dtypes gives one dtype per column; .value_counts() on THAT tells us how many columns are each
# type (e.g. "55 columns are int64, 5 are float64, 6 are text (object)") -- a two-line way to get
# an overview of 88 columns without reading all of them individually.

# ---------------------------------------------------------------------------------------------
# 2b. The target column: what are we actually trying to predict?
# ---------------------------------------------------------------------------------------------
# Approved_Flag is the column that tells us what actually happened for each applicant: which of
# four priority segments (P1 = best, P4 = worst, in this dataset's convention) they were placed
# into. We are NOT changing this column yet (that happens in Step 3) -- just looking at it.
print("\nApproved_Flag (the raw, 4-class target) -- counts and percentages:")
counts = df["Approved_Flag"].value_counts()
percentages = df["Approved_Flag"].value_counts(normalize=True) * 100
for label in counts.index:
    print(f"  {label}: {counts[label]:>6} applicants ({percentages[label]:.1f}%)")

# We already decided (see the project's README) to collapse this into a binary target:
# Approve = P1 or P2, Reject = P3 or P4. Let's look at what that split would actually look like,
# still without changing anything yet:
approve_count = df["Approved_Flag"].isin(["P1", "P2"]).sum()
reject_count = df["Approved_Flag"].isin(["P3", "P4"]).sum()
print(f"\nIf collapsed to binary: Approve (P1+P2) = {approve_count} ({approve_count / len(df):.1%}), "
      f"Reject (P3+P4) = {reject_count} ({reject_count / len(df):.1%})")
print("-> Roughly a 74%/26% split. Not perfectly balanced, but not extreme either -- workable")
print("   for logistic regression without needing special imbalance handling.")

# ---------------------------------------------------------------------------------------------
# 2c. Real missing values (NaN) -- the kind pandas already knows about
# ---------------------------------------------------------------------------------------------
total_real_missing = df.isna().sum().sum()
print(f"\nTotal genuinely-missing (NaN) cells across the whole table: {total_real_missing}")
print("-> If this is 0, it does NOT mean the data has no missing values -- see the next check.")

# ---------------------------------------------------------------------------------------------
# 2d. The sentinel value -99999 -- missing values DISGUISED as real numbers
# ---------------------------------------------------------------------------------------------
# This is the single most important thing to catch in this dataset. Some numeric columns use the
# literal number -99999 to mean "not available / not applicable" instead of leaving the cell
# blank. pandas has NO way to know this on its own -- as far as pandas is concerned, -99999 is
# just a normal, valid, meaningful number in that column, exactly as legitimate as 5 or 200. If we
# don't catch this ourselves, every average, every model coefficient touching these columns would
# be wildly distorted by thousands of fake "-99999" data points pulling numbers down.
sentinel_value = -99999
numeric_cols = df.select_dtypes(include="number").columns
sentinel_counts = (df[numeric_cols] == sentinel_value).sum()
sentinel_counts = sentinel_counts[sentinel_counts > 0].sort_values(ascending=False)

print(f"\nColumns containing the sentinel value {sentinel_value} (count of affected rows, and as a "
      f"% of all {len(df)} rows):")
for col, count in sentinel_counts.items():
    print(f"  {col:<32} {count:>6} rows  ({count / len(df):.1%})")

print(f"\n-> {len(sentinel_counts)} columns contain this sentinel. Some are minor (a handful of "
      f"rows); a few are extreme -- e.g. CC_utilization and PL_utilization are MOSTLY sentinel. "
      f"Step 3 will decide, column by column, what to do about each one, using these exact "
      f"percentages.")

# ---------------------------------------------------------------------------------------------
# 2e. A specific, known real-world data-quality check: income
# ---------------------------------------------------------------------------------------------
print("\nNETMONTHLYINCOME summary statistics:")
print(df["NETMONTHLYINCOME"].describe())
very_low_income = (df["NETMONTHLYINCOME"] < 1000).sum()
very_high_income = (df["NETMONTHLYINCOME"] > df["NETMONTHLYINCOME"].quantile(0.99)).sum()
print(f"\nRows with income below 1000 (implausibly low for a loan applicant): {very_low_income}")
print(f"Rows above the 99th percentile of income (extreme outliers, e.g. the max of "
      f"{df['NETMONTHLYINCOME'].max():,.0f} against a median of {df['NETMONTHLYINCOME'].median():,.0f}): "
      f"{very_high_income}")
print("-> Both ends look like they need attention in Step 3: logistic regression fits a straight")
print("   line in log-odds space, so unlike a tree-based model, it's genuinely sensitive to a")
print("   small number of extreme values pulling that line off course.")

# ---------------------------------------------------------------------------------------------
# 2f. Categorical columns -- what values do they actually take?
# ---------------------------------------------------------------------------------------------
categorical_cols = df.select_dtypes(include="object").columns.tolist()
categorical_cols = [c for c in categorical_cols if c != "Approved_Flag"]  # that's the target, not a feature
print(f"\nCategorical (text) feature columns: {categorical_cols}")
for col in categorical_cols:
    print(f"\n{col} value counts:")
    print(df[col].value_counts())

print("\nDone with Step 2. Next: 03_clean_data.py")
