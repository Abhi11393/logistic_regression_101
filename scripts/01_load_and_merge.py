 """
STEP 1 of 8: Load the raw data and merge it into one table.

WHAT this step does:
    Read the two source Excel files and combine them into a single table, one row per applicant.

WHY this step exists, explained from zero:
    Real credit-risk data at an NBFC/bank almost never lives in one file. Here we have exactly the
    two-source pattern that's realistic: the bank's OWN records about the applicant (loans they
    already know about -- "Internal_Bank_Dataset"), and a report pulled from India's CIBIL credit
    bureau about that same applicant's credit history at OTHER lenders too
    ("External_Cibil_Dataset" -- this is the "bureau data" you mentioned). A model that only saw
    the bank's own data would be blind to a person's debts everywhere else; a model that only saw
    bureau data would miss the bank's own relationship history with that person. We need both,
    joined together, before we can do anything else.

WHAT you'll learn in this step:
    - How to read an Excel file into a pandas DataFrame (pd.read_excel).
    - What a "join" (merge) is, and why we check the join key BEFORE trusting the result, instead
      of assuming it will "just work."
"""

import pandas as pd

# ---------------------------------------------------------------------------------------------
# 1a. Where the data lives, and where we'll write our output
# ---------------------------------------------------------------------------------------------
# Using a path relative to this script (not a hardcoded personal folder) so this project still
# works if you move the whole "logistic_regression_101" folder somewhere else.
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data"

CIBIL_PATH = RAW_DIR / "External_Cibil_Dataset.xlsx"
BANK_PATH = RAW_DIR / "Internal_Bank_Dataset.xlsx"
MERGED_OUT_PATH = OUT_DIR / "01_merged.csv"

print("=" * 90)
print("STEP 1: Load and merge")
print("=" * 90)

# ---------------------------------------------------------------------------------------------
# 1b. Load both files
# ---------------------------------------------------------------------------------------------
# pd.read_excel needs to know WHICH sheet inside the Excel file to read (an .xlsx file can contain
# more than one sheet, like tabs at the bottom of an Excel window). We checked ahead of time
# (see the exploration done before this script was written) that each file has exactly one sheet
# that matters: "case_study2" for the bureau data, "case_study1" for the bank data.
cibil = pd.read_excel(CIBIL_PATH, sheet_name="case_study2")
bank = pd.read_excel(BANK_PATH, sheet_name="case_study1")

print(f"\nExternal_Cibil_Dataset (bureau data): {cibil.shape[0]} rows, {cibil.shape[1]} columns")
print(f"Internal_Bank_Dataset (bank's own data): {bank.shape[0]} rows, {bank.shape[1]} columns")

# ---------------------------------------------------------------------------------------------
# 1c. Before merging: check that the join key actually lines up -- don't assume it
# ---------------------------------------------------------------------------------------------
# Both files have a column called PROSPECTID -- this is the applicant's unique ID, and it's the
# only thing that lets us say "this row in the bureau file and that row in the bank file are
# talking about the SAME person." Before trusting a merge on this column, we check three things
# a real analyst should always check first:
#   1. Is PROSPECTID actually unique within each file? (If not, a merge could accidentally
#      multiply rows -- one bureau row matching multiple bank rows for the "same" ID.)
#   2. Do the exact same IDs appear in both files? (If not, some applicants would be dropped, or
#      would end up with missing bureau/bank data after the merge, and we'd want to know that.)
n_cibil_unique_ids = cibil["PROSPECTID"].nunique()
n_bank_unique_ids = bank["PROSPECTID"].nunique()
print(f"\nUnique PROSPECTID count -- bureau file: {n_cibil_unique_ids} (out of {len(cibil)} rows)")
print(f"Unique PROSPECTID count -- bank file:   {n_bank_unique_ids} (out of {len(bank)} rows)")

ids_in_both = set(cibil["PROSPECTID"]) & set(bank["PROSPECTID"])
ids_only_in_cibil = set(cibil["PROSPECTID"]) - set(bank["PROSPECTID"])
ids_only_in_bank = set(bank["PROSPECTID"]) - set(cibil["PROSPECTID"])
print(f"\nIDs present in BOTH files: {len(ids_in_both)}")
print(f"IDs only in the bureau file (no matching bank record): {len(ids_only_in_cibil)}")
print(f"IDs only in the bank file (no matching bureau record): {len(ids_only_in_bank)}")

if n_cibil_unique_ids == len(cibil) and n_bank_unique_ids == len(bank) and not ids_only_in_cibil and not ids_only_in_bank:
    print("\n-> PROSPECTID is unique in both files, and every ID appears in both.")
    print("-> Safe to do a simple merge -- no applicant will be duplicated or silently dropped.")
else:
    print("\n-> WARNING: the join key is not a clean 1-to-1 match. Investigate before merging --")
    print("   proceeding anyway would risk duplicated rows or silently losing applicants.")

# ---------------------------------------------------------------------------------------------
# 1d. Merge the two tables into one
# ---------------------------------------------------------------------------------------------
# pd.merge(left, right, on=..., how=...) is pandas' version of a SQL JOIN.
#   - on="PROSPECTID" tells pandas which column identifies "the same row" in both tables.
#   - how="inner" keeps only rows whose PROSPECTID exists in BOTH tables. We just confirmed above
#     that every ID already exists in both, so "inner" here is a safe choice that won't drop
#     anyone -- but we're stating it explicitly rather than leaving it to pandas' default, because
#     the choice of "inner" vs "left" vs "outer" is a real decision (see the pandas lesson in
#     python_learning/day07_pandas_advanced if you want the fuller explanation of each option).
merged = pd.merge(cibil, bank, on="PROSPECTID", how="inner")

print(f"\nMerged table: {merged.shape[0]} rows, {merged.shape[1]} columns")
expected_rows = len(cibil)  # since every ID matched, we expect to keep every original row
if merged.shape[0] == expected_rows:
    print(f"-> Row count matches the source data ({expected_rows} rows) -- no applicants were lost or duplicated.")
else:
    print(f"-> WARNING: expected {expected_rows} rows but got {merged.shape[0]} -- something is off, investigate.")

# ---------------------------------------------------------------------------------------------
# 1e. Save the merged table so the next script can pick up from here
# ---------------------------------------------------------------------------------------------
# Every script in this project follows the same shape: load the PREVIOUS script's saved output,
# do one clearly-scoped thing, save ITS output for the next script. This means you can open any
# single script and understand it without needing to trace through the other seven at the same
# time -- and you can re-run just one step (e.g. re-run cleaning without re-running the merge)
# whenever you're experimenting.
merged.to_csv(MERGED_OUT_PATH, index=False)
print(f"\nSaved merged data to: {MERGED_OUT_PATH}")
print("\nDone with Step 1. Next: 02_explore_data.py")
