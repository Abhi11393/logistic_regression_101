"""
STEP 4 of 8: Decide which columns are actually worth putting into the model.

WHAT this step does:
    1. For every CATEGORICAL column, run a Chi-square test of independence against the target --
       does knowing this column tell us anything about whether an applicant is Approved?
    2. For every NUMERIC column, run a t-test comparing its average value between the Approved
       group and the Rejected group -- do the two groups actually differ on this number?
    3. Among the numeric columns that survive, check for multicollinearity using VIF (Variance
       Inflation Factor) and remove redundant columns.

WHY this step exists, explained from zero:
    We currently have ~85 columns. Feeding all of them into a model "because more data can't
    hurt" is a common beginner assumption -- and it's wrong, for two concrete reasons this step
    addresses directly:
      (a) A column with NO real relationship to the target adds noise: the model may fit
          meaningless patterns in it (this is called overfitting), and it makes the model's
          coefficients harder to interpret.
      (b) Two columns that are highly correlated WITH EACH OTHER (not just with the target) cause
          a specific, well-documented problem for logistic regression called multicollinearity:
          the model can't tell which of the two redundant columns deserves credit for their shared
          effect, so their individual coefficients become unstable and can even flip to a
          nonsensical sign -- this is exactly the "sign flip" problem investigated in the
          vehicle_loan_default_risk project's scorecard, and VIF is the standard tool for
          catching it before it happens.

WHAT you'll learn in this step:
    - The Chi-square test of independence (for a categorical column vs. a categorical target).
    - The two-sample t-test (for a numeric column vs. a two-group target).
    - What a p-value actually means, and the 0.05 threshold convention.
    - VIF: the formula, how to read it, and the iterative removal procedure.
    - A real production-ML constraint: a model can only use, at prediction time, features that
      will actually be AVAILABLE for a brand-new applicant -- not just features that happened to
      be statistically useful in the historical data used to build it.
"""

import warnings
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# When a column is PERFECTLY predictable from the others (VIF's internal R-squared = 1.0 exactly),
# the VIF formula divides by (1 - 1.0) = 0, which numpy correctly flags as a RuntimeWarning before
# returning `inf`. That's not a bug -- it's the expected, mathematically correct signal for total
# redundancy -- so we suppress the low-level warning here and explain the `inf` result ourselves,
# in plain language, at the point where it's printed below instead.
warnings.filterwarnings("ignore", message="divide by zero encountered", category=RuntimeWarning)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CLEANED_PATH = DATA_DIR / "03_cleaned.csv"
UNSEEN_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "Unseen_Dataset.xlsx"
SELECTED_FEATURES_PATH = DATA_DIR / "04_selected_features.csv"

print("=" * 90)
print("STEP 4: Feature selection")
print("=" * 90)

df = pd.read_csv(CLEANED_PATH)
target = "Approved_Binary"

categorical_cols = df.select_dtypes(include="object").columns.tolist()
numeric_cols = [c for c in df.select_dtypes(include="number").columns if c not in ("PROSPECTID", target)]

print(f"\nStarting point: {len(categorical_cols)} categorical columns, {len(numeric_cols)} numeric columns.")

# ---------------------------------------------------------------------------------------------
# 4a-0. Restrict candidates to columns we'll actually be able to get for a NEW applicant
# ---------------------------------------------------------------------------------------------
# Step 8 of this project scores 100 brand-new applicants from Unseen_Dataset.xlsx. That file
# represents what a real "new application" would realistically provide: it does NOT include every
# column our training data has -- for example, it has no AGE and no Credit_Score at all. If we let
# Step 4 select a feature that Unseen_Dataset doesn't have, the model would learn to depend on
# information we can never actually supply when scoring a real new applicant later -- a model
# that's unusable in practice, even if it looks great on the historical test set.
#
# The fix: BEFORE running any significance test, we shrink the candidate list down to only the
# columns that also exist in Unseen_Dataset (this is exactly what a real ML team does when they
# define their "feature store" / "serving schema" up front). This does mean giving up some
# columns that might otherwise turn out to be statistically useful (AGE, most likely, among them)
# -- a real team facing this would push to get those fields captured for new applicants too,
# rather than silently dropping them; here, we treat Unseen_Dataset's schema as a fixed given.
unseen_columns = pd.read_excel(UNSEEN_RAW_PATH, nrows=0).columns.tolist()
print(f"\nUnseen_Dataset.xlsx (the 'brand new applicant' data Step 8 will score) has "
      f"{len(unseen_columns)} columns available.")


def available_for_new_applicants(column_name):
    # A "<something>_was_missing" flag column (created in Step 3) can only be reconstructed for a
    # new applicant if the RAW column it's based on is itself available -- so we check the
    # original name, not the flag's name, which never existed in the raw data at all.
    base_name = column_name.replace("_was_missing", "") if column_name.endswith("_was_missing") else column_name
    return base_name in unseen_columns


numeric_cols_before = len(numeric_cols)
categorical_cols_before = len(categorical_cols)
numeric_cols = [c for c in numeric_cols if available_for_new_applicants(c)]
categorical_cols = [c for c in categorical_cols if available_for_new_applicants(c)]

print(f"Numeric candidates usable for new applicants: {len(numeric_cols)} (of {numeric_cols_before})")
print(f"Categorical candidates usable for new applicants: {len(categorical_cols)} (of {categorical_cols_before})")
print("-> Every significance test below runs only on this already-narrowed, 'servable' candidate list.")

# ---------------------------------------------------------------------------------------------
# 4a. Chi-square test: does each CATEGORICAL column relate to the target?
# ---------------------------------------------------------------------------------------------
# The logic of this test, explained without assuming prior stats knowledge:
#   - We build a "contingency table": for a column like GENDER, this is a small grid counting
#     how many Male applicants were Approved, how many Male were Rejected, how many Female were
#     Approved, how many Female were Rejected.
#   - The NULL HYPOTHESIS (the "boring" assumption we're testing against) is: "GENDER and
#     Approved_Binary are independent -- knowing someone's gender tells you nothing about their
#     approval odds; any difference we happen to observe in this specific dataset is just random
#     noise."
#   - The Chi-square test measures how far the ACTUAL counts in our contingency table are from
#     what we'd EXPECT to see if that null hypothesis were exactly true, and converts that
#     distance into a p-value.
#   - The p-value is: "if the null hypothesis (no real relationship) were actually true, what's
#     the probability we'd see a difference at least this large just by chance?" A SMALL p-value
#     means "this would be a very unlikely coincidence" -- so we conclude the relationship is
#     probably real, not noise.
#   - Convention (used here, like in almost all applied statistics): p < 0.05 -- there'd be less
#     than a 5% chance of seeing this by pure coincidence -- counts as "statistically significant."
print("\n--- Chi-square test: categorical columns vs. Approved_Binary ---")
CHI2_ALPHA = 0.05
significant_categoricals = []
for col in categorical_cols:
    contingency_table = pd.crosstab(df[col], df[target])
    chi2_stat, p_value, degrees_of_freedom, expected_counts = chi2_contingency(contingency_table)
    decision = "KEEP (p < 0.05)" if p_value < CHI2_ALPHA else "DROP (p >= 0.05)"
    print(f"  {col:<20} chi2={chi2_stat:>9.2f}   p-value={p_value:.6f}   -> {decision}")
    if p_value < CHI2_ALPHA:
        significant_categoricals.append(col)

print(f"\nCategorical columns kept: {significant_categoricals}")

# ---------------------------------------------------------------------------------------------
# 4b. t-test: does each NUMERIC column differ between the Approved and Rejected groups?
# ---------------------------------------------------------------------------------------------
# Same idea as the Chi-square test, but for a NUMBER instead of a category:
#   - Split the column into two groups: its values for Approved applicants, and its values for
#     Rejected applicants.
#   - NULL HYPOTHESIS: "the AVERAGE value of this column is the same in both groups -- any
#     difference we see between the two group means is just random sampling noise."
#   - The t-test converts the difference between the two group means (relative to how spread out
#     each group is) into a p-value, with the same interpretation and the same 0.05 threshold as
#     above.
#   - We use Welch's t-test (equal_var=False) rather than the classic "Student's" version, because
#     Welch's version does NOT assume the two groups have equal variance -- with an Approved group
#     roughly 3x the size of the Rejected group, assuming equal variance would be an unjustified
#     shortcut, and Welch's test is the safer default whenever you're not sure.
print("\n--- t-test: numeric columns vs. Approved_Binary ---")
TTEST_ALPHA = 0.05
significant_numerics = []
approved_group = df[df[target] == 1]
rejected_group = df[df[target] == 0]
for col in numeric_cols:
    t_stat, p_value = ttest_ind(approved_group[col], rejected_group[col], equal_var=False)
    decision = "KEEP (p < 0.05)" if p_value < TTEST_ALPHA else "DROP (p >= 0.05)"
    print(f"  {col:<32} t={t_stat:>9.2f}   p-value={p_value:.6f}   -> {decision}")
    if p_value < TTEST_ALPHA:
        significant_numerics.append(col)

print(f"\nNumeric columns kept: {len(significant_numerics)} of {len(numeric_cols)}")

# ---------------------------------------------------------------------------------------------
# 4c. VIF: remove numeric columns that are redundant WITH EACH OTHER
# ---------------------------------------------------------------------------------------------
# VIF asks a different question than the t-test above. A column can be strongly related to the
# target AND still be a problem, if it's ALSO strongly related to another column we're keeping.
#
# The formula, explained: for a given feature X, VIF(X) = 1 / (1 - R-squared), where R-squared
# comes from a "helper" regression that tries to PREDICT X using every OTHER remaining feature.
#   - If X can't be predicted well from the other features (they're not measuring the same thing),
#     R-squared is low, so VIF is close to 1 -- no redundancy problem.
#   - If X CAN be predicted well from the other features (e.g. "Total_TL" and "Tot_Active_TL" +
#     "Tot_Closed_TL" are almost measuring the same underlying thing), R-squared approaches 1,
#     and VIF explodes toward infinity -- a severe redundancy problem.
# Common thresholds: VIF < 5 is conservative/safe, VIF > 10 is a clear problem. We use 10 here,
# and remove the SINGLE worst offender, then RECOMPUTE every remaining VIF from scratch (removing
# one column changes every other column's VIF, since "the other features" that column is being
# predicted from just changed) -- repeating until nothing is left above the threshold.
print("\n--- VIF: checking multicollinearity among the significant numeric columns ---")
VIF_THRESHOLD = 10.0
vif_candidates = significant_numerics.copy()

while True:
    X = add_constant(df[vif_candidates])  # VIF's underlying regression needs an intercept term
    vif_values = pd.Series(
        [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
        index=X.columns,
    )
    vif_values = vif_values.drop("const")  # the intercept's own VIF isn't meaningful, ignore it

    worst_col = vif_values.idxmax()
    worst_vif = vif_values.max()

    if worst_vif <= VIF_THRESHOLD:
        print(f"All remaining VIFs are <= {VIF_THRESHOLD} -- stopping.")
        break

    if worst_vif == float("inf"):
        print(f"  Highest VIF: {worst_col} = inf  -> this column is PERFECTLY predictable from the "
              f"others (R-squared = 1.0 exactly), meaning it carries zero information beyond what's "
              f"already captured elsewhere -- almost always because several '_was_missing' flag "
              f"columns come from fields that are always missing TOGETHER for the same applicants "
              f"(e.g. a whole 'enquiry history' sub-report is either present or absent as a group). "
              f"Dropping it, then recomputing.")
    else:
        print(f"  Highest VIF: {worst_col} = {worst_vif:.1f}  (> {VIF_THRESHOLD}) -> dropping it, then recomputing")
    vif_candidates.remove(worst_col)

print(f"\nFinal VIF table for the surviving {len(vif_candidates)} numeric columns:")
print(vif_values.sort_values(ascending=False).to_string())

final_numeric_features = vif_candidates
final_categorical_features = significant_categoricals

print(f"\nFinal feature set: {len(final_numeric_features)} numeric + {len(final_categorical_features)} "
      f"categorical = {len(final_numeric_features) + len(final_categorical_features)} total features "
      f"(down from {len(numeric_cols) + len(categorical_cols)} we started this step with).")

# ---------------------------------------------------------------------------------------------
# 4d. Save the decision -- which columns to carry forward
# ---------------------------------------------------------------------------------------------
selected = pd.DataFrame({
    "feature": final_numeric_features + final_categorical_features,
    "type": ["numeric"] * len(final_numeric_features) + ["categorical"] * len(final_categorical_features),
})
selected.to_csv(SELECTED_FEATURES_PATH, index=False)
print(f"\nSaved selected feature list to: {SELECTED_FEATURES_PATH}")
print("\nDone with Step 4. Next: 05_encode_scale_split.py")
