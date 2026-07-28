"""
STEP 6 of 8: Understand what logistic regression actually IS, then fit one.

WHAT this step does:
    Walk through the mathematics of logistic regression with real, tiny numbers first, THEN fit
    scikit-learn's LogisticRegression on the training data, and finally read and interpret what it
    learned.

WHY this step exists, explained from zero:
    Everything up to this point was preparing ingredients. This is the step you actually asked
    for -- but "just call .fit()" would skip over what that single function call is doing, and you
    said not to assume anything, so this script spends real time on the mechanics BEFORE running
    the one line that does the fitting.

WHAT you'll learn in this step:
    - The sigmoid function -- the mathematical core of logistic regression -- worked out by hand
      on a toy example.
    - Log-odds, and why logistic regression is really "linear regression on log-odds," not on the
      probability directly.
    - What .fit() is actually searching for (the coefficients that minimize log-loss).
    - How to read a fitted coefficient as an odds ratio.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
TRAIN_PATH = DATA_DIR / "05_train.csv"
MODEL_OUT_PATH = OUTPUT_DIR / "06_logistic_regression_model.joblib"
COEFFICIENTS_OUT_PATH = OUTPUT_DIR / "06_coefficients.csv"

target = "Approved_Binary"
SEED = 42

print("=" * 90)
print("STEP 6: Understand and train the logistic regression model")
print("=" * 90)

# ---------------------------------------------------------------------------------------------
# 6a. The sigmoid function, worked out by hand on made-up numbers first
# ---------------------------------------------------------------------------------------------
# A logistic regression model computes ONE number first, called "z" (or the "log-odds", explained
# in 6b): z = intercept + (coef_1 * feature_1) + (coef_2 * feature_2) + ... for every feature.
# That z can be ANY real number -- from very negative to very positive -- but we need a
# PROBABILITY, which must be between 0 and 1. The sigmoid function is what converts z into a valid
# probability:
#
#     sigmoid(z) = 1 / (1 + e^(-z))
#
# Let's compute this by hand for a few example z values, so the shape of this function is
# concrete before we ever see it applied to real data:
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


print("\nThe sigmoid function applied to a few example z values:")
for z_example in [-4, -1, 0, 1, 4]:
    p = sigmoid(z_example)
    print(f"  z = {z_example:>3}  ->  sigmoid(z) = {p:.4f}   (a {p:.1%} probability)")
print("-> Notice: z=0 always maps to exactly 0.5 probability. Very negative z squashes toward 0.")
print("   Very positive z squashes toward 1. z can be any size in either direction, but sigmoid(z)")
print("   is ALWAYS strictly between 0 and 1 -- this is exactly why it's the right tool for turning")
print("   an unbounded formula into something that behaves like a real probability.")

# ---------------------------------------------------------------------------------------------
# 6b. Log-odds: why logistic regression is "linear regression on a transformed target"
# ---------------------------------------------------------------------------------------------
# "Odds" is a different way of expressing a probability, familiar from betting: if p is the
# probability of Approval, the odds of Approval are p / (1 - p). A 75% probability is "3 to 1"
# odds (0.75 / 0.25 = 3).
#
# The "log-odds" (also called the "logit") is just the natural log of the odds: ln(p / (1 - p)).
# The reason logistic regression is built around log-odds specifically: log-odds is the ONE
# transformation of a probability that can be modeled as a plain weighted sum of features (exactly
# like ordinary linear regression), stretching the bounded [0, 1] probability range out into the
# full, unbounded real number line -- which is exactly the "z" from 6a. Fitting the model means
# finding the coefficients that make this weighted sum best predict the log-odds of Approval; the
# sigmoid function is then just the one step that converts that back into an actual probability.
example_p = 0.75
example_odds = example_p / (1 - example_p)
example_log_odds = np.log(example_odds)
print(f"\nWorked example: a probability of {example_p} means odds of {example_odds:.1f} to 1, "
      f"and log-odds of {example_log_odds:.4f}.")
print(f"Sanity check -- running that log-odds back through sigmoid() should return us to {example_p}: "
      f"sigmoid({example_log_odds:.4f}) = {sigmoid(example_log_odds):.4f}")

# ---------------------------------------------------------------------------------------------
# 6c. What .fit() is actually doing
# ---------------------------------------------------------------------------------------------
print("\nWhat happens when we call .fit(): scikit-learn searches for the intercept and the one")
print("coefficient per feature that make the model's predicted probabilities match the REAL")
print("Approved_Binary outcomes in the training data as closely as possible. 'As closely as")
print("possible' is measured by a specific score called LOG-LOSS, which penalizes the model much")
print("more heavily for being confidently WRONG (e.g. predicting 95% Approve for someone who was")
print("actually Rejected) than for being cautiously wrong (predicting 55% Approve for someone who")
print("was actually Rejected). The search itself is an iterative numerical optimization -- you")
print("don't need to hand-run it, but that's what's happening inside the single line below.")

# ---------------------------------------------------------------------------------------------
# 6d. Actually fit the model
# ---------------------------------------------------------------------------------------------
train_df = pd.read_csv(TRAIN_PATH)
X_train = train_df.drop(columns=[target])
y_train = train_df[target]

# max_iter raises the cap on how many optimization steps scikit-learn is allowed before giving up
# -- the default (100) is sometimes not enough to fully converge on a dataset this size, and an
# unconverged model is a genuine, easy-to-miss problem (its coefficients would be reported as if
# final, but the search hadn't actually settled yet).
model = LogisticRegression(max_iter=2000, random_state=SEED)
model.fit(X_train, y_train)
print(f"\nModel fit on {X_train.shape[0]} training rows, {X_train.shape[1]} features.")
print(f"Did the optimizer converge? {'Yes' if model.n_iter_[0] < 2000 else 'NO -- see warning above, increase max_iter'} "
      f"(used {model.n_iter_[0]} of the 2000 allowed iterations)")

# ---------------------------------------------------------------------------------------------
# 6e. Read what the model learned
# ---------------------------------------------------------------------------------------------
# model.intercept_ is the model's baseline log-odds when every feature is at its "reference" value
# (for scaled numeric features, this is the average applicant; for one-hot columns, it's the
# category we dropped in Step 5). model.coef_ has one number per feature: how much that feature
# pushes the log-odds up (positive) or down (negative), holding every other feature fixed.
intercept = model.intercept_[0]
coefficients = pd.Series(model.coef_[0], index=X_train.columns)

# A raw coefficient (in log-odds units) is hard to build intuition for directly. Converting it to
# an ODDS RATIO via exp(coefficient) is much more readable: an odds ratio of 1.5 means "a one-unit
# increase in this feature multiplies the odds of Approval by 1.5" (a 50% increase in odds); an
# odds ratio of 0.5 means the odds are cut in half. An odds ratio of exactly 1.0 would mean the
# feature has no effect at all.
odds_ratios = np.exp(coefficients)

results = pd.DataFrame({"coefficient_log_odds": coefficients, "odds_ratio": odds_ratios})
results = results.reindex(results["coefficient_log_odds"].abs().sort_values(ascending=False).index)

print(f"\nIntercept (baseline log-odds): {intercept:.4f}  "
      f"(baseline probability if every feature were at its reference value: {sigmoid(intercept):.1%})")
print(f"\nAll {len(results)} coefficients, sorted by how strongly they influence the prediction "
      f"(largest absolute effect first):")
print(results.to_string(float_format=lambda x: f"{x:.4f}"))

print("\nHow to read a row of this table, using the top row as an example:")
top_feature = results.index[0]
top_or = results.loc[top_feature, "odds_ratio"]
direction = "increases" if top_or > 1 else "decreases"
print(f"  '{top_feature}': odds ratio = {top_or:.3f} -> a one-unit increase in this feature "
      f"{direction} the odds of Approval by a factor of {top_or:.3f}, holding every other feature "
      f"fixed. (Remember: for scaled numeric features, 'one unit' means one standard deviation, "
      f"because of the scaling done in Step 5 -- not one raw rupee/year/count.)")

# ---------------------------------------------------------------------------------------------
# 6f. Save the model and the coefficient table
# ---------------------------------------------------------------------------------------------
joblib.dump(model, MODEL_OUT_PATH)
results.to_csv(COEFFICIENTS_OUT_PATH)
print(f"\nSaved trained model to: {MODEL_OUT_PATH}")
print(f"Saved coefficient table to: {COEFFICIENTS_OUT_PATH}")
print("\nDone with Step 6. Next: 07_evaluate_model.py")
