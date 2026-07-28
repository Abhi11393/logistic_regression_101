"""
STEP 7 of 8: Evaluate the model honestly, on data it never saw during training.

WHAT this step does:
    1. Use the trained model to predict on the TEST set (held out since Step 5).
    2. Build a confusion matrix BY HAND from the raw predictions, and compute accuracy, precision,
       recall, and F1 directly from it -- then confirm scikit-learn's built-in functions agree.
    3. Plot an ROC curve and compute AUC.
    4. Compare the model against a "naive baseline" to prove it's actually adding value.

WHY this step exists, explained from zero:
    A single "accuracy" number can be dangerously misleading on its own -- Section 7d below shows
    exactly why, with real numbers from this dataset. Computing the confusion matrix and its
    derived metrics BY HAND first (before calling the sklearn shortcut) is deliberate: it proves
    where every number actually comes from, instead of trusting a function you can't yet verify
    yourself.

WHAT you'll learn in this step:
    - The confusion matrix: True Positives, False Positives, True Negatives, False Negatives.
    - Accuracy, Precision, Recall, and F1 -- their exact formulas, and when each one matters most.
    - ROC curve and AUC -- what they measure that a single accuracy number can't.
    - Why comparing against a naive baseline is a necessary sanity check, not an optional extra.
"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                              recall_score, roc_auc_score, roc_curve)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
TEST_PATH = DATA_DIR / "05_test.csv"
MODEL_PATH = OUTPUT_DIR / "06_logistic_regression_model.joblib"

target = "Approved_Binary"

print("=" * 90)
print("STEP 7: Evaluate the model")
print("=" * 90)

test_df = pd.read_csv(TEST_PATH)
X_test = test_df.drop(columns=[target])
y_test = test_df[target]
model = joblib.load(MODEL_PATH)

# ---------------------------------------------------------------------------------------------
# 7a. Predict on the test set
# ---------------------------------------------------------------------------------------------
# predict_proba gives the model's actual output: a probability for each class. predict_proba(X)
# returns TWO columns (probability of class 0, probability of class 1); we keep column 1, the
# probability of Approval, since that's the outcome we defined as "1" back in Step 3.
predicted_probability = model.predict_proba(X_test)[:, 1]

# To turn a probability into an actual yes/no decision, we need a THRESHOLD. 0.5 is the default,
# most natural starting point: "predict Approve if the model thinks Approval is more likely than
# not." Section 7e revisits whether 0.5 is really the right threshold for THIS business problem.
THRESHOLD = 0.5
predicted_class = (predicted_probability >= THRESHOLD).astype(int)

print(f"\nPredicted on {len(X_test)} test applicants never seen during training.")
print(f"Using a decision threshold of {THRESHOLD}: predict Approve if predicted probability >= {THRESHOLD}.")

# ---------------------------------------------------------------------------------------------
# 7b. Build the confusion matrix BY HAND
# ---------------------------------------------------------------------------------------------
# Four numbers, each counting one specific combination of (what actually happened) x (what we
# predicted):
#   True Positive  (TP): actually Approved (1), we predicted Approve (1)  -- correct
#   True Negative  (TN): actually Rejected (0), we predicted Reject  (0)  -- correct
#   False Positive (FP): actually Rejected (0), we predicted Approve (1) -- WRONG (a costly miss:
#                         we'd approve someone who should have been rejected)
#   False Negative (FN): actually Approved (1), we predicted Reject  (0) -- WRONG (a different
#                         kind of miss: we'd reject someone who should have been approved)
y_test_array = y_test.to_numpy()
TP = int(((predicted_class == 1) & (y_test_array == 1)).sum())
TN = int(((predicted_class == 0) & (y_test_array == 0)).sum())
FP = int(((predicted_class == 1) & (y_test_array == 0)).sum())
FN = int(((predicted_class == 0) & (y_test_array == 1)).sum())

print(f"\nConfusion matrix, computed by hand:")
print(f"  True Positives  (predicted Approve, actually Approved): {TP}")
print(f"  True Negatives  (predicted Reject,  actually Rejected): {TN}")
print(f"  False Positives (predicted Approve, actually Rejected): {FP}   <- costly: bad risk let through")
print(f"  False Negatives (predicted Reject,  actually Approved): {FN}   <- costly: good applicant turned away")
print(f"  Total: {TP + TN + FP + FN}  (should equal the test set size, {len(y_test)})")

# Confirm against scikit-learn's own confusion_matrix function -- it should produce identical
# numbers, just arranged as a small 2x2 grid instead of four separate named variables.
sk_cm = confusion_matrix(y_test, predicted_class)
print(f"\nscikit-learn's confusion_matrix (rows=actual [0,1], columns=predicted [0,1]):\n{sk_cm}")
print(f"Matches our by-hand counts: TN={sk_cm[0,0]}, FP={sk_cm[0,1]}, FN={sk_cm[1,0]}, TP={sk_cm[1,1]}")

# ---------------------------------------------------------------------------------------------
# 7c. Accuracy, Precision, Recall, F1 -- computed by hand, then confirmed with sklearn
# ---------------------------------------------------------------------------------------------
accuracy_by_hand = (TP + TN) / (TP + TN + FP + FN)
precision_by_hand = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall_by_hand = TP / (TP + FN) if (TP + FN) > 0 else 0.0
f1_by_hand = 2 * (precision_by_hand * recall_by_hand) / (precision_by_hand + recall_by_hand)

print(f"\nACCURACY = (TP + TN) / total = ({TP} + {TN}) / {len(y_test)} = {accuracy_by_hand:.4f}")
print("  -> \"Out of everyone, what fraction did we classify correctly, Approve or Reject?\"")

print(f"\nPRECISION = TP / (TP + FP) = {TP} / ({TP} + {FP}) = {precision_by_hand:.4f}")
print("  -> \"Of everyone we PREDICTED would be Approved, what fraction actually were?\"")
print("     A false positive is the mistake precision punishes: approving someone who shouldn't be.")

print(f"\nRECALL = TP / (TP + FN) = {TP} / ({TP} + {FN}) = {recall_by_hand:.4f}")
print("  -> \"Of everyone who was ACTUALLY Approved, what fraction did we correctly catch?\"")
print("     A false negative is the mistake recall punishes: rejecting someone who deserved approval.")

print(f"\nF1 = 2 * (precision * recall) / (precision + recall) = {f1_by_hand:.4f}")
print("  -> A single number balancing precision and recall -- useful when you care about both")
print("     mistakes and don't want to optimize one at the complete expense of the other.")

# Confirm against scikit-learn:
print(f"\nscikit-learn confirms: accuracy={accuracy_score(y_test, predicted_class):.4f}  "
      f"precision={precision_score(y_test, predicted_class):.4f}  "
      f"recall={recall_score(y_test, predicted_class):.4f}  "
      f"f1={f1_score(y_test, predicted_class):.4f}")

# ---------------------------------------------------------------------------------------------
# 7d. Why accuracy alone can be misleading -- a concrete demonstration on THIS data
# ---------------------------------------------------------------------------------------------
# Recall from Step 2: about 74% of applicants are actually Approved. A "model" that does zero
# actual thinking and just predicts "Approve" for EVERY single applicant would already score:
naive_accuracy = y_test.mean()
print(f"\nA naive 'always predict Approve' baseline would score {naive_accuracy:.1%} accuracy on")
print(f"this test set -- just by exploiting the fact that Approve is the majority class, with zero")
print(f"real predictive skill. Our model's {accuracy_by_hand:.1%} needs to be judged AGAINST this")
print(f"baseline, not in isolation -- and its precision/recall (which the naive baseline would score")
print(f"very differently on, since it produces zero True Negatives) are what actually reveal whether")
print(f"it's doing real work.")

naive_TP = int((y_test_array == 1).sum())
naive_FN = 0
naive_precision = naive_TP / len(y_test)
naive_recall = 1.0
print(f"Naive baseline: precision={naive_precision:.4f}  recall={naive_recall:.4f}  "
      f"(recall is a perfect 1.0 only because it never predicts Reject at all -- not because")
print(f" it's good at finding approvable applicants)")

# ---------------------------------------------------------------------------------------------
# 7e. ROC curve and AUC -- evaluating across EVERY possible threshold, not just 0.5
# ---------------------------------------------------------------------------------------------
# Every metric above depended on our choice of THRESHOLD = 0.5. What if 0.5 isn't the best cutoff
# for this business problem? The ROC (Receiver Operating Characteristic) curve sidesteps that
# question: it plots the True Positive Rate (= recall) against the False Positive Rate
# (FP / (FP + TN), "of everyone actually Rejected, what fraction did we wrongly Approve") at EVERY
# possible threshold from 0 to 1, all at once.
# AUC (Area Under that Curve) compresses the whole curve into one number: the probability that,
# if you randomly picked one truly-Approved applicant and one truly-Rejected applicant, the model
# would assign a HIGHER predicted probability to the Approved one. AUC = 1.0 is a perfect model;
# AUC = 0.5 is exactly what random guessing would achieve.
fpr, tpr, thresholds = roc_curve(y_test, predicted_probability)
auc = roc_auc_score(y_test, predicted_probability)
print(f"\nAUC (Area Under the ROC Curve): {auc:.4f}")
print(f"  -> Interpretation: {auc:.1%} chance the model ranks a random Approved applicant above a")
print(f"     random Rejected applicant. 0.50 = random guessing, 1.00 = perfect separation.")

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(fpr, tpr, color="darkorange", linewidth=2, label=f"Model (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random guessing (AUC = 0.5)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate (Recall)")
ax.set_title("ROC Curve -- Logistic Regression on Test Set")
ax.legend()
fig.savefig(OUTPUT_DIR / "07_roc_curve.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Saved ROC curve plot to: {OUTPUT_DIR / '07_roc_curve.png'}")

# ---------------------------------------------------------------------------------------------
# 7f. Save a summary of everything computed in this step
# ---------------------------------------------------------------------------------------------
summary = pd.DataFrame([{
    "threshold": THRESHOLD, "TP": TP, "TN": TN, "FP": FP, "FN": FN,
    "accuracy": accuracy_by_hand, "precision": precision_by_hand, "recall": recall_by_hand,
    "f1": f1_by_hand, "auc": auc, "naive_baseline_accuracy": naive_accuracy,
}])
summary.to_csv(OUTPUT_DIR / "07_evaluation_summary.csv", index=False)
print(f"Saved evaluation summary to: {OUTPUT_DIR / '07_evaluation_summary.csv'}")
print("\nDone with Step 7. Next: 08_predict_unseen.py")
