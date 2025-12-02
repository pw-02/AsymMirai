#!/usr/bin/env python3
# ============================================================
# Compute AUC for AsymMirai predictions using your input CSVs
# ============================================================

import pandas as pd
import numpy as np
import re
import sklearn.metrics
import matplotlib.pyplot as plt


# ============================================================
# Clean tensor outputs found in your predictions CSV
# ============================================================
def clean_tensor_value(x):
    """
    Converts strings like:
        tensor(42, device='cuda:0')
        tensor([1,2,3], device='cuda:0')
    Into Python numbers or numpy arrays.
    """
    if isinstance(x, (int, float)):
        return x

    if not isinstance(x, str):
        return x

    x = x.strip()

    # Case: scalar tensor e.g., tensor(42)
    scalar_match = re.findall(r"tensor\(([-0-9\.]+)", x)
    if scalar_match:
        return float(scalar_match[0])

    # Case: array tensor e.g., tensor([ 1, 2, ... ])
    if "tensor([" in x:
        arr_str = x.replace("tensor(", "").replace(")", "")
        arr_str = arr_str.split("device=")[0]
        try:
            return np.array(eval(arr_str))
        except Exception:
            return np.nan

    return x


# ============================================================
# Compute AUC using censoring logic (matches your notebook)
# ============================================================
def compute_asym_auc(df, followup_year=5):
    probs = df["prediction_pos"].values
    years_to_cancer = df["years_to_cancer"].values
    years_to_last_followup = df["years_to_last_followup"].values

    included_labels = []
    included_probs = []

    for p, ct, fup in zip(probs, years_to_cancer, years_to_last_followup):

        pos = ct <= followup_year        # had cancer within followup
        neg = fup >= followup_year       # cancer-free at least until followup

        # exclude censored cases
        if not (pos or neg):
            continue

        included_probs.append(p)
        included_labels.append(1 if pos else 0)

    # Not enough classes
    if len(set(included_labels)) < 2:
        return np.nan, [], []

    auc = sklearn.metrics.roc_auc_score(included_labels, included_probs)
    return auc, included_labels, included_probs


# ============================================================
# Plot ROC
# ============================================================
def plot_roc(labels, probs, title):
    fpr, tpr, _ = sklearn.metrics.roc_curve(labels, probs)
    plt.plot(fpr, tpr, label=title)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    # plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# ============================================================
# Main
# ============================================================
def main():

    print("\n=== Loading metadata ===")
    meta = pd.read_csv("data\\input_test.csv")

    # Need only these 3 fields to compute AUC
    if not {"exam_id", "years_to_cancer", "years_to_last_followup"}.issubset(meta.columns):
        raise ValueError("Your metadata CSV must contain exam_id, years_to_cancer, years_to_last_followup")

    print("Metadata loaded:", meta.shape)

    print("\n=== Loading AsymMirai predictions ===")
    preds = pd.read_csv("validation_predictions2.csv")
    print("Predictions loaded:", preds.shape)

    # Clean tensor columns
    tensor_cols = ["y_argmin_cc", "x_argmin_cc", "y_argmin_mlo", "x_argmin_mlo"]
    for c in tensor_cols:
        if c in preds.columns:
            preds[c] = preds[c].apply(clean_tensor_value)

    # Merge predictions ↔ metadata
    print("\n=== Merging predictions with metadata ===")
    merged = preds.merge(
        meta[["exam_id", "years_to_cancer", "years_to_last_followup"]],
        on="exam_id",
        how="inner"
    )
    print("Merged dataset:", merged.shape)

    auc_results = []

    print("\n=== Computing AUCs for follow-up years 1–5 ===")
    for year in range(1, 6):
        auc, labels, probs = compute_asym_auc(merged, followup_year=year)
        auc_results.append({"year": year, "auc": auc})
        print(f"Year {year} AUC = {auc:.4f}")

        if auc == auc:  # check not NaN
            plot_roc(labels, probs, f"AsymMirai ROC — Year {year}")

    # Save results
    results_df = pd.DataFrame(auc_results)
    results_df.to_csv("asym_auc_results.csv", index=False)

    print("\n=== Saved to asym_auc_results.csv ===")


# Run main
if __name__ == "__main__":
    main()
