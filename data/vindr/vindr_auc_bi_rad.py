import pandas as pd
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt

def compute_detection_auc(df):
    y_true = df["birads_binary"].values
    y_pred = df["prediction_pos"].values

    auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, y_pred)

    return auc, fpr, tpr

def main():
    
    meta = pd.read_csv("data\\breast_level_annotations.csv")
    preds = pd.read_csv("validation_predictions2.csv")
    preds = preds.rename(columns={"exam_id": "study_id"})

    print("META COLUMNS:\n", meta.columns)
    print("\nPREDICTION COLUMNS:\n", preds.columns)

    print("\n=== Loading VinDr metadata ===")

    # Convert BI-RADS to binary detection labels
    # Positive = 0,4,5; Negative = 1,2
    meta["birads_binary"] = meta["birads"].apply(
        lambda x: 1 if x in [0, 4, 5] else 0
    )

    print("\n=== Loading AsymMirai predictions ===")

    # Merge on exam ID
    merged = preds.merge(meta[["study_id", "birads_binary"]], on="study_id", how="inner")

    # Compute AUC
    auc, fpr, tpr = compute_detection_auc(merged)
    print(f"\nAUC for BI-RADS detection-style evaluation: {auc:.4f}")

    # --- Single-column, publication-quality ROC plot ---
    plt.figure(figsize=(3.3, 2.3), dpi=300)  # Single-column width (~3.3 in)

    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)

    plt.xlabel("False Positive Rate", fontsize=10)
    plt.ylabel("True Positive Rate", fontsize=10)

    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend(fontsize=9, loc="lower right")

    plt.tight_layout()

    # Save in both pixel and vector formats
    plt.savefig("asymmirai_vindr_roc.png", dpi=300, bbox_inches="tight")
    plt.savefig("asymmirai_vindr_roc.pdf", bbox_inches="tight")

    plt.show()



if __name__ == "__main__":
    main()
