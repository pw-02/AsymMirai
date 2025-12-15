import pandas as pd
import matplotlib.pyplot as plt

def main(phenotype_file: str = "test_phenotypes.csv"):

    # Load phenotype assignments (test set only)
    ph_df = pd.read_csv(phenotype_file)

    # Load screening metadata
    meta_df = pd.read_csv(
        "data/embed/asymirai_input/EMBED_OpenData_metadata_screening_2D_complete_exams.csv"
    )

    # Sanity checks
    assert "exam_id" in ph_df.columns
    assert "exam_id" in meta_df.columns
    assert "developed_cancer" in meta_df.columns

    print(f"Total test exams: {len(ph_df)}")
    print(f"Total metadata exams: {len(meta_df)}")

    # Identify exams that developed cancer
    cancer_exam_ids = meta_df.loc[
        meta_df["developed_cancer"] == True, "exam_id"
    ].unique()

    print(f"Number of exams with cancer: {len(cancer_exam_ids)}")

    #remove duplicates exams in meta_df
    cancer_exam_ids = pd.Series(cancer_exam_ids).drop_duplicates()

    print(f"Number of unique exams with cancer: {len(cancer_exam_ids)}")

    # Flag cancer outcome in phenotype table
    ph_df["developed_cancer"] = ph_df["exam_id"].isin(cancer_exam_ids)

    # Optional: check prevalence
    print(ph_df["developed_cancer"].value_counts())

    # Save augmented table
    ph_df.to_csv("test_phenotypes_with_cancer_info.csv", index=False) if phenotype_file == "test_phenotypes.csv" else ph_df.to_csv("train_phenotypes_with_cancer_info.csv", index=False)

    summary = (
        ph_df
        .groupby("phenotype")["developed_cancer"]
        .agg(
            total_exams="count",
            cancer_cases="sum",
            cancer_rate="mean"
        )
        .reset_index()
    )

    summary["cancer_rate_percent"] = 100 * summary["cancer_rate"]

    print(summary)
    fig, ax = plt.subplots(figsize=(5, 4))

    x = summary["phenotype"].astype(str)
    y = summary["cancer_rate_percent"]
    n = summary["total_exams"]

    ax.bar(x, y)

    # Annotate bars with percentages and N
    for i, (rate, total) in enumerate(zip(y, n)):
        ax.text(
            i, rate + 0.02,
            f"{rate:.2f}%\n(N={total})",
            ha="center",
            va="bottom",
            fontsize=9
        )

    ax.set_xlabel("Radiomic Phenotype")
    ax.set_ylabel("Cancer incidence (%)")
    ax.set_ylim(0, max(y) * 1.4)

    set_name = "test set" if phenotype_file == "test_phenotypes.csv" else "training set"
    ax.set_title(f"Cancer incidence by radiomic phenotype ({set_name})")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    test_file = "test_phenotypes.csv"
    train_file = "train_phenotypes.csv"
    main(test_file)
    main(train_file)
