import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline


DEBUG_MODE = False
DENSITY_FILTER = 4  # set to None to disable density filteringS

PLOT_BY_FEATURE = "Caner"  # e.g., "Caner" to color by that feature, or None for no coloring


def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df[cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def main():
    # --------------------------------------------------
    # 1. Load data
    # --------------------------------------------------
    features_df = pd.read_csv("radiomics/results/radiomics_features_breast_density_4.csv")
    #features_df = pd.read_csv("radiomics/results/radiomics_features_cc.csv")
    meta_df = pd.read_csv("data/embed/asymirai_input/EMBED_OpenData_metadata_screening_2D_complete_exams.csv")
    features_df = features_df.merge(
        meta_df[["dicom_path", "patient_id", "exam_id"]],
        on="dicom_path",
        how="left",
        validate="many_to_one",  # radiomics → metadata
    )
    # --------------------------------------------------
    # 2. Normalize keys / basic validation
    # --------------------------------------------------
    required_feat_cols = {"exam_id", "patient_id", "dicom_path"}
    if not required_feat_cols.issubset(features_df.columns):
        raise ValueError(f"features_df missing required columns: {required_feat_cols - set(features_df.columns)}")

    required_meta_cols = {"exam_id", "patient_id", "tissueden"}
    if not required_meta_cols.issubset(meta_df.columns):
        raise ValueError(f"meta_df missing required columns: {required_meta_cols - set(meta_df.columns)}")
    
    # --------------------------------------------------
    # 3. Filter to dense breasts (tissueden == 4)
    # --------------------------------------------------
    if DENSITY_FILTER is not None:
        meta_dense = meta_df.loc[meta_df["tissueden"] == DENSITY_FILTER, ["exam_id"]]
        before = len(features_df)
        features_df = features_df[features_df["exam_id"].isin(meta_dense["exam_id"])].copy()
        print(f"After density={DENSITY_FILTER} filter: {features_df.shape[0]} rows (from {before})")
    # # --------------------------------------------------
    # # 4. Keep ONE exam per exam”
    # # --------------------------------------------------
    features_df = (
        features_df
        .groupby("exam_id", group_keys=False)
        .sample(n=1, random_state=42)
        .copy()
    )
    assert features_df["exam_id"].is_unique
    print(f"After keeping one row per exam: {features_df.shape[0]} rows")
    
    # --------------------------------------------------
    # 5. Build feature matrix (explicitly exclude non-features)
    # --------------------------------------------------
    id_cols = ["exam_id", "patient_id", "dicom_path"]
    # Drop common non-feature columns if they exist
    drop_if_present = {
        "GENDER_DESC", "ETHNICITY_DESC", "MARITAL_STATUS_DESC", "age_at_study", "tissueden"
    }
    non_feature_cols = set(id_cols) | (drop_if_present & set(features_df.columns))
    feature_cols = [c for c in features_df.columns if c not in non_feature_cols]

    # Optional: if you only want “radiomics-like” columns, uncomment:
    feature_cols = [c for c in feature_cols if c.startswith("original")]

    if len(feature_cols) == 0:
        raise ValueError("No feature columns found after exclusions.")

    # Quick per-feature NaN rate report (top 15 worst)
    X_df = safe_numeric(features_df, feature_cols)
    X = X_df.values

    # --------------------------------------------------
    # 6. Preprocessing + PCA pipeline
    # --------------------------------------------------
    #run this if num features are > 30
    if X.shape[1] <= 30:
        print(f"Feature count {X.shape[1]} <= 30, skipping PCA step.")
        pipe = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("var", VarianceThreshold(threshold=1e-8)),
            ("scale", RobustScaler()),
        ])
    else:
        pipe = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("var", VarianceThreshold(threshold=1e-8)),
            ("scale", RobustScaler()),
            ("pca", PCA(n_components=30, random_state=42)),
        ])

    X_pre = pipe.fit_transform(X)
    pca = pipe.named_steps["pca"] if "pca" in pipe.named_steps else None
    if pca is not None:
        print("\nNumber of PCA components:", pca.n_components_)

    # --------------------------------------------------
    # 7. Inspect which features were kept / dropped (VarianceThreshold)
    # --------------------------------------------------
    if DEBUG_MODE:
        var_step = pipe.named_steps["var"]
        kept_mask = var_step.get_support()
        kept_features = np.array(feature_cols)[kept_mask]
        dropped_features = np.array(feature_cols)[~kept_mask]
        print(f"\nKept features after VarianceThreshold: {len(kept_features)}")
        print(f"Dropped (near-constant) features: {len(dropped_features)}")
        print("Dropped features (first 10):", dropped_features[:10])
        print("Cumulative variance explained (first 10):")
        print(pca.explained_variance_ratio_.cumsum()[:10])
        loadings = pd.DataFrame(
            pca.components_.T,
            index=kept_features,
            columns=[f"PC{i+1}" for i in range(pca.n_components_)]
        )
        # Print top contributors for first couple PCs (guard if only 1 PC exists)
        print("\nTop contributors to PC1:")
        print(loadings["PC1"].abs().sort_values(ascending=False).head(10))

        if pca.n_components_ >= 2:
            print("\nTop contributors to PC2:")
            print(loadings["PC2"].abs().sort_values(ascending=False).head(10))

    # # --------------------------------------------------
    # # 9. Run t-SNE 
    # # --------------------------------------------------
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=42
    )
    X_tsne = tsne.fit_transform(X_pre)

    # # --------------------------------------------------
    # # 10. Build exam → cancer label
    # # --------------------------------------------------
    # # meta_df must contain: exam_id, cancer (0/1)

    if "cancer" in PLOT_BY_FEATURE.lower():
        print("\nPlotting by cancer outcome...")

        cancer_map = (
        meta_df[["exam_id", "developed_cancer"]]
        .drop_duplicates()
        .set_index("exam_id")["developed_cancer"]
        )
        cancer_map = cancer_map.astype("boolean")  # pandas BooleanDtype
        cancer_labels = features_df["exam_id"].map(cancer_map)
        
        mask_cancer = cancer_labels == True
        mask_no_cancer = cancer_labels == False
        mask_unknown = cancer_labels.isna()

        plt.figure(figsize=(7, 6))

        # No cancer (background)
        plt.scatter(
            X_tsne[mask_no_cancer, 0],
            X_tsne[mask_no_cancer, 1],
            s=12,
            alpha=0.35,
            label="No cancer"
        )

        # Developed cancer (highlight)
        plt.scatter(
            X_tsne[mask_cancer, 0],
            X_tsne[mask_cancer, 1],
            s=30,
            edgecolor="black",
            linewidth=0.6,
            label="Developed cancer"
        )

        # Unknown outcome (optional)
        if mask_unknown.any():
            plt.scatter(
                X_tsne[mask_unknown, 0],
                X_tsne[mask_unknown, 1],
                s=12,
                alpha=0.2,
                label="Unknown outcome"
            )

        plt.legend()
        plt.title("t-SNE of exams (Breast density=4)\nCancer outcome highlighted")
        plt.xlabel("t-SNE dimension 1")
        plt.ylabel("t-SNE dimension 2")
        plt.tight_layout()
        plt.show()
    
    elif 'birad' in PLOT_BY_FEATURE.lower():
        print("\nPlotting by BIRAD...")

        birad_map = (
        meta_df[["exam_id", "birads"]]
        .drop_duplicates()
        .set_index("exam_id")["birads"]
        )
        birad_labels = features_df["exam_id"].map(birad_map)

        plt.figure(figsize=(7, 6))
        sc = plt.scatter(
            X_tsne[:, 0],
            X_tsne[:, 1],
            c=birad_labels,
            s=14,
            cmap="viridis",
        )
        plt.colorbar(sc, label="BIRAD")
        plt.title("t-SNE of exams (Breast density=4) colored by BIRAD")
        plt.xlabel("t-SNE dimension 1")
        plt.ylabel("t-SNE dimension 2")
        plt.tight_layout()
        plt.show()



    # # --------------------------------------------------
    # # 10. Plot: color by custom feature if available, else by nothing
    # # --------------------------------------------------
    # plt.figure(figsize=(7, 6))

    # if "custom_acd" in X_df.columns:
    #     cvals = X_df["custom_acd"].copy()
    #     # after imputation, plot the imputed values for a fair visual
    #     # easiest: transform a single column through imputer
    #     # but here we’ll just fill with median for coloring
    #     cvals = cvals.fillna(cvals.median())
    #     sc = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cvals.values, s=14)
    #     plt.colorbar(sc, label="custom_acd (NaNs→median for color)")
    #     plt.title("t-SNE (density=4, one row per patient) colored by custom_acd")
    # else:
    #     plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=14)
    #     plt.title("t-SNE (density=4, one row per patient)")

    # plt.xlabel("t-SNE 1")
    # plt.ylabel("t-SNE 2")
    # plt.tight_layout()
    # plt.show()

    # # --------------------------------------------------
    # # 11. Optional: quick redundancy check (ACD vs others)
    # # --------------------------------------------------
    # if "custom_acd" in X_df.columns:
    #     # Correlation computed on imputed + scaled features can be misleading,
    #     # so do a simple raw correlation with median-imputation for a quick look.
    #     tmp = X_df.copy()
    #     tmp = tmp.fillna(tmp.median(numeric_only=True))
    #     corr = tmp.corr(numeric_only=True)["custom_acd"].abs().sort_values(ascending=False)
    #     print("\nTop absolute correlations with custom_acd (raw, median-imputed):")
    #     print(corr.head(15))


if __name__ == "__main__":
    main()
