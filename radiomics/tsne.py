import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline

try:
    import joblib
except ImportError:
    joblib = None


# ---------------------------
# Helpers
# ---------------------------
def normalize_id(s: pd.Series) -> pd.Series:
    # makes comparisons stable across int/float/string + whitespace
    return (
        s.astype(str)
         .str.strip()
         .str.replace(r"\.0$", "", regex=True)
    )

def normalize_path(s: pd.Series) -> pd.Series:
    # stabilizes dicom_path comparisons if you ever use them
    return (
        s.astype(str)
         .str.strip()
         .str.replace(r"//+", "/", regex=True)
    )

def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df[cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def main():
    # --------------------------------------------------
    # 1. Load data
    # --------------------------------------------------
    features_df = pd.read_csv("radiomics/radiomics_features_cc_with_metadata.csv")
    meta_df = pd.read_csv(
        "data/embed/asymirai_input/EMBED_OpenData_metadata_screening_2D_complete_exams_with_demographics.csv"
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

    # features_df["exam_id"] = normalize_id(features_df["exam_id"])
    # features_df["patient_id"] = normalize_id(features_df["patient_id"])
    # features_df["dicom_path"] = normalize_path(features_df["dicom_path"])

    # meta_df["exam_id"] = normalize_id(meta_df["exam_id"])
    # meta_df["patient_id"] = normalize_id(meta_df["patient_id"])

    # --------------------------------------------------
    # 3. Filter to dense breasts (tissueden == 4)
    # --------------------------------------------------
    meta_dense = meta_df.loc[meta_df["tissueden"] == 1, ["exam_id"]].drop_duplicates()

    before = len(features_df)
    features_df = features_df[features_df["exam_id"].isin(meta_dense["exam_id"])].copy()
    print(f"After density=4 filter: {features_df.shape[0]} rows (from {before})")

    # --------------------------------------------------
    # 4. Keep ONE exam per patient (critical)
    # --------------------------------------------------
    # If you want “one EXAM per patient” but radiomics rows might be “one IMAGE per exam”,
    # sampling at the patient level is fine for exploratory viz, but know what a “row” means.
    features_df = (
        features_df
        .groupby("exam_id", group_keys=False)
        .sample(n=1, random_state=42)
        .copy()
    )

    assert features_df["exam_id"].is_unique
    print(f"After one-row-per-patient: {features_df.shape[0]} rows")

    # # Save dicom paths for reproducibility
    # out_list = "radiomics/density_4_files.txt"
    # with open(out_list, "w") as f:
    #     for path in features_df["dicom_path"]:
    #         f.write(f"{path}\n")
    # print(f"Wrote selected dicom paths to: {out_list}")

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
    feature_cols = [c for c in feature_cols if c.startswith("original_") or c.startswith("custom_")]

    if len(feature_cols) == 0:
        raise ValueError("No feature columns found after exclusions.")

    # Quick per-feature NaN rate report (top 15 worst)
    X_df = safe_numeric(features_df, feature_cols)
    X = X_df.values

    # --------------------------------------------------
    # 6. Preprocessing + PCA pipeline
    # --------------------------------------------------
    pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(threshold=1e-8)),
        ("scale", RobustScaler()),
        ("pca", PCA(n_components=0.97, random_state=42)),
    ])

    X_pre = pipe.fit_transform(X)
    print("\nShape after PCA:", X_pre.shape)

    # # Save pipeline (so you can transform future cohorts consistently)
    # if joblib is not None:
    #     joblib.dump(pipe, "radiomics/prep_pca_pipe.joblib")
    #     print("Saved pipeline to radiomics/prep_pca_pipe.joblib")

    # --------------------------------------------------
    # 7. Inspect which features were kept / dropped (VarianceThreshold)
    # --------------------------------------------------
    var_step = pipe.named_steps["var"]
    kept_mask = var_step.get_support()

    kept_features = np.array(feature_cols)[kept_mask]
    dropped_features = np.array(feature_cols)[~kept_mask]

    print(f"\nKept features after VarianceThreshold: {len(kept_features)}")
    print(f"Dropped (near-constant) features: {len(dropped_features)}")
    print("Dropped features (first 10):", dropped_features[:10])

    # --------------------------------------------------
    # 8. Inspect PCA components (loadings)
    # --------------------------------------------------
    pca = pipe.named_steps["pca"]
    print("\nNumber of PCA components:", pca.n_components_)
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

    # --------------------------------------------------
    # 9. Run t-SNE (visualization only)
    # --------------------------------------------------
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=42
    )
    X_tsne = tsne.fit_transform(X_pre)

    # --------------------------------------------------
    # 10. Plot: color by custom feature if available, else by nothing
    # --------------------------------------------------
    plt.figure(figsize=(7, 6))

    if "custom_acd" in X_df.columns:
        cvals = X_df["custom_acd"].copy()
        # after imputation, plot the imputed values for a fair visual
        # easiest: transform a single column through imputer
        # but here we’ll just fill with median for coloring
        cvals = cvals.fillna(cvals.median())
        sc = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cvals.values, s=14)
        plt.colorbar(sc, label="custom_acd (NaNs→median for color)")
        plt.title("t-SNE (density=4, one row per patient) colored by custom_acd")
    else:
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=14)
        plt.title("t-SNE (density=4, one row per patient)")

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------
    # 11. Optional: quick redundancy check (ACD vs others)
    # --------------------------------------------------
    if "custom_acd" in X_df.columns:
        # Correlation computed on imputed + scaled features can be misleading,
        # so do a simple raw correlation with median-imputation for a quick look.
        tmp = X_df.copy()
        tmp = tmp.fillna(tmp.median(numeric_only=True))
        corr = tmp.corr(numeric_only=True)["custom_acd"].abs().sort_values(ascending=False)
        print("\nTop absolute correlations with custom_acd (raw, median-imputed):")
        print(corr.head(15))


if __name__ == "__main__":
    main()
