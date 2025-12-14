import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------
features_df = pd.read_csv(
    "radiomics/radiomics_features_cc_with_metadata.csv"
)

meta_df = pd.read_csv(
    "data/embed/asymirai_input/EMBED_OpenData_metadata_screening_2D_complete_exams_with_demographics.csv"
)

id_cols = ["exam_id", "patient_id", "dicom_path"]

# --------------------------------------------------
# 2. Filter to dense breasts (tissueden == 4)
# --------------------------------------------------
meta_df = meta_df[meta_df["tissueden"] == 4]

features_df = features_df[
    features_df["exam_id"].isin(meta_df["exam_id"])
]

print(f"After density filter: {features_df.shape[0]} exams")

# --------------------------------------------------
# 3. Keep ONE exam per patient (critical)
# --------------------------------------------------
features_df = (
    features_df
    .groupby("patient_id", group_keys=False)
    .sample(n=1, random_state=42)
)

assert features_df["patient_id"].is_unique
print(f"After one-exam-per-patient filter: {features_df.shape[0]} exams")

# --------------------------------------------------
# 4. Build feature matrix
# --------------------------------------------------
feature_cols = [c for c in features_df.columns if c not in id_cols]

X = (
    features_df[feature_cols]
    .apply(pd.to_numeric, errors="coerce")
    .values
)

# --------------------------------------------------
# 5. Preprocessing + PCA pipeline
# --------------------------------------------------
pipe = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="median")),
    ("var", VarianceThreshold(threshold=1e-8)),
    ("scale", RobustScaler()),
    ("pca", PCA(n_components=0.97, random_state=42)),
])

X_pre = pipe.fit_transform(X)
print("Shape after PCA:", X_pre.shape)

# --------------------------------------------------
# 6. Inspect which features were kept / dropped
# --------------------------------------------------
var_step = pipe.named_steps["var"]
kept_mask = var_step.get_support()

kept_features = np.array(feature_cols)[kept_mask]
dropped_features = np.array(feature_cols)[~kept_mask]

print(f"Kept features: {len(kept_features)}")
print(f"Dropped features: {len(dropped_features)}")
print("Dropped features (first 10):", dropped_features[:10])

# --------------------------------------------------
# 7. Inspect PCA components
# --------------------------------------------------
pca = pipe.named_steps["pca"]

print("Number of PCA components:", pca.n_components_)
print("Cumulative variance explained:")
print(pca.explained_variance_ratio_.cumsum())

loadings = pd.DataFrame(
    pca.components_.T,
    index=kept_features,
    columns=[f"PC{i+1}" for i in range(pca.n_components_)]
)
# print(loadings["PC1"].abs().sort_values(ascending=False).head(10))


print("\nTop contributors to PC1:")
print(loadings["PC1"].abs().sort_values(ascending=False).head(10))

print("\nTop contributors to PC2:")
print(loadings["PC2"].abs().sort_values(ascending=False).head(10))

# --------------------------------------------------
# 8. Run t-SNE (visualization only)
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
# 9. Diagnostic plot (color by patient_id)
# --------------------------------------------------
plt.figure(figsize=(7, 6))
plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=features_df["patient_id"].astype("category").cat.codes,
    cmap="tab20",
    s=12
)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("t-SNE (density=4, one exam per patient)")
plt.show()
