
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt

df = pd.read_csv("radiomics/radiomics_features_cc_with_metadata.csv")

id_cols = ["exam_id", "patient_id", "dicom_path"]
feature_cols = [c for c in df.columns if c not in id_cols]

# Keep only numeric radiomics columns
X = df[feature_cols].apply(pd.to_numeric, errors="coerce").values

pipe = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="median")),
    ("var", VarianceThreshold(threshold=1e-8)),    # drop constant features
    ("scale", RobustScaler()),                    # often better for radiomics
    ("pca", PCA(n_components=0.97, random_state=42)),  # keep 97% variance (cap happens naturally)
])

X_pre = pipe.fit_transform(X)

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=42
)
X_tsne = tsne.fit_transform(X_pre)

plt.figure(figsize=(7,6))
plt.scatter(X_tsne[:,0], X_tsne[:,1], s=10)
plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
plt.title("t-SNE (Radiomics) | impute+robustscale+PCA")
plt.show()
