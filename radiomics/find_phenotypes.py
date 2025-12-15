# ============================================
# Radiomic Phenotype Discovery Pipeline (final)
# PCA + Ward clustering + stability + Gaussian null
# Subsample for clustering, centroid-assign for full set
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal
from scipy.cluster.hierarchy import dendrogram

RANDOM_STATE = 42
MAX_CLUSTER_N = 8000          # clustering is O(n^2); subsample for Ward
K_CANDIDATES = range(2, 7)
CONSENSUS_B = 50              # 200 can be slow; start with 50-100
CONSENSUS_P = 0.8
NULL_B = 200                  # 500 can be slow; start with 200


def load_and_prepare(csv_path: str) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(csv_path)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    # one row per exam
    if "exam_id" in df.columns:
        df = df.drop_duplicates(subset=["exam_id"]).reset_index(drop=True)
        print(f"After drop_duplicates(exam_id): {df.shape[0]} rows")

    feature_cols = df.drop(
        columns=["exam_id", "patient_id", "dicom_path"],
        errors="ignore"
    ).columns.tolist()

    if len(feature_cols) == 0:
        raise ValueError("No feature columns found after dropping ID/path columns.")

    # ensure numeric
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    # simple median imputation (fast + robust); optional but prevents PCA issues
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))

    return df, feature_cols


def fit_preprocess(X_train: np.ndarray, X_test: np.ndarray, var_keep: float = 0.90):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    pca = PCA(n_components=var_keep, svd_solver="full", random_state=RANDOM_STATE)
    Xtr_pca = pca.fit_transform(Xtr)
    Xte_pca = pca.transform(Xte)

    print(f"PCA retained {Xtr_pca.shape[1]} components "
          f"({pca.explained_variance_ratio_.sum():.2%} variance)")
    return scaler, pca, Xtr_pca, Xte_pca

def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)

    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)



def ward_cluster(X: np.ndarray, k: int) -> np.ndarray:
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    return model.fit_predict(X)


def subsample_rows(X: np.ndarray, n_max: int, rng: np.random.Generator):
    n = X.shape[0]
    if n <= n_max:
        return X, np.arange(n)
    idx = rng.choice(n, size=n_max, replace=False)
    return X[idx], idx


def consensus_stability(X: np.ndarray, k: int, B: int, p: float, rng: np.random.Generator) -> float:
    """
    Simple stability proxy: average fraction of sampled points that land in clusters of size > 1.
    (Fast, but not full Monti consensus. Good as a first-pass screen.)
    """
    n = X.shape[0]
    scores = []

    for _ in range(B):
        idx = rng.choice(n, size=int(p * n), replace=False)
        labels = ward_cluster(X[idx], k)

        # fraction of sampled points in clusters with >1 member
        frac = 0.0
        for c in np.unique(labels):
            m = np.sum(labels == c)
            if m > 1:
                frac += m
        scores.append(frac / len(idx))

    return float(np.mean(scores))


def choose_k_by_stability(X_cluster: np.ndarray, k_candidates, B: int, p: float, rng: np.random.Generator) -> tuple[int, dict]:
    stability = {}
    print("\nConsensus stability (on clustering subsample):")
    for k in k_candidates:
        stability[k] = consensus_stability(X_cluster, k, B=B, p=p, rng=rng)
        print(f"k = {k}: stability = {stability[k]:.3f}")

    print("\nRelative improvement:")
    best_k = min(k_candidates)
    for k in k_candidates:
        if k == min(k_candidates):
            continue
        prev = stability[k - 1]
        imp = (stability[k] - prev) / (prev + 1e-12)
        print(f"k = {k}: improvement = {imp:.2%}")

    # simple heuristic: smallest k after which improvement < 10%
    for k in range(min(k_candidates) + 1, max(k_candidates) + 1):
        prev = stability[k - 1]
        imp = (stability[k] - prev) / (prev + 1e-12)
        if imp < 0.10:
            best_k = k - 1
            break
        best_k = k

    print(f"\nSelected k = {best_k}")
    return best_k, stability


def cluster_index(X: np.ndarray, labels: np.ndarray) -> float:
    overall_mean = X.mean(axis=0)
    total_ss = ((X - overall_mean) ** 2).sum()

    within_ss = 0.0
    for c in np.unique(labels):
        Xc = X[labels == c]
        mu = Xc.mean(axis=0)
        within_ss += ((Xc - mu) ** 2).sum()

    return float(within_ss / (total_ss + 1e-12))


def gaussian_null_test(X: np.ndarray, labels: np.ndarray, B: int, rng: np.random.Generator) -> tuple[float, float]:
    """
    Gaussian null resampling test analogous to SigClust:
    simulate from single Gaussian with same mean/cov, recluster, compare cluster index.
    """
    n = X.shape[0]
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)

    obs_ci = cluster_index(X, labels)
    null_ci = []

    k = len(np.unique(labels))
    for _ in range(B):
        Xsim = multivariate_normal.rvs(mean=mu, cov=cov, size=n, random_state=rng.integers(0, 2**31 - 1))
        lab_sim = ward_cluster(Xsim, k)
        null_ci.append(cluster_index(Xsim, lab_sim))

    null_ci = np.array(null_ci)
    p_value = float(np.mean(null_ci <= obs_ci))
    return obs_ci, p_value


def compute_centroids(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    return np.vstack([X[labels == c].mean(axis=0) for c in range(k)])


def assign_by_nearest_centroid(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    distances = cdist(X, centroids, metric="euclidean")
    return distances.argmin(axis=1)


def plot_pca_variance(pca: PCA):
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(pca.explained_variance_ratio_) + 1),
             pca.explained_variance_ratio_.cumsum(),
             marker="o")
    plt.axhline(0.9, linestyle="--", alpha=0.5)
    plt.xlabel("Number of PCA components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA explained variance")
    plt.tight_layout()
    plt.show()


def plot_pc1_pc2_with_centroids(X_pca: np.ndarray, labels: np.ndarray, centroids: np.ndarray, title: str):
    plt.figure(figsize=(7, 6))
    # light background subsample for speed in rendering
    n = X_pca.shape[0]
    if n > 8000:
        idx = np.random.default_rng(RANDOM_STATE).choice(n, size=8000, replace=False)
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], s=6, alpha=0.08)
    else:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], s=6, alpha=0.08)

    # centroids on top
    plt.scatter(centroids[:, 0], centroids[:, 1], s=250, marker="X")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()


def main():
    rng = np.random.default_rng(RANDOM_STATE)

    # 1) Load / prepare
    df, feature_cols = load_and_prepare("radiomics/results/radiomics_features_cc_with_metadata.csv")
    X = df[feature_cols].values
    exam_ids = df["exam_id"].values if "exam_id" in df.columns else np.arange(df.shape[0])

    # 2) Split (keep indices so we can save exam_id properly)
    idx_all = np.arange(X.shape[0])
    idx_train, idx_test = train_test_split(idx_all, test_size=0.2, random_state=RANDOM_STATE)

    X_train, X_test = X[idx_train], X[idx_test]
    exam_train, exam_test = exam_ids[idx_train], exam_ids[idx_test]

    # 3–4) Scale + PCA
    scaler, pca, Xtr_pca, Xte_pca = fit_preprocess(X_train, X_test, var_keep=0.90)

    plot_pca_variance(pca)

    # Loadings (safe)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )
    print("\nTop |loadings| PC1:")
    print(loadings["PC1"].abs().sort_values(ascending=False).head(10))
    if pca.n_components_ >= 2:
        print("\nTop |loadings| PC2:")
        print(loadings["PC2"].abs().sort_values(ascending=False).head(10))

    # 5) Subsample for Ward (speed)
    X_cluster, idx_cluster = subsample_rows(Xtr_pca, MAX_CLUSTER_N, rng)
    print(f"\nClustering on subsample: {X_cluster.shape[0]} / {Xtr_pca.shape[0]} train exams")

    # 6–7) Choose k on subsample
    k, stability = choose_k_by_stability(X_cluster, K_CANDIDATES, B=CONSENSUS_B, p=CONSENSUS_P, rng=rng)

    # 8–9) Fit Ward on subsample, compute centroids, assign all
    labels_cluster = ward_cluster(X_cluster, k)

    agg = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    agg.fit(X_cluster)

    ig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.scatter(X_cluster[:, 0], X_cluster[:, 1], c=labels_cluster, cmap='viridis', s=70)
    ax1.set_title("Agglomerative Clustering")
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")

    plt.sca(ax2)
    plot_dendrogram(agg, truncate_mode='level', p=5)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Sample index")
    plt.ylabel("Distance")

    plt.tight_layout()
    plt.show()




    centroids = compute_centroids(X_cluster, labels_cluster, k)

    labels_tr = assign_by_nearest_centroid(Xtr_pca, centroids)
    labels_te = assign_by_nearest_centroid(Xte_pca, centroids)

    print("\nCluster sizes (train, centroid-assigned):")
    print(pd.Series(labels_tr).value_counts().sort_index())
    print("\nCluster sizes (test, centroid-assigned):")
    print(pd.Series(labels_te).value_counts().sort_index())

    # 10) Gaussian null test (run on subsample only; full set is too slow)
    ci, pval = gaussian_null_test(X_cluster, labels_cluster, B=NULL_B, rng=rng)
    print(f"\nCluster index (subsample) = {ci:.3f}")
    print(f"Gaussian null p-value (subsample) = {pval:.4f}")

    # Optional visualization: centroids on PCA plane
    plot_pc1_pc2_with_centroids(
        Xtr_pca,
        labels_tr,
        centroids,
        title="Radiomic phenotypes: PCA space with phenotype centroids (train)"
    )

    # 11) Save with exam_id (critical for downstream merges)
    train_results = pd.DataFrame({"exam_id": exam_train, "phenotype": labels_tr})
    test_results = pd.DataFrame({"exam_id": exam_test, "phenotype": labels_te})

    train_results.to_csv("train_phenotypes.csv", index=False)
    test_results.to_csv("test_phenotypes.csv", index=False)
    print("\nPhenotype assignment complete. Saved train_phenotypes.csv and test_phenotypes.csv")


if __name__ == "__main__":
    main()
