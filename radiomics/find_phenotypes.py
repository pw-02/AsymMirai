# ============================================
# Radiomic Phenotype Discovery Pipeline
# Ward clustering + consensus + SigClust-like test
# ============================================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal

# --------------------------------------------
# 1. Load CSV (rows = exams, columns = features)
# --------------------------------------------
df = pd.read_csv("radiomics.csv")
X = df.drop(columns=["exam_id"], errors="ignore").values

# --------------------------------------------
# 2. Train / test split
# --------------------------------------------
X_train, X_test = train_test_split(
    X, test_size=0.2, random_state=42
)

# --------------------------------------------
# 3. Standardize features (CRITICAL)
# --------------------------------------------
scaler = StandardScaler()
Xtr = scaler.fit_transform(X_train)
Xte = scaler.transform(X_test)

# --------------------------------------------
# 4. Ward clustering function
# --------------------------------------------
def ward_cluster(X, k):
    model = AgglomerativeClustering(
        n_clusters=k, linkage="ward"
    )
    return model.fit_predict(X)

# --------------------------------------------
# 5. Consensus clustering (stability)
# --------------------------------------------
def consensus_score(X, k, B=200, p=0.8):
    n = X.shape[0]
    co = np.zeros((n, n))
    ct = np.zeros((n, n))

    for _ in range(B):
        idx = np.random.choice(n, int(p * n), replace=False)
        labels = ward_cluster(X[idx], k)

        for i, a in enumerate(idx):
            for j, b in enumerate(idx):
                ct[a, b] += 1
                if labels[i] == labels[j]:
                    co[a, b] += 1

    consensus = np.divide(co, ct, out=np.zeros_like(co), where=ct > 0)
    return consensus.mean()

# --------------------------------------------
# 6. Choose k via stability
# --------------------------------------------
k_candidates = range(2, 7)
stability = {}

print("Consensus stability:")
for k in k_candidates:
    stability[k] = consensus_score(Xtr, k)
    print(f"k = {k}, stability = {stability[k]:.3f}")

print("\nRelative improvement:")
for k in range(3, 7):
    improvement = (stability[k] - stability[k-1]) / stability[k-1]
    print(f"k = {k}, improvement = {improvement:.2%}")

# ---- Manually select k where improvement < 10%
k = 3
print(f"\nSelected k = {k}")

# --------------------------------------------
# 7. SigClust-like statistical test
# --------------------------------------------
def cluster_index(X, labels):
    overall_mean = X.mean(axis=0)
    total_ss = ((X - overall_mean) ** 2).sum()

    within_ss = 0
    for c in np.unique(labels):
        Xc = X[labels == c]
        mu = Xc.mean(axis=0)
        within_ss += ((Xc - mu) ** 2).sum()

    return within_ss / total_ss

def sigclust_test(X, labels, B=1000):
    n, d = X.shape
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)

    obs_ci = cluster_index(X, labels)

    null_ci = []
    for _ in range(B):
        Xsim = multivariate_normal.rvs(mean=mu, cov=cov, size=n)
        lab_sim = ward_cluster(Xsim, len(np.unique(labels)))
        null_ci.append(cluster_index(Xsim, lab_sim))

    p_value = np.mean(np.array(null_ci) <= obs_ci)
    return obs_ci, p_value

# --------------------------------------------
# 8. Final clustering on training set
# --------------------------------------------
labels_tr = ward_cluster(Xtr, k)
ci, pval = sigclust_test(Xtr, labels_tr)

print(f"\nCluster index = {ci:.3f}")
print(f"SigClust-like p-value = {pval:.4f}")

# --------------------------------------------
# 9. Assign test exams (nearest centroid)
# --------------------------------------------
centroids = np.vstack([
    Xtr[labels_tr == c].mean(axis=0)
    for c in range(k)
])

distances = cdist(Xte, centroids, metric="euclidean")
labels_te = distances.argmin(axis=1)

# --------------------------------------------
# 10. Save results
# --------------------------------------------
train_results = pd.DataFrame({
    "phenotype": labels_tr
})

test_results = pd.DataFrame({
    "phenotype": labels_te
})

train_results.to_csv("train_phenotypes.csv", index=False)
test_results.to_csv("test_phenotypes.csv", index=False)

print("\nPhenotype assignment complete.")
