import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

train_ph = pd.read_csv("train_phenotypes_with_cancer_info.csv")
test_ph  = pd.read_csv("test_phenotypes_with_cancer_info.csv")

meta = pd.read_csv(
    "data/embed/asymirai_input/EMBED_OpenData_metadata_screening_2D_complete_exams.csv"
)

train = train_ph.merge(meta, on="exam_id", how="left")
test  = test_ph.merge(meta, on="exam_id", how="left")


OUTCOME = "developed_cancer"

COVARIATES = [
    "phenotype",       # radiomic phenotype
    "age_at_study",
    "tissueden",
    "birads"
]

df_or = train[[OUTCOME] + COVARIATES].dropna()

y = df_or[OUTCOME].astype(int)
X = df_or[COVARIATES]

# Add intercept
X = sm.add_constant(X)

logit_model = sm.Logit(y, X)
result = logit_model.fit()

print(result.summary())

or_table = pd.DataFrame({
    "OR": np.exp(result.params),
    "CI_lower": np.exp(result.conf_int()[0]),
    "CI_upper": np.exp(result.conf_int()[1]),
    "p_value": result.pvalues
})

print("\nAdjusted odds ratios:")
print(or_table)


X_train = train[COVARIATES]
y_train = train[OUTCOME].astype(int)

X_test = test[COVARIATES]
y_test = test[OUTCOME].astype(int)

clinical_covs = ["age_at_study", "tissueden", "birads"]

pipe_clinical = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("clf", LogisticRegression(
        max_iter=500,
        class_weight="balanced"
    ))
])

pipe_clinical.fit(X_train[clinical_covs], y_train)

pred_test_clin = pipe_clinical.predict_proba(X_test[clinical_covs])[:, 1]
auc_clin = roc_auc_score(y_test, pred_test_clin)



pipe_full = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("clf", LogisticRegression(
        max_iter=500,
        class_weight="balanced"
    ))
])

pipe_full.fit(X_train[COVARIATES], y_train)

pred_test_full = pipe_full.predict_proba(X_test[COVARIATES])[:, 1]
auc_full = roc_auc_score(y_test, pred_test_full)


print("\nTest-set AUCs:")
print(f"Clinical only:        {auc_clin:.3f}")
print(f"Clinical + phenotype: {auc_full:.3f}")
