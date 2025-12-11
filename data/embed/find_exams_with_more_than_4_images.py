import pandas as pd

# Load your CSV
df = pd.read_csv("data/embed/asymirai_input/EMBED_OpenData_metadata_screening_2D_complete_exams.csv")

# ----------------------------------------------------------
# 1. STRICT view normalization (only CC and MLO allowed)
# ----------------------------------------------------------

def normalize_view(v):
    """
    Extracts canonical mammography view:
    - returns 'CC' or 'MLO'
    - returns None for ANYTHING else (Spot, Mag, AT, AX, IMF, Combo variants, etc.)
    """
    v = str(v).upper()

    # Canonical views only
    if "MLO" in v:
        return "MLO"

    if v.startswith("CC") or "CC" in v:
        return "CC"

    # Everything else is discarded
    return None

df["view"] = df["view"].apply(normalize_view)

# Keep only CC and MLO
df = df[df["view"].isin(["CC", "MLO"])].copy()

print("After keeping only CC/MLO views:", df.shape)

# ----------------------------------------------------------
# 2. Deduplicate so we keep only ONE image per view & laterality
# ----------------------------------------------------------

df_unique = (
    df.sort_values(["patient_id", "exam_id", "view", "laterality", "dicom_path"])
      .drop_duplicates(subset=["patient_id", "exam_id", "view", "laterality"])
)

print("After deduplicating view/laterality:", df_unique.shape)

# ----------------------------------------------------------
# 3. Keep ONLY exams with exactly the 4 canonical views
#    (MLO-L, MLO-R, CC-L, CC-R)
# ----------------------------------------------------------

required_pairs = {("MLO", "L"), ("MLO", "R"), ("CC", "L"), ("CC", "R")}

# Create pair
df_unique["pair"] = list(zip(df_unique["view"], df_unique["laterality"]))

# Collect pairs present per exam
exam_pairs = (
    df_unique.groupby(["patient_id", "exam_id"])["pair"]
    .apply(set)
    .reset_index(name="pairs")
)

# Identify exams that have ALL FOUR required views
valid_exams = exam_pairs[
    exam_pairs["pairs"].apply(lambda x: required_pairs.issubset(x))
][["patient_id", "exam_id"]]

print("Number of valid full 4-view exams:", valid_exams.shape[0])

# Keep ONLY those rows
df_final = df_unique.merge(valid_exams, on=["patient_id", "exam_id"])

#drop pairs column
df_final = df_final.drop(columns=["pair"])

print("Final dataset shape (exactly 4 views each):", df_final.shape)

# ----------------------------------------------------------
# 4. Save cleaned dataset
# ----------------------------------------------------------

outpath = "data/embed/asymirai_input/EMBED_OpenData_metadata_screening_2D_complete_exams_CLEANED_4VIEW.csv"
df_final.to_csv(outpath, index=False)

print("\nSaved cleaned 4-view dataset to:")
print(outpath)
