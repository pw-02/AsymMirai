import os
import ast
import pandas as pd

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
breast_csv = r"C:\Users\pw\projects\AsymMirai\data\breast-level_annotations.csv"
finding_csv = r"C:\Users\pw\projects\AsymMirai\data\finding_annotations.csv"
root_png_dir = r"/home/pwatters/projects/data/images_png"

OUTPUT_TRAIN = "input_train.csv"
OUTPUT_TEST  = "input_test.csv"

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
breast_df = pd.read_csv(breast_csv)
finding_df = pd.read_csv(finding_csv)

print("Breast-level columns:", breast_df.columns)
print("Finding annotation columns:", finding_df.columns)

# ---------------------------------------------------------
# 2. Parse finding_categories → list
# ---------------------------------------------------------
def parse_categories(cat_str):
    """Converts "['Mass','Calc']" → ['Mass', 'Calc']"""
    try:
        return [c.strip() for c in ast.literal_eval(cat_str)]
    except:
        return []

finding_df["parsed_categories"] = finding_df["finding_categories"].apply(parse_categories)

# ---------------------------------------------------------
# 3. Define malignant categories (CLINICALLY CORRECT)
# ---------------------------------------------------------
MALIGNANT_CATEGORIES = {
    "Mass",
    "Architectural Distortion",
    "Nipple Retraction",
    "Spiculated Mass",
    "Calcification",
    "Calcifications"
}

# Convert BI-RADS into numeric form
def birads_number(x):
    try:
        return int(x.replace("BI-RADS", "").strip())
    except:
        return None

finding_df["birads_num"] = finding_df["finding_birads"].apply(birads_number)

# ---------------------------------------------------------
# 4. Determine cancer per study_id
# ---------------------------------------------------------
def study_has_cancer(group):
    """
    Cancer-positive if:
        BI-RADS 4 or 5
        AND
        any malignant category is present
    """
    birads = list(group["birads_num"])
    categories = [c for lst in group["parsed_categories"] for c in lst]

    suspicious_birads = any(b in (4, 5) for b in birads)
    malignant_category = any(c in MALIGNANT_CATEGORIES for c in categories)

    return suspicious_birads and malignant_category


study_cancer = (
    finding_df
    .groupby("study_id")
    .apply(study_has_cancer)
    .reset_index()
    .rename(columns={0: "has_cancer"})
)

# Assign model-friendly labels
study_cancer["years_to_cancer"] = study_cancer["has_cancer"].apply(lambda x: 0 if x else 100)
study_cancer["years_to_last_followup"] = 10

print("\nCancer label distribution:")
print(study_cancer["has_cancer"].value_counts())

# ---------------------------------------------------------
# 5. FUNCTION: Construct final AsymMirai input DF
# ---------------------------------------------------------
def build_input_csv(df, output_path):
    """Build a complete AsymMirai CSV for a given split DataFrame (train or test)."""
    df = df.copy()

    # Standardized names used by AsymMirai
    df["exam_id"] = df["study_id"]
    df["patient_id"] = df["series_id"]
    df["laterality"] = df["laterality"]
    df["view"] = df["view_position"]

    # Construct PNG path for a ubudnu (not Windows) system
    root_png_dir = "/home/pwatters/projects/data/images_png"
    df["file_path"] = df.apply(
        lambda row: f"{root_png_dir}/{row['exam_id']}/{row['image_id']}.png",
        axis=1
    )

    # df["file_path"] = df.apply(
    #     lambda row: os.path.join(
    #         root_png_dir,
    #         str(row["exam_id"]),
    #         str(row["image_id"]) + ".png"
    #     ),
    #     axis=1
    # )

    

    # Merge cancer outcome (study-level → image-level)
    df = df.merge(
        study_cancer[["study_id", "years_to_cancer", "years_to_last_followup"]],
        how="left",
        left_on="exam_id",
        right_on="study_id"
    )
    df.drop(columns=["study_id_y"], inplace=True)
    df.rename(columns={"study_id_x": "study_id"}, inplace=True)

    # Additional required columns
    df["desc"] = "Screening"
    df["spot_mag"] = ""
    df["FinalImageType"] = "2D"

    # Select correct columns
    out = df[[
        "exam_id",
        "patient_id",
        "laterality",
        "view",
        "file_path",
        "years_to_cancer",
        "years_to_last_followup",
        "desc",
        "spot_mag",
        "FinalImageType"
    ]]

    # Save
    out.to_csv(output_path, index=False)
    print(f"Wrote {output_path} with {len(out)} rows.")


# ---------------------------------------------------------
# 6. BUILD TRAIN + TEST CSVs
# ---------------------------------------------------------
train_df = breast_df[breast_df["split"] == "training"]
test_df  = breast_df[breast_df["split"] == "test"]

build_input_csv(train_df, OUTPUT_TRAIN)
build_input_csv(test_df, OUTPUT_TEST)
