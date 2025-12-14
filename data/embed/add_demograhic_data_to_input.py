import os
import pandas as pd
import boto3
from tqdm import tqdm

# ---------------------------
# AWS / S3 SETUP
# ---------------------------
s3 = boto3.client("s3")
BUCKET_NAME = "embdedpng"

# ---------------------------
# S3 HELPERS
# ---------------------------
def list_all_s3_keys(bucket, prefix=None):
    keys = set()
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.add(obj["Key"])

    return keys


def build_s3_key(path):
    s3_key = path.replace("images", "png_tmp").replace(".dcm", ".png")
    return os.path.normpath(s3_key).replace("\\", "/")


def check_file_not_exists_in_s3(all_s3_keys, path):
    s3_key = path.replace('images', 'png_images/../png_tmp').replace('.dcm', '.png')
     #ensure forward slashes
    s3_key = s3_key.replace('\\', '/')
    return s3_key not in all_s3_keys

# ---------------------------
# MAIN
# ---------------------------
def main():

    # ---------------------------
    # LOAD / CACHE S3 KEYS
    # ---------------------------
    s3_keys_cache = "data/embed/asymirai_input/all_s3_keys.txt"

    if os.path.exists(s3_keys_cache):
        with open(s3_keys_cache, "r") as f:
            all_s3_keys = set(line.strip() for line in f)
    else:
        all_s3_keys = list_all_s3_keys(BUCKET_NAME, prefix="png_images/")
        os.makedirs(os.path.dirname(s3_keys_cache), exist_ok=True)
        with open(s3_keys_cache, "w") as f:
            for key in all_s3_keys:
                f.write(f"{key}\n")

    print(f"Total S3 files: {len(all_s3_keys)}")

    # ---------------------------
    # LOAD CLINICAL DATA (FIXED)
    # ---------------------------
    clinical_data = pd.read_csv("data/embed/tables/EMBED_OpenData_clinical.csv")

    #check if 'acc_anon' column exists
    if 'acc_anon' not in clinical_data.columns:
        raise ValueError("Column 'acc_anon' not found in clinical data.")
    else:
        print("'acc_anon' column found in clinical data.Type: ", clinical_data['acc_anon'].dtype)
        val = clinical_data.loc[clinical_data["acc_anon"].astype(str).str.contains("9718", na=False),"acc_anon"].iloc[0]
        print(val)
        print(type(val))
        print(len(str(val)))
        print(str(val))

    demographic_data = clinical_data[
        [
            "empi_anon",
            "acc_anon",
            "GENDER_DESC",
            "ETHNICITY_DESC",
            "MARITAL_STATUS_DESC",
            "age_at_study",
            "tissueden",
            "path_severity",
        ]
    ].rename(
        columns={
            "empi_anon": "patient_id",
            "acc_anon": "exam_id",
        }
    )

    if 'exam_id' not in demographic_data.columns:
        raise ValueError("Column 'exam_id' not found in demographic data.")
    else:
        print("'exam_id' column found in demographic data. Type: ", demographic_data['exam_id'].dtype)

    #ensure that alls exam id in demographic_data are equal to the exam ids in clinical_data
    if not demographic_data['exam_id'].isin(clinical_data['acc_anon']).all():
        raise ValueError("Some exam_ids in demographic_data are not present in clinical_data.") 

    # ---------------------------
    # LOAD INPUT METADATA (FIXED)
    # ---------------------------
    input_data = pd.read_csv("data/embed/asymirai_input/EMBED_OpenData_metadata_screening_2D_complete_exams.csv")
    if 'exam_id' not in input_data.columns:
        raise ValueError("Column 'exam_id' not found in input data.")
    else:
        print("'exam_id' column found in input data. Type: ", input_data['exam_id'].dtype)

    #ensure all exam ids in input data are also within deomographic data

    if not input_data['exam_id'].isin(demographic_data['exam_id']).all():
        #print an example 
        missing_exam_ids = input_data.loc[~input_data['exam_id'].isin(demographic_data['exam_id']), 'exam_id']
        print("Missing exam_ids in demographic_data: ", missing_exam_ids.tolist())
        raise ValueError("Some exam_ids in input_data are not present in demographic_data.")


    IMAGE_PATH_COL = "dicom_path"
    if IMAGE_PATH_COL not in input_data.columns:
        raise ValueError(f"Missing column: {IMAGE_PATH_COL}")

    # ---------------------------
    # MERGE
    # ---------------------------
    print("Merging demographics...")
    input_data = input_data.merge(
        demographic_data,
        on=["patient_id", "exam_id"],
        how="left"
    )

    # ---------------------------
    # CHECK S3 FILES
    # ---------------------------
    tqdm.pandas()
    input_data["png_missing"] = input_data[IMAGE_PATH_COL].progress_apply(
        lambda p: check_file_not_exists_in_s3(all_s3_keys, p)
    )

    # ---------------------------
    # SAVE
    # ---------------------------
    input_data.to_csv(
        "data/embed/asymirai_input/EMBED_OpenData_metadata_screening_2D_complete_exams_with_demographics.csv",
        index=False
    )

    print("Done.")


if __name__ == "__main__":
    main()
