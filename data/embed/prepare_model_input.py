import os
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import pickle
from IPython.display import display
import pydicom
from dataclasses import dataclass, field
from tableone import TableOne
import embed_toolkit # version 0.2.*
from tqdm import tqdm
from constants_val import *
import boto3
# these let us set the number of rows/cols of our dataframes we want to see
# EMBED has a lot of columns so it's a good idea to increase this from the default


REQUIRED_VIEWS = {"CC", "MLO"}
REQUIRED_SIDES = {"L", "R"}
REQUIRED_PAIRS = {("CC", "L"), ("CC", "R"), ("MLO", "L"), ("MLO", "R")}

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.options.mode.chained_assignment = None
tqdm.pandas() # initialize tqdm wrapper for pandas.apply


s3 = boto3.client("s3")
BUCKET_NAME = "embdedpng"

# ---------------------------
# S3 HELPERS
# ---------------------------
def list_all_s3_keys(bucket, prefix=None):
    s3_keys_cache = "data/embed/asymirai_input/all_s3_keys.txt"
    if os.path.exists(s3_keys_cache):
        with open(s3_keys_cache, "r") as f:
            all_s3_keys = set(line.strip() for line in f)
    else:
        all_s3_keys = set()
        paginator = s3.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                all_s3_keys.add(obj["Key"])
                
        os.makedirs(os.path.dirname(s3_keys_cache), exist_ok=True)
        with open(s3_keys_cache, "w") as f:
            for key in all_s3_keys:
                f.write(f"{key}\n")
    return all_s3_keys



def build_s3_key(path):
    s3_key = path.replace("images", "png_tmp").replace(".dcm", ".png")
    return s3_key.replace("\\", "/")
    # return os.path.normpath(s3_key).replace("\\", "/")

def check_file_exists_in_s3(all_s3_keys, path):
    s3_key = build_s3_key(path)
    return s3_key in all_s3_keys

def get_exam_laterality(row: pd.Series) -> str | None:
    # extract description and lowercase it
    finding_desc = row.desc.lower()
    
    if ("bilat" in finding_desc):
        return "B"
    elif ("left" in finding_desc):
        return "L"
    elif ("right" in finding_desc):
        return "R"
    else:
        return None
    
    # for each screening exam, take the most severe birads as the representative
def get_worst_ps(group):
    return group.path_severity.min()

def get_worst_br(group):
    exam_desc = group.desc.tolist()[0]
    if "screen" in exam_desc.lower():
        br_to_val_dict = {
            'A': 0, # 'A' maps to birads 0
            'B': 1, # 'B' maps to birads 2
            'N': 2  # 'N' maps to birads 1
        }
    else:
        br_to_val_dict = {
            'N': 5, # 'N' maps to birads 1
            'B': 4, # 'B' maps to birads 2
            'P': 3, # 'P' maps to birads 3
            'S': 2, # 'S' maps to birads 4
            'M': 1, # 'M' maps to birads 5
            'K': 0  # 'K' maps to birads 6
        }
        
    val_to_br_dict = {v:k for k,v in br_to_val_dict.items()}
    worst_br_val = min(group.asses.map(br_to_val_dict).tolist())
    return val_to_br_dict.get(worst_br_val, '')

def get_incomplete_exams(df: pd.DataFrame) -> list:
    incomplete_exams = []

    #save all unique exam ids to csv
    df.groupby("acc_anon").nunique().value_counts()
    for exam_id, exam_df in df.groupby("acc_anon"):
        present = set(
            zip(
                exam_df[EMBED_VIEW_COL],
                exam_df[EMBED_SIDE_COL]
            )
        )

        # check all required combinations
        for view in REQUIRED_VIEWS:
            for side in REQUIRED_SIDES:
                if (view, side) not in present:
                    incomplete_exams.append(exam_id)
                    break
            else:
                continue
            break

    return incomplete_exams


def preprocess_metadata(meta_df: pd.DataFrame) -> pd.DataFrame:


    # Only consider 2D images
    meta_df = meta_df[meta_df[EMBED_IMAGE_TYPE_COL] == "2D"]
    print("After filtering to 2D images:")
    print_dataset_stats(meta_df)
    
    # ---- drop incomplete exams ----
    incomplete_exams = set(get_incomplete_exams(meta_df))
    meta_df = meta_df[~meta_df["acc_anon"].isin(incomplete_exams)].copy()
    print("After filtering out incomplete exams:")
    print_dataset_stats(meta_df)


    
    # ---- keep canonical views only ---
    meta_df = meta_df[
        meta_df["ViewPosition"].isin(REQUIRED_VIEWS)
        & meta_df["ImageLateralityFinal"].isin(REQUIRED_SIDES)
    ].copy()

    print("After filtering to canonical views only:")
    print_dataset_stats(meta_df)

     # ---- sanity: exactly 4 images per exam ----
    counts = meta_df.groupby("acc_anon").size()
    if not counts.eq(4).all():
        bad = counts[counts != 4]
        # print("Exams with incorrect number of images:")
        # print(len(bad))
        #remove these exams from the dataframe
        meta_df = meta_df[~meta_df['acc_anon'].isin(bad.index)].copy()
    
    print("After ensuring exactly 4 images per exam:")
    print_dataset_stats(meta_df)

    #drop all rows that are not screening exams based on StudyDescription column
    screening_mask = meta_df[EMBED_PROCEDURE_COL].str.contains('screen', case=False, na=False)
    meta_df = meta_df[screening_mask].copy()

    print("After filtering to screening exams only:")
    print_dataset_stats(meta_df)

    #only keep screening exams
    return meta_df



def build_labels_and_followup(meta_df: pd.DataFrame, mag_contra_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - developed_cancer (bool): whether cancer occurs after this exam for that patient
      - years_to_cancer (float): time from this exam to first cancer date (else 100)
      - years_to_last_followup (float): time from this exam to last follow-up (last screen date or first cancer date)
    """
    # Define “cancer” rows (keep your definition, but now vectorized)
    cancer_df = mag_contra_df[mag_contra_df["path_severity"].isin([0, 1])].copy()

    # First cancer date per patient (earliest cancer-associated exam)
    first_cancer_date = (
        cancer_df.groupby("empi_anon")["study_date_anon"]
        .min()
        .rename("first_cancer_date")
        .reset_index()
        .rename(columns={"empi_anon": "empi_anon"})
    )

    # Attach first cancer date to each exam row
    meta_df = meta_df.merge(
        first_cancer_date,
        on="empi_anon",
        how="left",
        validate="many_to_one",
    )

    # developed_cancer if first_cancer_date exists and is after this exam_date
    meta_df["developed_cancer"] = (
        meta_df["first_cancer_date"].notna()
        & (meta_df["first_cancer_date"] > meta_df["study_date_anon"])
    )

    # years_to_cancer
    meta_df["years_to_cancer"] = (
        (meta_df["first_cancer_date"] - meta_df["study_date_anon"]).dt.days / 365.25
    )

    # default for no subsequent cancer
    meta_df.loc[~meta_df["developed_cancer"], "years_to_cancer"] = 100.0
    # last screening exam date per patient
    last_screen_date = (
        meta_df.groupby("empi_anon")["study_date_anon"]
        .max()
        .rename("last_exam_date")
        .reset_index()
    )
    
    meta_df = meta_df.merge(
    last_screen_date,
    on='empi_anon',
    how='left',
    validate='many_to_one')

    # default: follow-up ends at last screening exam
    meta_df['last_followup_date'] = meta_df['last_exam_date']
    # if cancer occurs AFTER this exam, follow-up ends at cancer date
    mask = (
        meta_df['first_cancer_date'].notna() &
        (meta_df['first_cancer_date'] > meta_df['study_date_anon'])
    )

    meta_df.loc[mask, 'last_followup_date'] = (
        meta_df.loc[mask, 'first_cancer_date']
    )

    # compute follow-up time
    meta_df['years_to_last_followup'] = (
        (meta_df['last_followup_date'] - meta_df['study_date_anon'])
        .dt.days / 365.25
    )

    return meta_df


def add_cliniclal_data(meta_df: pd.DataFrame, mag_df: pd.DataFrame) -> pd.DataFrame:
     #ensure that alls exam id in demographic_data are equal to the exam ids in clinical_data
    if not meta_df['acc_anon'].isin(mag_df['acc_anon']).all():
        raise ValueError("Some exam_ids in demographic_data are not present in clinical_data.")
    
    print("Merging clinical data...")
    print_dataset_stats(meta_df)

    clinical_cols = [
        "empi_anon",
        "acc_anon",
        "GENDER_DESC",
        "ETHNICITY_DESC",
        "MARITAL_STATUS_DESC",
        "age_at_study",
        "tissueden",
        "path_severity",
        'desc',
        'asses'
    ]

    agg_map = {
    "GENDER_DESC": "first",
    "ETHNICITY_DESC": "first",
    "MARITAL_STATUS_DESC": "first",
    "age_at_study": "first",
    "tissueden": "first",
    "desc": "first",
    "asses": "first",          # keep exam description
    "path_severity": "max",   # cancer logic
    }


    #  # Subset BEFORE merge
    #  #only innclude screening exams i.e. when desc contains 'screen'
    # mag_df = mag_df[mag_df['desc'].str.contains('screen', case=False, na=False)]

    mag_subset = (
        mag_df[clinical_cols]
        .groupby(["empi_anon", "acc_anon"], as_index=False)
        .agg(agg_map)
    )

    #check for duplicate rows in mag_subset
    print("Number of duplicate rows in clinical data subset:", mag_subset.duplicated().sum())
    #drop duplicate rows
    mag_subset = mag_subset.drop_duplicates()

    #for each exam in meta_df, get the corresponding patient data from mag_df
    meta_df = meta_df.merge(
        mag_subset,
        on=["empi_anon", "acc_anon"],
        how="left",
        validate="many_to_one",  # many meta rows → one clinical row
    )


    print_dataset_stats(meta_df)
    return meta_df


#help function to print dataset statistics
def print_dataset_stats(df: pd.DataFrame):
    print(" Number of unique patients:", df['empi_anon'].nunique())
    print(" Number of unique exams:", df['acc_anon'].nunique())
    print(" Total number of images:", df.shape[0])
    print(" Number of duplicate rows:", df.duplicated().sum())
    print("")

def ensure_key_column_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    # df['empi_anon'] = pd.to_numeric(df['empi_anon'])
    # df['acc_anon'] = pd.to_numeric(df['acc_anon'])
    df['study_date_anon'] = pd.to_datetime(df['study_date_anon'])
    return df

def check_png_files_exist_in_s3(meta_df: pd.DataFrame) -> pd.DataFrame:
    all_s3_keys = list_all_s3_keys(BUCKET_NAME, prefix="png_images/")
    print(f"Total S3 files: {len(all_s3_keys)}")

    tqdm.pandas()
    meta_df["png_exists_in_s3"] = meta_df['dicom_path'].progress_apply(
        lambda p: check_file_exists_in_s3(all_s3_keys, p)
    )
    num_missing = len(meta_df[~meta_df['png_exists_in_s3']])
    print(f"Number of missing png files in s3: {num_missing} out of {len(meta_df)}")
    
    meta_df["s3_png_path"] = meta_df['dicom_path'].apply(build_s3_key)

    return meta_df

if __name__ == "__main__":
    # ----------------------------
    # Load
    # ----------------------------
    mag_df = pd.read_csv("data/embed/tables/EMBED_OpenData_clinical.csv")
    meta_df = pd.read_csv("data/embed/tables/EMBED_OpenData_metadata.csv")

    meta_df = ensure_key_column_dtypes(meta_df)
    mag_df = ensure_key_column_dtypes(mag_df)

    print("Initial meta statistics:")
    print_dataset_stats(meta_df)
    #ensure all exam ids in meta_df are acc_anon in mag_df
    if not meta_df['acc_anon'].isin(mag_df['acc_anon']).all():
        num_rows_with_missing_acc_anon = len(meta_df.loc[~meta_df['acc_anon'].isin(mag_df['acc_anon'])])
        meta_df = meta_df[meta_df['acc_anon'].isin(mag_df['acc_anon'])].copy()
        print("Removed", num_rows_with_missing_acc_anon, "rows with missing acc_anon.")
        print_dataset_stats(meta_df)

    #ensure empi_anon  in metda are also in mag_df
    if not meta_df['empi_anon'].isin(mag_df['empi_anon']).all():
        num_rows_with_missing_empi_anon = len(meta_df.loc[~meta_df['empi_anon'].isin(mag_df['empi_anon'])])
        meta_df = meta_df[meta_df['empi_anon'].isin(mag_df['empi_anon'])].copy()
        print("Removed", num_rows_with_missing_empi_anon, "rows with missing empi_anon.")
        print_dataset_stats(meta_df)

    # ---------------------------
    # Metadata filtering (2D, complete exams, canonical views)
    # ----------------------------
    meta_df = preprocess_metadata(meta_df)

    # ----------------------------
    # Reduce columns + rename
    # ----------------------------
    cols_to_retain = [
        "empi_anon",
        "acc_anon",
        "study_date_anon",
        "anon_dicom_path",
        "ViewPosition",
        "ImageLateralityFinal",
        "spot_mag",
    ]

    meta_df = meta_df[cols_to_retain]

     # ----------------------------
    # Build cancer outcomes + follow-up (vectorized)
    # ----------------------------
    meta_df = build_labels_and_followup(meta_df, mag_df)

    print_dataset_stats(meta_df)

    # meta_df["desc"] = "Screening Bilateral"
    print_dataset_stats(meta_df)
    # Rename columns to match expected names

    meta_df = add_cliniclal_data(meta_df, mag_df)

    meta_df = meta_df.rename(
        columns={
            "empi_anon": "patient_id",
            "acc_anon": "exam_id",
            "study_date_anon": "exam_date",
            "anon_dicom_path": "dicom_path",
            "ViewPosition": "view",
            "ImageLateralityFinal": "laterality",
        }
    )

    # ---------------------------
    #check if png files already exist in s3
    # ---------------------------
    check_png_files_exist_in_s3(meta_df)


    out_path = "data/embed/asymirai_input/EMBED_OpenData_metadata_screening_2D_complete_exams.csv"
    meta_df.to_csv(out_path, index=False)

    print("✔ Saved:", out_path)
    print("Rows:", len(meta_df))
    print("Unique exams:", meta_df["exam_id"].nunique())
    print("Unique patients:", meta_df["patient_id"].nunique())
