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

# these let us set the number of rows/cols of our dataframes we want to see
# EMBED has a lot of columns so it's a good idea to increase this from the default


REQUIRED_VIEWS = {"CC", "MLO"}
REQUIRED_SIDES = {"L", "R"}
REQUIRED_PAIRS = {("CC", "L"), ("CC", "R"), ("MLO", "L"), ("MLO", "R")}



pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.options.mode.chained_assignment = None
tqdm.pandas() # initialize tqdm wrapper for pandas.apply


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

def get_incomplete_exams(df):
    incomplete_exams = []    

    for exam_id, exam_df in df.groupby("acc_anon"):

        # set of (view, side) pairs present in this exam
        present = set(
            zip(
                exam_df["ViewPosition"],
                exam_df["ImageLateralityFinal"]
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
    print("Initial meta statistics:")
    print_dataset_stats(meta_df)

    # ---- Ensure datetime for exam_date logic later ----
    # If already datetime, this is harmless.
    meta_df["study_date_anon"] = pd.to_datetime(meta_df["study_date_anon"], errors="coerce")

     # ---- 2D only ----
    meta_df = meta_df[meta_df["FinalImageType"] == "2D"].copy()
    print("After filtering to 2D images:")
    print_dataset_stats(meta_df)

    # ---- drop incomplete exams ----
    incomplete_exams = set(get_incomplete_exams(meta_df))
    meta_df = meta_df[~meta_df["acc_anon"].isin(incomplete_exams)].copy()
    print("After filtering out incomplete exams:")
    print_dataset_stats(meta_df)

    # ---- keep canonical views only ----
    meta_df = meta_df[
        meta_df["ViewPosition"].isin(REQUIRED_VIEWS)
        & meta_df["ImageLateralityFinal"].isin(REQUIRED_SIDES)
    ].copy()

     # ---- sanity: exactly 4 images per exam ----
    counts = meta_df.groupby("acc_anon").size()
    if not counts.eq(4).all():
        bad = counts[counts != 4]
        print("Exams with incorrect number of images:")
        print(bad)
        #remove these exams from the dataframe
        meta_df = meta_df[~meta_df['acc_anon'].isin(bad.index)].copy()

    print("✔ All remaining exams have exactly 4 images")
    return meta_df



def build_labels_and_followup(meta_df_reduced: pd.DataFrame, mag_contra_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - developed_cancer (bool): whether cancer occurs after this exam for that patient
      - years_to_cancer (float): time from this exam to first cancer date (else 100)
      - years_to_last_followup (float): time from this exam to last follow-up (last screen date or first cancer date)
    """

    # Ensure datetime
    meta_df_reduced["exam_date"] = pd.to_datetime(meta_df_reduced["exam_date"], errors="coerce")
    mag_contra_df["study_date_anon"] = pd.to_datetime(mag_contra_df["study_date_anon"], errors="coerce")

    # Define “cancer” rows (keep your definition, but now vectorized)
    cancer_df = mag_contra_df[mag_contra_df["path_severity"].isin([0, 1])].copy()

    # First cancer date per patient (earliest cancer-associated exam)
    first_cancer_date = (
        cancer_df.groupby("empi_anon")["study_date_anon"]
        .min()
        .rename("first_cancer_date")
        .reset_index()
        .rename(columns={"empi_anon": "patient_id"})
    )

    # Attach first cancer date to each exam row
    meta_df_reduced = meta_df_reduced.merge(
        first_cancer_date,
        on="patient_id",
        how="left",
        validate="many_to_one",
    )

    # developed_cancer if first_cancer_date exists and is after this exam_date
    meta_df_reduced["developed_cancer"] = (
        meta_df_reduced["first_cancer_date"].notna()
        & (meta_df_reduced["first_cancer_date"] > meta_df_reduced["exam_date"])
    )

    # years_to_cancer
    meta_df_reduced["years_to_cancer"] = (
        (meta_df_reduced["first_cancer_date"] - meta_df_reduced["exam_date"]).dt.days / 365.25
    )

    # default for no subsequent cancer
    meta_df_reduced.loc[~meta_df_reduced["developed_cancer"], "years_to_cancer"] = 100.0

    # last screening exam date per patient
    last_screen_date = (
        meta_df_reduced.groupby("patient_id")["exam_date"]
        .max()
        .rename("last_exam_date")
        .reset_index()
    )
    
    meta_df_reduced = meta_df_reduced.merge(
    last_screen_date,
    on='patient_id',
    how='left',
    validate='many_to_one')

    # default: follow-up ends at last screening exam
    meta_df_reduced['last_followup_date'] = meta_df_reduced['last_exam_date']
    # if cancer occurs AFTER this exam, follow-up ends at cancer date
    mask = (
        meta_df_reduced['first_cancer_date'].notna() &
        (meta_df_reduced['first_cancer_date'] > meta_df_reduced['exam_date'])
    )

    meta_df_reduced.loc[mask, 'last_followup_date'] = (
        meta_df_reduced.loc[mask, 'first_cancer_date']
    )

    # compute follow-up time
    meta_df_reduced['years_to_last_followup'] = (
        (meta_df_reduced['last_followup_date'] - meta_df_reduced['exam_date'])
        .dt.days / 365.25
    )

    # followup_df = last_screen_date.merge(first_cancer_date, on="patient_id", how="left")

    # # follow-up ends at last exam unless cancer occurs (then at first cancer date)
    # followup_df["last_followup_date"] = followup_df["last_exam_date"]
    # mask = followup_df["first_cancer_date"].notna()
    # followup_df.loc[mask, "last_followup_date"] = followup_df.loc[mask, "first_cancer_date"]

    # meta_df_reduced = meta_df_reduced.merge(
    #     followup_df[["patient_id", "last_followup_date"]],
    #     on="patient_id",
    #     how="left",
    #     validate="many_to_one",
    # )

    # meta_df_reduced["years_to_last_followup"] = (
    #     (meta_df_reduced["last_followup_date"] - meta_df_reduced["exam_date"]).dt.days / 365.25
    # )

    # Optional: drop helper column
    # meta_df_reduced = meta_df_reduced.drop(columns=["first_cancer_date"])

    return meta_df_reduced




#help function to print dataset statistics
def print_dataset_stats(df: pd.DataFrame):
    print(" Number of unique patients:", df['empi_anon'].nunique())
    print(" Number of unique exams:", df['acc_anon'].nunique())
    print(" Total number of exam images:", df.shape[0])

def ensure_key_column_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df['empi_anon'] = pd.to_numeric(df['empi_anon'])
    df['acc_anon'] = pd.to_numeric(df['acc_anon'])
    df['study_date_anon'] = pd.to_datetime(df['study_date_anon'])
    return df

if __name__ == "__main__":
    # ----------------------------
    # Load
    # ----------------------------
    mag_df = pd.read_csv("data/embed/tables/EMBED_OpenData_clinical.csv")
    meta_df = pd.read_csv("data/embed/tables/EMBED_OpenData_metadata.csv")

    meta_df = ensure_key_column_dtypes(meta_df)
    mag_df = ensure_key_column_dtypes(mag_df)

    # Ensure datetime early (helps a lot downstream)
    mag_df["study_date_anon"] = pd.to_datetime(mag_df.get("study_date_anon"), errors="coerce")
    meta_df["study_date_anon"] = pd.to_datetime(meta_df.get("study_date_anon"), errors="coerce")

    # ----------------------------
    # Screening vs diagnostic counts (safe contains)
    # ----------------------------
    mag_df["screen_exam"] = mag_df["desc"].fillna("").str.contains("screen", case=False)

    print("Screening vs Diagnostic exam counts:")
    display(mag_df["screen_exam"].value_counts())

    # ----------------------------
    # Drop rows with no description BEFORE contralateral correction
    # ----------------------------
    mag_df = mag_df.dropna(subset=["desc"]).copy()
    # ----------------------------
    # Contralateral correction
    # ----------------------------
    mag_contra_df = embed_toolkit.correct_contralaterals(mag_df)
    mag_contra_df = ensure_key_column_dtypes(mag_contra_df)

    # ----------------------------
    # Derive laterality & worst findings per exam (your existing logic)
    # ----------------------------
    mag_contra_df["exam_laterality"] = mag_contra_df.progress_apply(get_exam_laterality, axis=1)  # type: ignore

    worst_br_dict = mag_contra_df.groupby("acc_anon").progress_apply(get_worst_br).to_dict()  # type: ignore
    mag_contra_df["exam_birads"] = mag_contra_df["acc_anon"].map(worst_br_dict)

    worst_path_dict = (
        mag_contra_df[mag_contra_df["path_severity"].notna()]
        .groupby("acc_anon")
        .progress_apply(get_worst_ps)
        .to_dict()  # type: ignore
    )
    mag_contra_df["exam_path_severity"] = mag_contra_df["acc_anon"].map(worst_path_dict)

    mag_contra_df.embed.summarize("Magview - before filtering metadata")
    
    # ----------------------------
    # Metadata filtering (2D, complete exams, canonical views)
    # ----------------------------
    meta_df = preprocess_metadata(meta_df)
     # ----------------------------
    # Screening exam IDs (safe contains + unique)
    # ----------------------------
    screening_exam_ids = mag_contra_df[
        mag_contra_df["desc"].fillna("").str.contains("screen", case=False)
    ]["acc_anon"].unique()

    meta_df = meta_df[meta_df["acc_anon"].isin(screening_exam_ids)].copy()
    print("After filtering to screening exams only:")
    print_dataset_stats(meta_df)

    # Re-check after screening filter: still exactly 4 images/exam
    counts = meta_df.groupby("acc_anon").size()
    if not counts.eq(4).all():
        bad = counts[counts != 4]
        raise ValueError(f"After screening filter, exams not equal to 4 images:\n{bad.head(10)}")

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

    meta_df_reduced = meta_df[cols_to_retain].copy()

    meta_df_reduced = meta_df_reduced.rename(
        columns={
            "empi_anon": "patient_id",
            "acc_anon": "exam_id",
            "study_date_anon": "exam_date",
            "anon_dicom_path": "dicom_path",
            "ViewPosition": "view",
            "ImageLateralityFinal": "laterality",
        }
    )

    # ----------------------------
    # Build cancer outcomes + follow-up (vectorized)
    # ----------------------------
    meta_df_reduced = build_labels_and_followup(meta_df_reduced, mag_contra_df)

    # Add desc placeholder (as you had)
    meta_df_reduced["desc"] = "Screening Bilateral"

    # ----------------------------
    # Save
    # ----------------------------
    out_path = "data/embed/asymirai_input/EMBED_OpenData_metadata_screening_2D_complete_exams.csv"
    meta_df_reduced.to_csv(out_path, index=False)

    print("✔ Saved:", out_path)
    print("Rows:", len(meta_df_reduced))
    print("Unique exams:", meta_df_reduced["exam_id"].nunique())
    print("Unique patients:", meta_df_reduced["patient_id"].nunique())

