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

# This chunk comes from the dataset class itself, where some incomplete exams are excluded
def get_incomplete_exams(metadata_frame):
    incomplete_exams = []
    for i, eid in tqdm(enumerate(metadata_frame['acc_anon'].unique()), total=metadata_frame['acc_anon'].unique().shape[0]):

        # cur_exam = metadata_frame[metadata_frame['acc_anon'].values == eid]
        cur_exam = metadata_frame[metadata_frame['acc_anon'] == eid]


        patient_exam = {'MLO': {'L': None, 'R': None},
                        'CC': {'L': None, 'R': None}}
        for view in patient_exam.keys():
            def indices_for_side_view(side):
                indices = np.logical_and(cur_exam['ViewPosition'].values == view, cur_exam['ImageLateralityFinal'].values == side)
                return indices

            if len(cur_exam[indices_for_side_view('L')]['anon_dicom_path']) == 0:
                incomplete_exams.append(eid)
                continue
            else:
                for laterality in ['L', 'R']: 
                    if len(cur_exam[indices_for_side_view(laterality)]['anon_dicom_path']) == 0:
                        incomplete_exams.append(eid)
                        continue
                        
    return incomplete_exams

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
    # load clinical and metadata dataframes
    mag_df = pd.read_csv("data/embed/tables/EMBED_OpenData_clinical.csv")
    meta_df = pd.read_csv("data/embed/tables/EMBED_OpenData_metadata.csv")

    meta_df = ensure_key_column_dtypes(meta_df)
    mag_df = ensure_key_column_dtypes(mag_df)
    
    # create a helper column for exam screen-status
    mag_df['screen_exam'] = mag_df.desc.str.contains('screen', case=False)

    #count number of screening vs diagnostic exams
    print("Screening vs Diagnostic exam counts:")
    display(mag_df['screen_exam'].value_counts())

    mag_df.embed.summarize("Magview")
    # apply contralateral correction, drop any data for exams with no description
    mag_df = mag_df.dropna(subset="desc")
    mag_contra_df = embed_toolkit.correct_contralaterals(mag_df)
    mag_contra_df = ensure_key_column_dtypes(mag_contra_df)
    # derive exam laterality from their descriptions
    mag_contra_df["exam_laterality"] = mag_contra_df.progress_apply(get_exam_laterality, axis=1) # type: ignore
    # apply the 'get_worst_br' function to the data (grouped by exam) and output [exam > birads] mappings as a dict
    worst_br_dict = mag_contra_df.groupby('acc_anon').progress_apply(get_worst_br).to_dict() # type: ignore

    # map back to magview
    mag_contra_df['exam_birads'] = ''
    mag_contra_df['exam_birads'] = mag_contra_df['acc_anon'].map(worst_br_dict)
    # apply the 'get_worst_ps' function to the data (grouped by exam) and output [exam > pathology] mappings as a dict
    # don't apply it to exam findings with no path severity (since they can't affect results)
    worst_path_dict = mag_contra_df[~pd.isnull(mag_contra_df.path_severity)].groupby('acc_anon').progress_apply(get_worst_ps).to_dict() # type: ignore

    # map back to magview
    mag_contra_df['exam_path_severity'] = np.nan
    mag_contra_df['exam_path_severity'] = mag_contra_df['acc_anon'].map(worst_path_dict)



    print("Initial meta statistics:")
    print_dataset_stats(meta_df)

    # #remove any where the exame year is after 2020
    # meta_df = meta_df[meta_df['study_date_anon'].dt.year <= 2021]
    # print("After filtering to exams before 2021:")
    # print_dataset_stats(meta_df)

    # Filter to 2d images
    meta_df = meta_df[meta_df['FinalImageType'] == '2D']
    print("After filtering to 2D images:")
    print_dataset_stats(meta_df)

    incomplete_exams = get_incomplete_exams(meta_df)
    meta_df = meta_df[~meta_df['acc_anon'].isin(incomplete_exams)]
    print("After filtering out incomplete exams:")
    print_dataset_stats(meta_df)
    
    screening_exam_ids = mag_contra_df[mag_contra_df.desc.str.contains('screen', case=False)]['acc_anon'].unique()

    # Finally, remove diagnostic exams  but use the mga_df to identify screening exams
    #for each exam, we need to identify if it's a screening exam or not by looking at the mag_df and the 'desc' column
    meta_df = meta_df[meta_df['acc_anon'].isin(screening_exam_ids)]
    print("After filtering to screening exams only:")
    print_dataset_stats(meta_df)

    #now save new cdv with only the columns I want to retain and rename some columns
    cols_to_retain = ['empi_anon', 'acc_anon', 'study_date_anon', 'anon_dicom_path', 'ViewPosition', 'ImageLateralityFinal', 'spot_mag']
    meta_df_reduced = meta_df[cols_to_retain]
    meta_df_reduced = meta_df_reduced.rename(columns={
        'empi_anon': 'patient_id',
        'acc_anon': 'exam_id',
        'study_date_anon': 'exam_date',
        'anon_dicom_path': 'dicom_path',
        'ViewPosition': 'view',
        'ImageLateralityFinal': 'laterality',
        'spot_mag': 'spot_mag'
    })

    #next I need to find patents who developed cancer following a screen exam and record years until diagnosis
    #first, create a dataframe with only cancer diagnoses
    cancer_df = mag_contra_df[mag_contra_df['path_severity'].isin([0, 1])]
    #create a dict mapping exam_id to (path_severity, exam_date)
    cancer_exam_dict = {}
    for i, row in cancer_df.iterrows():
        exam_id = row['acc_anon']
        path_severity = row['path_severity']
        exam_date = row['study_date_anon']
        cancer_exam_dict[exam_id] = (path_severity, exam_date)
    #now, for each exam in meta_df_reduced, check if there's a subsequent cancer diagnosis for that patient
    meta_df_reduced['developed_cancer'] = False
    meta_df_reduced['years_to_cancer'] = 100 #set to 100 by default for no cancer
    
    for i, row in tqdm(meta_df_reduced.iterrows(), total=meta_df_reduced.shape[0]):
        patient_id = row['patient_id']
        exam_id = row['exam_id']
        exam_date = row['exam_date']
        #find all cancer exams for this patient
        patient_cancer_exams = cancer_df[cancer_df['empi_anon'] == patient_id]
        for j, cancer_row in patient_cancer_exams.iterrows():
            cancer_exam_id = cancer_row['acc_anon']
            cancer_exam_date = cancer_row['study_date_anon']
            if cancer_exam_date > exam_date:
                #calculate years to cancer
                years_to_cancer = (cancer_exam_date - exam_date).days / 365.25
                #round to nearest integer
                # years_to_cancer = round(years_to_cancer)
                meta_df_reduced.at[i, 'developed_cancer'] = True
                if years_to_cancer < meta_df_reduced.at[i, 'years_to_cancer']:
                    meta_df_reduced.at[i, 'years_to_cancer'] = years_to_cancer


    # first cancer exam per patient (if they ever had cancer)
    first_cancer_date = (
        cancer_df
        .groupby('empi_anon')['study_date_anon']
        .min()  # earliest cancer exam
        .rename('first_cancer_date')
        .reset_index()
        .rename(columns={'empi_anon': 'patient_id'})
    )

    # last screening exam date per patient (from screening-only meta_df_reduced)
    last_screen_date = (
        meta_df_reduced
        .groupby('patient_id')['exam_date']
        .max()
        .rename('last_exam_date')
        .reset_index()
    )

    # merge to get both last_exam_date and (optional) first_cancer_date
    followup_df = last_screen_date.merge(first_cancer_date, on='patient_id', how='left')

    # start by assuming follow-up ends at last exam
    followup_df['last_followup_date'] = followup_df['last_exam_date']

    # if a patient has cancer, follow-up ends at first_cancer_date instead
    mask = followup_df['first_cancer_date'].notna()
    followup_df.loc[mask, 'last_followup_date'] = followup_df.loc[mask, 'first_cancer_date']

    # attach last_followup_date to every exam row
    meta_df_reduced = meta_df_reduced.merge(
        followup_df[['patient_id', 'last_followup_date']],
        on='patient_id',
        how='left'
    )

    # compute years from this exam until last cancer-free date
    meta_df_reduced['years_to_last_followup'] = (
        (meta_df_reduced['last_followup_date'] - meta_df_reduced['exam_date']).dt.days / 365.25
    )

    meta_df_reduced['desc'] = 'Screening Bilateral'

    # # just in case: exams after last_followup_date → set to 0
    # meta_df_reduced.loc[meta_df_reduced['years_to_last_followup'] < 0, 'years_to_last_followup'] = 0


    meta_df_reduced.to_csv("data/embed/asymirai_input/EMBED_OpenData_metadata_screening_2D_complete_exams.csv", index=False)
