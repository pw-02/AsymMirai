import os
from typing import Union
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

def aggregate_bsides(group):
    # applied to exam groups
    bside_list = group.bside.unique().tolist()
    # return the only bside if we only have 1 (this should never be 0 since NaN is included)
    # this should return an IndexError if it ever is 0
    if len(bside_list) == 1:
        return bside_list[0]

    # otherwise aggregate bilateral bsides
    elif ('B' in bside_list) or (('L' in bside_list) & ('R' in bside_list)):
        return 'B'
    # handle left bsides with no right or 'B' (other is a NaN)
    elif ('L' in bside_list):
        return 'L'
    # handle right bsides with no left or 'B' (other is a NaN)
    elif ('R' in bside_list):
        return 'R'
    else:
        return 'ERROR'

def get_bside_aggregation_dict(df: pd.DataFrame) -> dict[float, str]:
    # we only need to apply this to exam findings with no exam-level pathology registered
    path_na_mask: pd.Series[bool] = pd.isna(mag_contra_df.exam_path_severity)

    # or exam findings where the finding-level path severity matches the exam-level path severity
    path_match_mask: pd.Series[bool] = (
        ~pd.isna(mag_contra_df.exam_path_severity)
        & (mag_contra_df.path_severity == mag_contra_df.exam_path_severity)
    )

    # define a list of columns to consider
    col_list: list[str] = ['acc_anon', 'empi_anon', 'study_date_anon', 'exam_birads', 'exam_path_severity', 'bside']

    # get the relevant subset of the data, then 
    df_subset: pd.DataFrame = df.loc[path_na_mask | path_match_mask, col_list]

    # drop any duplicate rows, group by exam, then apply the agg func and output a [exam > bside] mapping dict
    bside_agg_dict: dict[float, str] = (
        df_subset
        .drop_duplicates()
        .groupby('acc_anon')
        .progress_apply(aggregate_bsides) # type: ignore
        .to_dict()
    )
    return bside_agg_dict

def get_followup_map_dict(df: pd.DataFrame, followup_df: pd.DataFrame, time_delta: Union[int, float] = 180):
    # don't consider followups with an undefined exam_birads (indicates an invalid birads for that stage)
    # followup_df = followup_df[(followup_df.exam_birads != '') & ~pd.isna(followup_df.exam_birads)]
    
    # time delta in days
    # expects df to have been corrected for contralateral findings (and for no NA finding sides to exist)
    # previous versions assumed no 'B' findings but this does not
    merge_df = df.merge(followup_df, on='empi_anon', how='inner', suffixes=(None, "_fu"))
    
    # ensure exam laterality match, L==L, R==R, or either original/followup is bilateral
    merge_df = merge_df.loc[
        (merge_df.exam_laterality==merge_df.exam_laterality_fu)
        | (merge_df.exam_laterality=="B")
        | (merge_df.exam_laterality_fu=="B")
    ]

    # exclude followups with an invalid time delta
    merge_df["fu_delta"] = (merge_df.study_date_anon_fu - merge_df.study_date_anon).dt.days
    merge_df = merge_df.loc[(merge_df.fu_delta >= 0) & (merge_df.fu_delta <= time_delta)]

    # get the accession of the first valid followup for each exam and output a dict of mappings
    map_dict = merge_df.sort_values('fu_delta').drop_duplicates('acc_anon', keep='first').set_index('acc_anon')['acc_anon_fu'].to_dict()
    return map_dict
    



if __name__ == "__main__":
    # load clinical and metadata dataframes
    mag_df = pd.read_csv("data/embed/tables/EMBED_OpenData_clinical.csv")
    meta_df = pd.read_csv("data/embed/tables/EMBED_OpenData_metadata.csv")
    # ensure key columns have the correct data types
    mag_df['empi_anon'] = pd.to_numeric(mag_df['empi_anon'])
    mag_df['acc_anon'] = pd.to_numeric(mag_df['acc_anon'])
    mag_df['study_date_anon'] = pd.to_datetime(mag_df['study_date_anon'])
    # create a helper column for exam screen-status
    mag_df['screen_exam'] = mag_df.desc.str.contains('screen', case=False)

    # summarize dataframe contents
    mag_df.embed.summarize("Magview")

    # apply contralateral correction, drop any data for exams with no description
    mag_df = mag_df.dropna(subset="desc")
    mag_contra_df = embed_toolkit.correct_contralaterals(mag_df)

    # correct column dtypes
    mag_contra_df['study_date_anon'] = pd.to_datetime(mag_contra_df['study_date_anon'])
    mag_contra_df['acc_anon'] = pd.to_numeric(mag_contra_df['acc_anon'])
    mag_contra_df['empi_anon'] = pd.to_numeric(mag_contra_df['empi_anon'])
    mag_contra_df["exam_laterality"] = mag_contra_df.progress_apply(get_exam_laterality, axis=1) # type: ignore

    #apply the 'get_worst_br' function to the data (grouped by exam) and output [exam > birads] mappings as a dict
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

    # apply the agg function and get a dict of exam mappings
    bside_agg_dict: dict[float, str] = get_bside_aggregation_dict(mag_contra_df)

    # map the agg dict back to the dataframe
    mag_contra_df['exam_bside'] = mag_contra_df['acc_anon'].map(bside_agg_dict)
    mag_contra_df['exam_bside'].value_counts(dropna=False)

    # correct exam birads to 'A' for any 'negative' exams with a pathology assigned directly to the screen findings
    # a review of the associated rad. notes indicated these were all addended to 'A's
    addended_exam_list = mag_contra_df[
        (mag_contra_df.screen_exam == True) 
        & mag_contra_df.exam_birads.isin(['N', 'B']) 
        & ~pd.isna(mag_contra_df.exam_path_severity)
    ].acc_anon.unique().tolist()

    mag_contra_df.loc[mag_contra_df.acc_anon.isin(addended_exam_list), 'exam_birads'] = 'A'

    scr_br_0_list = mag_contra_df[
        (mag_contra_df.screen_exam == True)
        & (mag_contra_df.exam_birads.isin(['A']))
    ].acc_anon.unique().tolist()

    scr_br_12_list = mag_contra_df[
        (mag_contra_df.screen_exam == True)
        & (mag_contra_df.exam_birads.isin(['N', 'B']))
    ].acc_anon.unique().tolist()

    mag_contra_df['scr_br_0'] = False
    mag_contra_df.loc[mag_contra_df.acc_anon.isin(scr_br_0_list), 'scr_br_0'] = True

    mag_contra_df['scr_br_12'] = False
    mag_contra_df.loc[mag_contra_df.acc_anon.isin(scr_br_12_list), 'scr_br_12'] = True

    followup_cols = ['acc_anon', 'empi_anon', 'study_date_anon', 'exam_laterality', 'exam_birads', 'exam_path_severity', 'exam_bside']
    # get subset of magview corresponding to diagnostic exams
    mag_diag = mag_contra_df.loc[mag_contra_df.desc.str.contains('diag', case=False)]
    mag_diag = mag_diag[followup_cols].drop_duplicates()

    # ensure we have exactly 1 row for each exam
    print('any duplicate exam rows?', mag_diag.acc_anon.nunique() != len(mag_diag))



    # we only want to consider screening exams in our target subset, so we'll drop any diagnostic cases present
    mag_sample_df = mag_sample_df[mag_sample_df.desc.str.contains('screen', case=False)]

    # get birads 0 diagnostic followup map dict
    # mag_br0 = mag_contra_df[mag_contra_df.scr_br_0 == True]
    mag_br0 = mag_sample_df[mag_sample_df.scr_br_0 == True]


    br0_dx_map_dict = get_followup_map_dict(mag_br0, mag_diag, time_delta=180)
    print(f"{len(br0_dx_map_dict)} valid DX followups found for Screen BIRADS 0s")











 