import math
import os
import numpy as np
import pandas as pd
import datetime
import pathlib

from enum import Enum
# Everything on util is loaded directly in datasets/__init__.py, so it is not necessary to import it directly.
import re


def col_search(df, pattern, ignore_case=True):
    """
    Find columns that match a pattern
    """
    flag = re.NOFLAG if ignore_case is False else re.I
    return [c for c in df.columns if re.match(pattern, c, flags=flag)]


def lead_with_columns(df, lead_columns):
    """
    Rearranges dataframe column order by pulling a few columns to the front but keeping the order for other columns
    """
    post_columns = [col for col in df.columns if col not in lead_columns]
    return df[lead_columns + post_columns]



# EMBED_OPEN_DATA_DIR=/usr/xtmp/jcd97/datasets/embed
# EMBED_OPEN_DATA_DICOM_DIR=/usr/xtmp/jcd97/datasets/embed/images
# EMBED_OPEN_DATA_PNG_DIR=/usr/xtmp/lam135/embed/png
# EMBED_OPEN_DATA_TABLE_DIR=/usr/xtmp/jcd97/datasets/embed/tables

KEY_COLUMNS_MAG = ["empi_anon", "acc_anon", "side", "numfind", "desc"]
KEY_COLUMNS_MET = [
    "empi_anon",
    "acc_anon",
    "ImageLateralityFinal",
    "ViewPosition",
    "FinalImageType",
    "AcquisitionTime",
]
CONTEXT_COLUMNS = ["cohort_num", "study_date_anon"]

# BIRADS 0: A – Additional evaluation
# BIRADS 1: N – Negative
# BIRADS 2: B - Benign
# BIRADS 3: P – Probably benign
# BIRADS 4: S – Suspicious
# BIRADS 5: M- Highly suggestive of malignancy
# BIRADS 6: K - Known biopsy proven
__birads_map = {"A": 0, "N": 1, "B": 2, "P": 3, "S": 4, "M": 5, "K": 6, "X": "X"}


def translate_birads(embed_asses):
    return __birads_map[embed_asses]


def select__goldscr(df_met, image_type="2D"):
    """
    Selects the most recent gold standard screening images from the metadata dataframe.

    Assume that the last standard images taken in an exam are the gold standard images.
    """

    df_met_goldscreen = df_met[
        (df_met["ViewPosition"].isin(["MLO", "CC"]))
        & (df_met["ImageLateralityFinal"].isin(["L", "R"]))
        & (df_met["FinalImageType"] == image_type)
        & (df_met["spot_mag"].isna())
        & (df_met["scr_or_diag"] == "S")
    ]

    drop_col = False
    if df_met.index.name == "acc_anon":
        drop_col = True
        df_met["acc_anon"] = df_met.index

    df_met_goldscreen = df_met_goldscreen.sort_values(
        ["AcquisitionTime"], ascending=False
    )
    df_met_goldscreen = df_met_goldscreen.drop_duplicates(
        ["ViewPosition", "ImageLateralityFinal"]
    )

    if drop_col:
        df_met_goldscreen = df_met_goldscreen.drop(columns=["acc_anon"])

    return df_met_goldscreen


def merge__df_clinicalXmet(df_clinical, df_met, image_type=None):
    """
    Merge the clinical and metadata, correctly handling missing records and side matching.

    Returns: A dataframe with the clinical and metadata information merged.
    """

    if image_type is not None:
        df_met = df_met.loc[df_met["FinalImageType"] == image_type]

    left_on = ["empi_anon", "acc_anon"]
    right_on = ["empi_anon", "acc_anon"]
    df_clinicalXmet_full = df_clinical.merge(
        df_met, left_on=left_on, right_on=right_on, suffixes=("_clin", "_met")
    )

    matching_side_filter = (
        # implicitly both
        (df_clinicalXmet_full["side"].isna())
        # explicitly both in clinical
        | (df_clinicalXmet_full["side"] == "B")
        # otherwise, matching sides
        | (df_clinicalXmet_full["side"] == df_clinicalXmet_full["ImageLateralityFinal"])
    )

    return df_clinicalXmet_full.loc[matching_side_filter]


CANCER_KEY = [
    "DC",
    "I",
    "B",
    "IDC",
    "ADE",
    "ANM",
    "CMT",
    "CPL",
    "DCH",
    "DCL",
    "DMP",
    "ICC",
    "LRC",
    "MFH",
    "MIC",
    "PTM",
    "CS",
    "DS",
    "IC",
    "ID",
    "II",
    "IL",
    "INC",
    "IPC",
    "MH",
    "MDC",
    "PC",
    "RM",
    "FS",
    "LPS",
    "TC",
    "LS",
    "AP",
    "ICP",
    "MDN",
    "MIM",
]


def cancer_field(df_clinical):
    """
    Determine has_cancer and years_to_cancer fields.
    """
    hasCancer = {}

    def isCancer(row):
        checklist = [
            row["path1"],
            row["path2"],
            row["path3"],
            row["path4"],
            row["path5"],
            row["path6"],
            row["path7"],
            row["path8"],
            row["path9"],
            row["path10"],
        ]

        for i in checklist:
            try:
                if not pd.isna(i) and i in CANCER_KEY:
                    hasCancer[row["empi_anon"]] = row["procdate_anon"]
                    return True
            except Exception as e:
                raise e
        return False

    def years_to_cancer(row):
        if row["empi_anon"] in hasCancer:
            ytc = hasCancer[row["empi_anon"]] - row["study_date_anon"]
        else:
            ytc = datetime.timedelta(-100)
        if ytc < datetime.timedelta(0):
            return 100
        else:
            return (ytc.days) / 365

    is_cancer = df_clinical.apply(isCancer, axis=1)
    years_to_cancer = df_clinical.apply(years_to_cancer, axis=1)
    return is_cancer, years_to_cancer


def metadata_for(df_legend, cols):
    """
    Generate a dataframe with the metadata for a particular column in the embed dataset.

    Usage: df_legend.pipe(mbd.metadata_for, ['col'])
    """
    return df_legend[df_legend["Header in export"].isin(cols)].sort_values(
        "Header in export"
    )


class EmbedOpenDataDataset:
    """
    A class to represent the EMBED Open Data dataset.
    """

    def __init__(self):

        # materialize the environment variables for this instance
        # this strikes a balance between complete dynamic defaults and hardcoding
        self.metadata_file =  "data/embed/tables/EMBED_OpenData_metadata.csv"
        self.clinical_file = "data/embed/tables/EMBED_OpenData_clinical.csv"
        self.legend_file = "data/embed/tables/AWS_Open_Data_Clinical_Legend.csv"

    def df_legend(self):
        """
        Load the legend of the EMBED Open Data as a pandas dataframe.
        """
        # this is just a hook for better future connections
        return pd.read_csv(self.legend_file)

    def df_metadata(self, scope="min", extra_cols=[], cohorts=None):

        force_string_cols = [
            "CollimatorShape",
            "0_ProcedureCodeSequence_CodeValue",
            "DerivationDescription",
            "CommentsOnRadiationDose",
            "DetectorDescription",
            "WindowCenter",
            "WindowWidth",
        ]

        categorical_cols = [
            "AcquisitionDeviceProcessingCode",
            "AcquisitionDeviceProcessingDescription",
            "DetectorConfiguration",
            "FieldOfViewShape",
            "CollimatorShape",
            "DetectorActiveShape",
            "ExposureStatus",
            "VOILUTFunction" "0_IconImageSequence_PhotometricInterpretation",
        ]

        tuple_cols = [
            "FieldOfViewDimensions",
            "DetectorActiveDimensions",
            "DetectorElementPhysicalSize",
            "DetectorElementSpacing",
            "WindowCenterWidthExplanation",
        ]

        date_cols = ["study_date_anon"]

        forced_types = {col: "string" for col in force_string_cols}
        forced_types.update({col: "category" for col in categorical_cols})
        forced_types.update({col: "string" for col in tuple_cols})

        if scope == "min":
            usecols = (
                KEY_COLUMNS_MET
                + CONTEXT_COLUMNS
                + [
                    "png_path",
                    "png_filename",
                    "num_roi",
                    "ROI_coords",
                    "match_level",
                    "spot_mag",
                ]
                + extra_cols
            )

        elif scope == "full":
            usecols = None

        else:
            return NotImplemented(f"{scope} is not a valid column scope")

        df_metadata = pd.read_csv(
            self.metadata_file,
            dtype=forced_types,
            parse_dates=date_cols,
            usecols=usecols,
        )

        # df_metadata["local_png_path"], df_metadata["local_dicom_path"] = (
        #     self.__local_paths(df_metadata["anon_dicom_path"])
        # )

        if cohorts:
            df_metadata = df_metadata.loc[df_metadata["cohort_num"].isin(cohorts)]

        return df_metadata.pipe(lead_with_columns, KEY_COLUMNS_MET + CONTEXT_COLUMNS)

    def df_clinical(self, scope="min", extra_cols=[], cohorts=None):
        # columns that have inconsistent value types - default casting doesn't work properly
        inconsistent_columns = [
            "case",
            "biopsite",
            "bcomp",
            "path7",
            "path8",
            "path9",
            "path10",
            "hgrade",
            "tnmpt",
            "tnmpn",
            "tnmm",
            "tnmdesc",
            "stage",
            "bdepth",
            "focality",
            "specinteg",
            "specembed",
            "her2",
            "fish",
            "extracap",
            "methodevl",
            "eic",
            "first_3_zip",
        ]
        type_overrides = {col: "string" for col in inconsistent_columns}

        categories = [
            "massshape",
            "massmargin",
            "massdens",
            "calcfind",
            "calcdistri",
            "otherfind",
            "implanfind",
            "side",
            "location",
            "depth",
            "distance",
            "asses",
            "recc",
            "proccode",
            "vtype",
            "tissueden",
            "MARITAL_STATUS_DESC",
        ]
        type_overrides.update({col: "category" for col in categories})

        date_cols = {"study_date_anon", "sdate_anon", "procdate_anon", "pdate_anon"}

        if scope == "min":
            usecols = (
                KEY_COLUMNS_MAG
                + CONTEXT_COLUMNS
                + [
                    "massshape",
                    "massmargin",
                    "otherfind",
                    "path_group",
                    "path_severity",
                ]
                + extra_cols
            )
            date_cols = date_cols.intersection(usecols)

        elif scope == "full":
            usecols = None

        else:
            return NotImplemented(f"{scope} is not a valid column scope")

        df_clinical = pd.read_csv(
            self.clinical_file,
            dtype=type_overrides,
            header=0,
            parse_dates=list(date_cols),
            usecols=usecols,
        )

        df_clinical["scr_or_diag"] = (
            df_clinical["desc"]
            .apply(lambda desc: "S" if "screen" in desc.lower() else "D")
            .astype("category")
        )
        df_clinical["birads"] = df_clinical["asses"].apply(translate_birads)

        df_clinical["has_cancer"], df_clinical["years_to_cancer"] = cancer_field(
            df_clinical
        )

        if cohorts:
            df_clinical = df_clinical.loc[df_clinical["cohort_num"].isin(cohorts)]

        return df_clinical.pipe(lead_with_columns, KEY_COLUMNS_MAG + CONTEXT_COLUMNS)

    def __local_paths(self, anon_dicom_path_series):
        """
        Generate local paths for the png and dicom files in the embed dataset.

        Usage: df.pipe(mbd.local_paths, png_dir, dicom_dir)
        """

        relative_path = anon_dicom_path_series.str[
            len("/mnt/NAS2/mammo/anon_dicom/") :
        ].apply(pathlib.Path)
        base_png_path = pathlib.Path(self.png_dir)
        base_dicom_path = pathlib.Path(self.dicom_dir)
        local_png_path_series = relative_path.apply(
            lambda path: base_png_path / path.with_suffix(".dcm.png")
        )
        local_dicom_path_series = relative_path.apply(
            lambda path: base_dicom_path / path
        )
        return local_png_path_series, local_dicom_path_series

    def __repr__(self):
        return f"EmbedOpenDataDataset({self.__dict__})"
    
def lat_filter(row):
    if row['side'] == 'R' and row['ImageLateralityFinal'] == 'R': return True
    elif row['side'] == 'L' and row['ImageLateralityFinal'] == 'L': return True
    elif row['side'] == 'B': return True
    elif isinstance(row['side'], float): 
        if math.isnan(row['side']):
            return True
    else: return False

    
if __name__ == "__main__":
    embed_open_data = EmbedOpenDataDataset()
    df_clinical = embed_open_data.df_clinical(scope="full")
    df_metadata = embed_open_data.df_metadata(scope="full")
    df_combined = pd.merge(df_clinical, df_metadata, how='inner', on='acc_anon')

    #save clinical and metadata to csv
    df_clinical.to_csv("data/embed/EMBED_OpenData_clinical_metadata_full_testing.csv", index=False)
    
    cols_x = sorted(df_combined.columns[df_combined.columns.str.endswith('_x')])
    cols_y = sorted(df_combined.columns[df_combined.columns.str.endswith('_y')])
    filtered_df_combined = df_combined.copy()
    for col_x, col_y in zip(cols_x, cols_y):
        print(col_x, col_y)
        filtered_df_combined = filtered_df_combined[filtered_df_combined[col_x] == filtered_df_combined[col_y]]
        col_name = col_x.split('_x')[0]
        print(col_name)
        filtered_df_combined[col_name] = filtered_df_combined[col_x]
        filtered_df_combined.drop([col_x, col_y], axis=1, inplace=True)
    
    filtered_df_combined = filtered_df_combined[filtered_df_combined.apply(lat_filter, axis=1)]
    df_combined_modified = filtered_df_combined.copy()
    df_combined_modified = df_combined_modified.reset_index(drop=True)

    # Instantiate lists for the four finding type -  mass, asymmetry, architectural distortion and calcification
    # Default value set to 0. 
    mass_list = [0]*df_combined_modified.shape[0]
    asymmetry_list = [0]*df_combined_modified.shape[0]
    arch_destortion_list = [0]*df_combined_modified.shape[0]
    calc_list = [0]*df_combined_modified.shape[0]


    # Architectural Distortion is defined as: 'massshape' ['Q', 'A']
    # Asymmetry is defined as: 'massshape' in ['T', 'B', 'S', 'F', 'V']
    # Mass is defined as: 'massshape' in ['G', 'R', 'O', 'X', 'N', 'Y', 'D', 'L']
    #       or 'massmargin' in ['D', 'U', 'M', 'I', 'S']
    #       or 'massdens' in ['+', '-', '=']
    # Calcification: defined as presence of any non-zero or non-null value in "calcdistri", "calcfind" or "calcnumber"

    #iterate through rows and assign values to the lists based on above rules
    for ind, row in df_combined_modified.iterrows():
        if (row['massshape'] in ['G', 'R', 'O', 'X', 'N', 'Y', 'D', 'L']) or (row['massmargin'] in ['D', 'U', 'M', 'I', 'S']) or (row['massdens'] in ['+', '-', '=']):
            mass_list[ind] = 1
            
        if row['massshape'] in ['T', 'B', 'S', 'F', 'V']:
            asymmetry_list[ind] = 1

        if row['massshape']in ['Q', 'A']:
            arch_destortion_list[ind] = 1
            
        if (row['calcdistri'] is not np.nan) or (row['calcfind'] is not np.nan) or (row['calcnumber'] != 0):
            calc_list[ind] = 1        

    # Append the final image findings columns to the dataframe        
    df_combined_modified['mass'] = mass_list
    df_combined_modified['asymmetry'] = asymmetry_list
    df_combined_modified['arch_distortion'] = arch_destortion_list
    df_combined_modified['calc'] = calc_list

    df_combined_modified = df_combined_modified[df_combined_modified['FinalImageType'] == '2D']
    # df_combined_modified = df_combined_modified[df_combined_modified['tissueden'] != 5]
    print("combined dataframe shape: ", df_combined_modified.shape)

    col_select_list = [
    "empi_anon",
    "acc_anon",
    "ImageLateralityFinal",
    "side",
    "ViewPosition",
    "tissueden",
    "cohort_num",
    "implanfind",
    "ETHNICITY_DESC",
    "path1", "path2", "path3", "path4", "path5", 
    "path6", "path7", "path8", "path9", "path10",
    "years_to_cancer",
    "age_at_study",
    "anon_dicom_path",
    'mass',
    'asymmetry',
    'arch_distortion',
    'calc',
    'KVP', # 20-30 has nan
    'BodyPartThickness',# round 63-99
    'Manufacturer' # 'HOLOGIC, Inc.', 'FUJIFILM Corporation', 'GE MEDICAL SYSTEMS', 'GE HEALTHCARE', 'Lorad, A Hologic Company'
    ]
    useful_df_combined2d = df_combined_modified[col_select_list]

    # image_paths = useful_df_combined2d['anon_dicom_path'].tolist()
    data_df = useful_df_combined2d

    data_df = data_df.dropna(subset=['tissueden', 'age_at_study'], axis=0)

    #save final dataframe to csv
    data_df.to_csv("data/embed/EMBED_OpenData_clinical_metadata_2D_findings_filtered_testing.csv", index=False)










    # df_clinical = df_clinical.drop(['Unnamed: 0'], axis=1)
    # df_metadata = df_metadata.drop(['Unnamed: 0'], axis=1)


