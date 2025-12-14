import pandas as pd

def age_group(age: float) -> str:
    if pd.isna(age):
        return "Unknown"
    elif age < 40:
        return "<40"
    elif age < 50:
        return "40–49"
    elif age < 60:
        return "50–59"
    elif age < 70:
        return "60–69"
    elif age < 80:
        return "70–79"
    else:
        return "≥80"
    
def normalize_race(race: str) -> str:
    if pd.isna(race):
        return "Other / Unknown"
    race = race.lower()
    if "white" in race:
        return "White"
    if "black" in race or "african" in race:
        return "African American"
    if "asian" in race:
        return "Asian"
    if "native" in race or "alaska" in race:
        return "American Indian or Alaska Native"
    if "hawaiian" in race or "pacific" in race:
        return "Native Hawaiian or Pacific Islander"
    return "Other / Unknown"



def is_cancer_positive(severity) -> bool:
    return severity in (0, 1)

def extract_stats(df: pd.DataFrame, patient_id_column_name, exam_id_column_name) -> pd.DataFrame:
    df = df.copy()

    # ----------------------------------
    # Basic counts
    # ----------------------------------
    n_patients = df[patient_id_column_name].nunique()
    n_exams = df[exam_id_column_name].nunique()

    # ----------------------------------
    # Age stats (per exam)
    # ----------------------------------
    age_mean = df["age_at_study"].mean()
    age_std = df["age_at_study"].std()

    # ----------------------------------
    # Age groups (per exam)
    # ----------------------------------
    df["age_group"] = df["age_at_study"].apply(age_group)
    age_group_counts = (
        df.groupby("age_group")[exam_id_column_name]
        .nunique()
        .reindex(
            ["<40", "40–49", "50–59", "60–69", "70–79", "≥80", "Unknown"],
            fill_value=0
        )
    )

    # ----------------------------------
    # Race (per patient)
    # ----------------------------------
    race_df = (
        df[[patient_id_column_name, "ETHNICITY_DESC"]]
        .drop_duplicates(patient_id_column_name)
        .copy()
    )
    race_df["race"] = race_df["ETHNICITY_DESC"].apply(normalize_race)

    race_counts = race_df["race"].value_counts()

    # ----------------------------------
    # Cancer-positive patients
    # ----------------------------------
    df["cancer_positive"] = df["path_severity"].apply(is_cancer_positive)

    cancer_patients = (
        df[df["cancer_positive"]]
        .groupby(patient_id_column_name)
        .size()
    )
    n_cancer_patients = cancer_patients.index.nunique()
    
    # ----------------------------------
    # Cancer-positive patients by age group
    # ----------------------------------
    cancer_age_df = df[df["cancer_positive"]].copy()
    cancer_age_df["age_group"] = cancer_age_df["age_at_study"].apply(age_group)

    cancer_age_counts = (
        cancer_age_df
        .groupby("age_group")[patient_id_column_name]
        .nunique()
        .reindex(
            ["<40", "40–49", "50–59", "60–69", "70–79", "≥80", "Unknown"],
            fill_value=0
        )
    )

    # ----------------------------------
    # Cancer-positive patients by race
    # ----------------------------------
    cancer_race_df = (
        df[df["cancer_positive"]][[patient_id_column_name, "ETHNICITY_DESC"]]
        .drop_duplicates(patient_id_column_name)
        .copy()
    )

    cancer_race_df["race"] = cancer_race_df["ETHNICITY_DESC"].apply(normalize_race)

    cancer_race_counts = cancer_race_df["race"].value_counts()


    # ----------------------------------
    # Assemble output table
    # ----------------------------------
    rows = []

    rows.append((
        "No. of patients",
        f"{n_patients} ({n_cancer_patients})"
    ))

    rows.append((
        "No. of examinations",
        n_exams
    ))

    rows.append((
        "Age at examination (mean ± SD)",
        f"{age_mean:.1f} ± {age_std:.1f}"
    ))

    for group in age_group_counts.index:
        total = age_group_counts[group]
        cancer = cancer_age_counts[group]
        rows.append((
            f"Age group: {group}",
            f"{total} ({cancer})"
        ))

    for race in race_counts.index:
        total = race_counts[race]
        cancer = cancer_race_counts.get(race, 0)
        rows.append((
            f"Race: {race}",
            f"{total} ({cancer})"
        ))

    stats_df = pd.DataFrame(rows, columns=["Metric", "Value"])

    return stats_df



if __name__ == "__main__":
    mega_df = pd.read_csv("data/embed/tables/EMBED_OpenData_clinical.csv")
    mega_stats_df = extract_stats(mega_df, "empi_anon", "acc_anon")
    print(mega_stats_df)

    filtered_stats_df = pd.read_csv("data/embed/asymirai_input/EMBED_OpenData_metadata_screening_2D_complete_exams.csv")
    filtered_stats_df = extract_stats(filtered_stats_df, "patient_id", "exam_id")
    print("Filtered Stats:")
    print(filtered_stats_df)

    #find differences across each metric between mega and filtered
    for metric in mega_stats_df['Metric']:
        mega_value = mega_stats_df[mega_stats_df['Metric'] == metric]['Value'].values[0]
        filtered_value = filtered_stats_df[filtered_stats_df['Metric'] == metric]['Value'].values[0]
        if mega_value != filtered_value:
            print(f"Difference in {metric}: Mega: {mega_value}, Filtered: {filtered_value}")
            





    train_stats_df = pd.read_csv("data/embed/asymirai_input/EMBED_train.csv")
    train_stats_df = extract_stats(train_stats_df, "patient_id", "exam_id")
    print("Train Stats:")
    print(train_stats_df)

    val_stats_df = pd.read_csv("data/embed/asymirai_input/EMBED_val.csv")
    val_stats_df = extract_stats(val_stats_df, "patient_id", "exam_id")
    print("Validation Stats:")
    print(val_stats_df)

    test_stats_df = pd.read_csv("data/embed/asymirai_input/EMBED_test.csv")
    test_stats_df = extract_stats(test_stats_df, "patient_id", "exam_id")
    print("Test Stats:")
    print(test_stats_df)


    # filtered_df = pd.read_csv("data/embed/tables/EMBED_OpenData_clinical_filtered_dense_breasts_one_exam_per_patient.csv")

   
