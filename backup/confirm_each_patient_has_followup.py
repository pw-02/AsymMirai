import pandas as pd

def main():
    path = "data/embed/asymirai_input/EMBED_OpenData_metadata_screening_2D_complete_exams.csv"
    data = pd.read_csv(path)
    # Check the number of exames for each patient (i.e., number of unique exam_ids per patient_id)
    followup_counts = data.groupby("patient_id")["exam_id"].nunique()

    #find pateints with only 1 exam
    patients_with_single_exam = followup_counts[followup_counts == 1]
    print(f"Number of patients with only 1 exam: {len(patients_with_single_exam)}")

    #remove patients with only 1 exam
    valid_patients = followup_counts[followup_counts > 1].index
    cleaned_data = data[data["patient_id"].isin(valid_patients)]
    cleaned_data.to_csv("data/embed/asymirai_input/EMBED_OpenData_with_demographics_ONLY_PATIENTS_WITH_FOLLOWUP.csv", index=False)


    print(followup_counts)

if __name__ == "__main__":
    main()