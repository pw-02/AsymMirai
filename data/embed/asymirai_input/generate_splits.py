import random
import pandas as pd
import os

set_random_seed = 42
random.seed(set_random_seed)

def print_splt_info(df, split_name):
    total_exams = len(df['exam_id'].unique())
    total_patients = len(df['patient_id'].unique())
    cancer_exams = len(df[(df['years_to_cancer'].notna()) & (df['years_to_cancer'] < 100)]['exam_id'].unique())
    cancer_patients = len(df[(df['years_to_cancer'].notna()) & (df['years_to_cancer'] < 100)]['patient_id'].unique())
    print(f"{split_name} - Total Exams: {total_exams}, Total Patients: {total_patients}, Cancer Exams: {cancer_exams}, Cancer Patients: {cancer_patients}")


all_data_path = 'C:\\Users\\pw\\projects\\AsymMirai\\data\\embed\\asymirai_input\\EMBED_OpenData_metadata_screening_2D_complete_exams_with_demographics.csv'
all_data = pd.read_csv(all_data_path)

unique_exams = all_data['exam_id'].unique().tolist()
total_exams = len(unique_exams)

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15


first_split = int(total_exams * 0.7)
second_split = first_split + int(total_exams * 0.2)
shuffle_patient = random.sample(unique_exams, len(unique_exams))
patient_train = shuffle_patient[:first_split]
patient_test = shuffle_patient[first_split:second_split]
patient_val = shuffle_patient[second_split:]

df_train = all_data[all_data["exam_id"].isin(patient_train)]
df_val = all_data[all_data["exam_id"].isin(patient_test)]
df_test = all_data[all_data["exam_id"].isin(patient_val)]
print(len(df_train), len(df_test), len(df_val))


print_splt_info(df_train, "Train")
print_splt_info(df_val, "Validation")
print_splt_info(df_test, "Test")

df_train.to_csv("data/embed/asymirai_input/EMBED_OpenData_metadata_screening_2D_complete_exams_with_demographics_train.csv", index=False)
df_val.to_csv("data/embed/asymirai_input/EMBED_OpenData_metadata_screening_2D_complete_exams_with_demographics_val.csv", index=False)
df_test.to_csv("data/embed/asymirai_input/EMBED_OpenData_metadata_screening_2D_complete_exams_with_demographics_test.csv", index=False) 

