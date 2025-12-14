
import pandas as pd
import boto3

s3 = boto3.client("s3")
files_exists = 0
files_not_exists = 0


def list_all_s3_keys(bucket, prefix=None):
    keys = set()
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.add(obj["Key"])

    return keys


# def check_file_exists_in_s3(path):
#     global files_exists
#     global files_not_exists
#     s3_key = path.replace('images', 'png_images/../png_tmp').replace('.dcm', '.png')
#      #ensure forward slashes
#     s3_key = s3_key.replace('\\', '/')
#     try:
#         s3.head_object(Bucket="embdedpng", Key=s3_key)
#         files_exists += 1
#         if (files_exists + files_not_exists) % 100 == 0:
#             print(f"Total: {files_exists + files_not_exists}, Total exists: {files_exists}, Total not exists: {files_not_exists}")
#         return True
#     except s3.exceptions.ClientError as e:
#         if e.response['Error']['Code'] == "404":
#             files_not_exists += 1
#             if (files_exists + files_not_exists) % 100 == 0:
#                 print(f"Total: {files_exists + files_not_exists}, Total exists: {files_exists}, Total not exists: {files_not_exists}")
#             return False
#         else:
#             raise


def check_file_not_exists_in_s3(all_s3_keys, path):
    s3_key = path.replace('images', 'png_images/../png_tmp').replace('.dcm', '.png')
     #ensure forward slashes
    s3_key = s3_key.replace('\\', '/')
    return s3_key not in all_s3_keys



def main():

    save_path = "data/embed/asymirai_input/all_s3_keys.txt"

    if save_path:
        with open(save_path, "r") as f:
            all_s3_keys = set(line.strip() for line in f)
    else:
        all_s3_keys = list_all_s3_keys("embdedpng", prefix="png_images/")
        #save all_s3_keys to a text file
        with open("data/embed/asymirai_input/all_s3_keys.txt", "w") as f:
            for key in all_s3_keys:
                f.write(f"{key}\n")
    print(f"Total files in S3 bucket embdedpng: {len(all_s3_keys)}")

    #read in csv file
    clinical_data = pd.read_csv("data/embed/tables/EMBED_OpenData_clinical.csv")
    #for each patitent and exam id (index) find GENDER_DESC, RACE_DESC, MARITAL_STATUS_DESC, age_at_study, tissueden
    demographic_data = clinical_data[["empi_anon", "acc_anon", "GENDER_DESC", "ETHNICITY_DESC", "MARITAL_STATUS_DESC", "age_at_study", "tissueden"]]
    #rename empi_anon to patient_id and acc_anon to exam_id
    demographic_data.rename(columns={"empi_anon": "patient_id", "acc_anon": "exam_id"}, inplace=True)
    demographic_data.set_index(["patient_id", "exam_id"], inplace=True)
    demographic_data = demographic_data.sort_index()
    demographic_data.to_csv("data/embed/example/demographic_data.csv")


    #read in another csv file
    input_data = pd.read_csv("data/embed/asymirai_input/EMBED_OpenData_metadata_screening_2D_complete_exams.csv")

    #for each patient_id and exam_id in input_data, find corresponding demographic data in demographic_data and add it to input_data
    input_data = input_data.sort_values(by=["patient_id", "exam_id"])

    





    #look up demographic data for each patient and exam id in demographic_data and add it to input_data
    demographic_data = demographic_data.reset_index()
    input_data = input_data.merge(demographic_data, on=["patient_id", "exam_id"], how="left")
    # input_data.to_csv("data/embed/asymirai_input/EMBED_OpenData_metadata_screening_2D_complete_exams_CLEANED_4VIEW_with_demographics.csv", index=False)
    #for each path in the dicom_paths column of input_data, check if the file exists and update a status column 
    input_data["file_exists"] = input_data["dicom_path"].apply(lambda x: not check_file_not_exists_in_s3(all_s3_keys, x))

    input_data.to_csv("data/embed/asymirai_input/EMBED_OpenData_metadata_screening_2D_complete_exams_with_demographics.csv", index=False)
    




if __name__ == "__main__":
    main()


