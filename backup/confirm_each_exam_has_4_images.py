import pandas as pd

def main():
    file_path = "data/embed/asymirai_input/EMBED_OpenData_metadata_screening_2D_complete_exams.csv"
    input_data = pd.read_csv(file_path)

    #check that each exam has 4 images
    exam_image_counts = input_data.groupby(["patient_id", "exam_id"]).size()
    exams_with_4_images = exam_image_counts[exam_image_counts == 4]
    print(f"Number of exams with exactly 4 images: {len(exams_with_4_images)}")
    exams_with_not_4_images = exam_image_counts[exam_image_counts != 4]
    print(f"Number of exams with not exactly 4 images: {len(exams_with_not_4_images)}")
    if len(exams_with_not_4_images) > 0:
        print("Exams with not exactly 4 images:")
        print(exams_with_not_4_images)
    #remove exams that do not have exactly 4 images
    valid_exams = exams_with_4_images.reset_index()[["patient_id", "exam_id"]]
    cleaned_data = input_data.merge(valid_exams, on=["patient_id", "exam_id"])
    cleaned_data.to_csv("data/embed/asymirai_input/EMBED_OpenData_with_demographics_ONLY_4_IMAGES.csv", index=False)






if __name__ == "__main__":
    main()