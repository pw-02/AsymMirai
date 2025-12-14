import pandas as pd

def main():
    #read the CSV file into a DataFrame
    rdf = pd.read_csv("radiomics/radiomics_features_cc.csv")
    meta_df = pd.read_csv("data/embed/asymirai_input/EMBED_OpenData_metadata_screening_2D_complete_exams_with_demographics.csv")

    #add columns from meta_df to rdf based on matching 'dicom_path', ensure added to start of DataFrame
    columns_to_add = [
        "patient_id",
        "exam_id",
    ]

    for col in columns_to_add:
        rdf.insert(0, col, rdf["dicom_path"].map(
            meta_df.set_index("dicom_path")[col]
        ))

    #save the updated DataFrame to a new CSV file
    rdf.to_csv("radiomics/radiomics_features_cc_with_metadata.csv", index=False)

if __name__ == "__main__":
    main()