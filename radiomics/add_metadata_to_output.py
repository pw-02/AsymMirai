import pandas as pd

def normalize_path(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.strip()
         .str.replace(r"//+", "/", regex=True)   # collapse double slashes
    )

def main():
    # ---------------------------
    # Load data
    # ---------------------------
    r_df = pd.read_csv("radiomics/radiomics_features_cc.csv")
    meta_df = pd.read_csv(
        "data/embed/asymirai_input/EMBED_OpenData_metadata_screening_2D_complete_exams_with_demographics.csv"
    )

    # ---------------------------
    # Validate required columns
    # ---------------------------
    required_cols = {"dicom_path", "patient_id", "exam_id"}
    missing = required_cols - set(meta_df.columns)
    if missing:
        raise ValueError(f"Missing columns in metadata: {missing}")

    if "dicom_path" not in r_df.columns:
        raise ValueError("dicom_path missing from radiomics file")

    # ---------------------------
    # Normalize dicom_path
    # ---------------------------
    # r_df["dicom_path"] = normalize_path(r_df["dicom_path"])
    # meta_df["dicom_path"] = normalize_path(meta_df["dicom_path"])

    # ---------------------------
    # Ensure metadata uniqueness
    # ---------------------------
    dupes = r_df["dicom_path"].duplicated()
    if dupes.any():
        num_dupes = dupes.sum()
        print(f"⚠️  Found {num_dupes} duplicate dicom_path values in radiomics")
        bad = r_df.loc[dupes, "dicom_path"].head(5).tolist()
        raise ValueError(
            f"Duplicate dicom_path values in radiomics (example): {bad}"
        )
    
    #remove duplicates from meta_df
    dupes = meta_df["dicom_path"].duplicated()
    if dupes.any():
        num_dupes = dupes.sum()
        print(f"⚠️  Found {num_dupes} duplicate dicom_path values in metadata")
        meta_df = meta_df.drop_duplicates(subset=["dicom_path"])
        print(f"✅ Removed duplicates from metadata")

    # ---------------------------
    # Merge metadata onto radiomics
    # ---------------------------
    merged = r_df.merge(
        meta_df[["dicom_path", "patient_id", "exam_id"]],
        on="dicom_path",
        how="left",
        validate="many_to_one",  # radiomics → metadata
    )

    # ---------------------------
    # Post-merge validation
    # ---------------------------
    missing_ids = merged["exam_id"].isna()
    if missing_ids.any():
        print("⚠️  Unmatched dicom_path examples:")
        print(
            merged.loc[missing_ids, "dicom_path"]
            .drop_duplicates()
            .head(10)
            .to_string(index=False)
        )
        print(f"\nTotal unmatched rows: {missing_ids.sum()}")
        raise ValueError("Some radiomics rows could not be matched to metadata")

    # ---------------------------
    # Save output
    # ---------------------------
    merged.to_csv(
        "radiomics/radiomics_features_cc_with_metadata.csv",
        index=False
    )

    print("✅ Metadata successfully attached to radiomics features")

if __name__ == "__main__":
    main()
