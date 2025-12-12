import pandas as pd
import boto3
import os
s3 = boto3.client("s3")


def list_all_s3_keys(bucket, prefix=None):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"]


def s3_key_to_dicom_path(key: str) -> str:
    """
    Convert S3 PNG key to corresponding DICOM path.
    """
    return (
        key
        .replace("png_images/../png_tmp", "images")
        .replace(".png", ".dcm")
        .replace("\\", "/")
    )

def delete_s3_key(bucket: str, key: str):
    s3.delete_object(Bucket=bucket, Key=key)
    print(f"Deleted S3 key: {key} from bucket: {bucket}")


def main():
    save_path = "data/embed/asymirai_input/all_s3_keys.txt"
    bucket = "embdedpng"
    prefix = "png_images/"

    # -------------------------
    # Load or build S3 key set
    # -------------------------
    if not os.path.exists(save_path):
        with open(save_path, "r") as f:
            all_s3_keys = {line.strip() for line in f}
    else:
        all_s3_keys = set(list_all_s3_keys(bucket, prefix))
        with open(save_path, "w") as f:
            f.write("\n".join(all_s3_keys))

    print(f"Total S3 keys: {len(all_s3_keys)}")

    # -------------------------
    # Load dataset paths ONCE
    # -------------------------
    input_data = pd.read_csv(
        "data/embed/asymirai_input/EMBED_OpenData_metadata_screening_2D_complete_exams.csv",
        usecols=["dicom_path"],
    )

    dicom_paths = set(input_data["dicom_path"].astype(str))
    print(f"Total dataset DICOM paths: {len(dicom_paths)}")

    # -------------------------
    # Convert S3 keys → dicom paths
    # -------------------------
    s3_dicom_paths = {
        s3_key_to_dicom_path(key)
        for key in all_s3_keys
    }

    # -------------------------
    # Set difference (FAST)
    # -------------------------
    overloaded_files = sorted(s3_dicom_paths - dicom_paths)


    print(f"Total overloaded files: {len(overloaded_files)}")

    # -------------------------
    # Save output but save as the avtual s3 keys
    # -------------------------

    out_path = "data/embed/asymirai_input/overloaded_files.txt"
    with open(out_path, "w") as f:
        for dicom_path in overloaded_files:
            s3_key = dicom_path.replace("images/", "png_images/../png_tmp/").replace(".dcm", ".png")
            f.write(f"{s3_key}\n")

    print(f"Saved: {out_path}")

    for dicom_path in overloaded_files:
        s3_key = dicom_path.replace("images/", "png_images/../png_tmp/").replace(".dcm", ".png")
        delete_s3_key(bucket, s3_key)
        

if __name__ == "__main__":
    main()
