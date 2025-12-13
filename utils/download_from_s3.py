import boto3
from pathlib import Path

BUCKET_NAME = "embdedpng"
FILES_LIST = "utils/files_to_download.txt"
OUTPUT_DIR = "downloaded_files"
#s3://embdedpng/png_images/../png_tmp/cohort_1/
s3 = boto3.client("s3")

def download_files():
    with open(FILES_LIST, "r") as f:
        keys = [line.strip() for line in f if line.strip()]

    print(f"Found {len(keys)} files to download")

    for key in keys:
        #change key extenstion to png instead of dcm
        key = key.replace(".dcm",".png")
        local_path = Path(OUTPUT_DIR) / key
        s3key = key.replace("images/","png_images/../png_tmp/")


        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            s3.download_file(
                Bucket=BUCKET_NAME,
                Key=s3key,
                Filename=str(local_path)
            )
            print(f"✓ {key}")
        except Exception as e:
            print(f"✗ Failed: {key} → {e}")

if __name__ == "__main__":
    download_files()
