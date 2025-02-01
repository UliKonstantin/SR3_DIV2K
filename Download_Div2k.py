import os
import urllib.request
import zipfile

def download_and_extract_div2k():
    # Create a DIV2K directory if it doesn't exist
    div2k_dir = "DIV2K"
    os.makedirs(div2k_dir, exist_ok=True)

    # Download the training set
    train_zip_url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
    train_zip_path = os.path.join(div2k_dir, "DIV2K_train_HR.zip")

    # Extract the training set
    print(f"Unzipping {train_zip_path} ...")
    with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
        zip_ref.extractall(div2k_dir)
    print("Extraction complete.")

    # Download the validation set
    valid_zip_url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
    valid_zip_path = os.path.join(div2k_dir, "DIV2K_valid_HR.zip")

    # Extract the validation set
    print(f"Unzipping {valid_zip_path} ...")
    with zipfile.ZipFile(valid_zip_path, 'r') as zip_ref:
        zip_ref.extractall(div2k_dir)
    print("Extraction complete.")

if __name__ == "__main__":
    download_and_extract_div2k()
