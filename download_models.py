import os
import gdown

# === Define Google Drive folder IDs ===
T5_FOLDER_ID = "1M17FAvVJz7gntv4dRNxpxL5j79dNksTR"
DISTILBERT_FOLDER_ID = "1PyHvAo5rWkMAdI81m_VXk-mlATu-K0KZ"

def download_if_not_exists(folder_name, gdrive_id):
    if not os.path.exists(folder_name):
        print(f"Downloading {folder_name} from Google Drive...")
        gdown.download_folder(f"https://drive.google.com/drive/folders/{gdrive_id}", output=folder_name, quiet=False, use_cookies=False)
    else:
        print(f"{folder_name} already exists. Skipping download.")

if __name__ == "__main__":
    download_if_not_exists("t5_resume_model_v2", T5_FOLDER_ID)
    download_if_not_exists("distilbert_resume_classifier_v2", DISTILBERT_FOLDER_ID)
