import os
import requests

# Directory to save models
os.makedirs("models", exist_ok=True)

# Model URLs (REPLACE THESE WITH YOUR ACTUAL HUGGING FACE LINKS)
models = {
    "lung_cancer_model.pth": "https://huggingface.co/Sowjanya2005/pulmoscan-models/resolve/main/lung_cancer_model.pth",
    "pneumonia_model.pth": "https://huggingface.co/Sowjanya2005/pulmoscan-models/resolve/main/pneumonia_model.pth",
    "tb_model_best.pth": "https://huggingface.co/Sowjanya2005/pulmoscan-models/resolve/main/tb_model_best.pth"
}

def download_file(url, dest_path):
    """Downloads a file only if it does not already exist at dest_path."""
    if not os.path.exists(dest_path):
        print(f"Downloading {dest_path}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"{dest_path} downloaded successfully.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {dest_path}: {e}")
            # Log an error but allow the script to continue for now
    else:
        print(f"{dest_path} already exists. Skipping download.")

def main():
    """Checks and downloads all specified model files."""
    print("\n--- Model Download Check Initiated ---")
    for filename, url in models.items():
        download_file(url, os.path.join("models", filename))
    print("--- Model Download Check Complete ---\n")

# This ensures the function runs immediately when the file is imported
if __name__ =="__main__":
    main()