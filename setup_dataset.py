import os
import shutil
import kagglehub

def setup_dataset():
    # Dataset handle
    dataset_handle = "snehaanbhawal/resume-dataset"
    
    # Target directory
    target_dir = "archive"
    
    print(f"Starting download of {dataset_handle} via kagglehub...")
    try:
        download_path = kagglehub.dataset_download(dataset_handle)
        print(f"Downloaded to cache: {download_path}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nNote: You might need to set up Kaggle credentials.")
        print("See: https://github.com/Kaggle/kagglehub#authentication")
        return

    print("Organizing dataset files...")
    
    # Ensure archive directory exists
    os.makedirs(target_dir, exist_ok=True)
    
    # 1. Handle Resume.csv
    resume_target_dir = os.path.join(target_dir, "Resume")
    os.makedirs(resume_target_dir, exist_ok=True)
    
    csv_source = os.path.join(download_path, "Resume.csv")
    csv_target = os.path.join(resume_target_dir, "Resume.csv")
    
    if os.path.exists(csv_source):
        shutil.copy2(csv_source, csv_target)
        print(f"Copied {csv_source} -> {csv_target}")
    
    # 2. Handle data/ folder
    data_source = os.path.join(download_path, "data")
    data_target = os.path.join(target_dir, "data")
    
    if os.path.exists(data_source):
        if os.path.exists(data_target):
            print(f"Removing existing {data_target}...")
            shutil.rmtree(data_target)
        
        # Copy the whole directory
        shutil.copytree(data_source, data_target)
        print(f"Copied {data_source} -> {data_target}")

    print("\nDataset setup complete!")
    print(f"CSV location: {os.path.abspath(csv_target)}")
    print(f"PDF root: {os.path.abspath(data_target)}")

if __name__ == "__main__":
    setup_dataset()
