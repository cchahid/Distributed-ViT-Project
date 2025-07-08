import os
import zipfile
import requests
from tqdm import tqdm
import shutil

# --- Configuration ---
DATA_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
DATA_DIR = './'
ZIP_FILE_PATH = os.path.join(DATA_DIR, 'tiny-imagenet-200.zip')
DATASET_DIR = os.path.join(DATA_DIR, 'tiny-imagenet-200')

def download_file(url, filename):
    """Downloads a file from a URL with a progress bar."""
    print(f"Downloading {filename} from {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(filename, 'wb') as f, tqdm(
            total=total_size, unit='iB', unit_scale=True, unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
    print("Download complete.")

def unzip_file(zip_path, extract_to):
    """Unzips a file."""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")

def format_validation_set(val_dir, val_annotations_file):
    """Formats the validation set into an ImageFolder-compatible structure."""
    print("Formatting validation set...")
    with open(val_annotations_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            img_name, class_id = parts[0], parts[1]
            
            class_dir = os.path.join(val_dir, class_id)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            
            src_path = os.path.join(val_dir, 'images', img_name)
            dst_path = os.path.join(class_dir, img_name)
            if os.path.exists(src_path):
                shutil.move(src_path, dst_path)
    
    shutil.rmtree(os.path.join(val_dir, 'images'))
    print("Validation set formatted.")

if not os.path.exists(DATASET_DIR):
    download_file(DATA_URL, ZIP_FILE_PATH)
    unzip_file(ZIP_FILE_PATH, DATA_DIR)
    val_dir = os.path.join(DATASET_DIR, 'val')
    val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')
    if os.path.exists(os.path.join(val_dir, 'images')):
        format_validation_set(val_dir, val_annotations_file)
else:
    print("Dataset already seems to be prepared.")

print("\n✅ Dataset is ready!")
