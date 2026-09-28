import os
import zipfile
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Dynamically resolve paths relative to this script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
EXTRACT_DIR = PROJECT_ROOT / "data" / "interim" / "extracted"

def extract_all():
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_DIR.exists():
        logging.error(f"Directory {RAW_DIR} not found. Ensure raw zip files are placed there.")
        return

    zip_files = list(RAW_DIR.glob("*.zip"))
    logging.info(f"Found {len(zip_files)} ZIP archives in {RAW_DIR}")

    for zip_path in zip_files:
        dest_dir = EXTRACT_DIR / zip_path.stem
        if dest_dir.exists():
            logging.info(f"Archive {zip_path.name} already extracted to {dest_dir.name}. Skipping.")
            continue
            
        logging.info(f"Extracting {zip_path.name} to {dest_dir.name}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Basic safety check against directory traversal attacks
                for member in zip_ref.namelist():
                    if member.startswith('/') or '..' in member:
                        logging.warning(f"Skipping unsafe path in {zip_path.name}: {member}")
                    else:
                        zip_ref.extract(member, dest_dir)
            logging.info(f"Extracted {zip_path.name} successfully.")
        except zipfile.BadZipFile:
            logging.error(f"File {zip_path.name} is not a valid ZIP archive.")
        except Exception as e:
            logging.error(f"Error extracting {zip_path.name}: {e}")

if __name__ == "__main__":
    extract_all()
