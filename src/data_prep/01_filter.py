import os
import cv2
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTRACT_DIR = PROJECT_ROOT / "data" / "interim" / "extracted"

def filter_images():
    if not EXTRACT_DIR.exists():
        logging.error(f"Directory {EXTRACT_DIR} not found. Please extract the data first.")
        return

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    total_images = 0
    corrupted_images = 0

    for root, _, files in os.walk(EXTRACT_DIR):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in image_extensions:
                total_images += 1
                img_path = Path(root) / file
                
                try:
                    # imread reads without modifying size/proportions.
                    # We just use it to validate the image structure.
                    img = cv2.imread(str(img_path))
                    if img is None:
                        raise ValueError("Image could not be read or is corrupted.")
                except Exception as e:
                    corrupted_images += 1
                    logging.warning(f"Corrupted image found and deleted: {img_path} ({e})")
                    img_path.unlink(missing_ok=True)
                    
                    # Delete corresponding label file if it exists in the same directory
                    label_path = img_path.with_suffix('.txt')
                    if label_path.exists():
                        label_path.unlink()
                        
                    # Also try to delete if labels are in a sibling directory (e.g. labels/train/)
                    label_path_sibling = Path(str(img_path).replace(f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}')).with_suffix('.txt')
                    if label_path_sibling.exists():
                        label_path_sibling.unlink()
                        
    logging.info(f"Filtering complete. Total valid images processed: {total_images}. Corrupted deleted: {corrupted_images}.")

if __name__ == "__main__":
    filter_images()
