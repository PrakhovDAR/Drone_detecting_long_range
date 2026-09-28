import os
import shutil
import random
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTRACT_DIR = PROJECT_ROOT / "data" / "interim" / "extracted"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def sample_hard_negatives(target_ratio=0.075):
    current_counts = {'train': 0, 'valid': 0, 'test': 0}
    for split in current_counts.keys():
        split_dir = PROCESSED_DIR / 'images' / split
        if split_dir.exists():
            current_counts[split] = len(list(split_dir.glob("*.*")))
            
    total_current = sum(current_counts.values())
    if total_current == 0:
        logging.warning("No images found in processed directory. Run the remap step first.")
        return

    # target_ratio = HN / (Total + HN) => HN = Total * ratio / (1 - ratio)
    target_hn_total = int(total_current * target_ratio / (1.0 - target_ratio))
    logging.info(f"Current objects dataset size: {total_current}. Target hard negatives: {target_hn_total}")
    
    cloud_dir = None
    if EXTRACT_DIR.exists():
        for ds_path in EXTRACT_DIR.iterdir():
            if "cloud" in ds_path.name.lower():
                cloud_dir = ds_path
                break
            
    if not cloud_dir:
        logging.error("Cloud dataset not found in extracted directory.")
        return
        
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    cloud_images = {'train': [], 'valid': [], 'test': []}
    
    for root, _, files in os.walk(cloud_dir):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                img_path = Path(root) / file
                path_parts = list(img_path.parts)
                split = 'train'
                if 'valid' in path_parts or 'val' in path_parts:
                    split = 'valid'
                elif 'test' in path_parts:
                    split = 'test'
                cloud_images[split].append(img_path)
                
    total_cloud = sum(len(v) for v in cloud_images.values())
    if total_cloud == 0:
        logging.error("No images found in cloud dataset.")
        return
        
    added_hn = 0
    for split in ['train', 'valid', 'test']:
        available = len(cloud_images[split])
        if available == 0:
            continue
        
        split_target = int(target_hn_total * (available / total_cloud))
        split_target = min(split_target, available)
        
        sampled_imgs = random.sample(cloud_images[split], split_target)
        
        dest_img_dir = PROCESSED_DIR / 'images' / split
        dest_lbl_dir = PROCESSED_DIR / 'labels' / split
        dest_img_dir.mkdir(parents=True, exist_ok=True)
        dest_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in sampled_imgs:
            new_name = f"cloud_hn_{img_path.name}"
            shutil.copy(img_path, dest_img_dir / new_name)
            
            # Generate empty label file for background
            with open(dest_lbl_dir / f"cloud_hn_{img_path.stem}.txt", 'w') as f:
                pass
                
        added_hn += split_target
        logging.info(f"Added {split_target} hard negatives to {split} split.")
        
    logging.info(f"Hard negatives sampling complete. Total added: {added_hn}")

if __name__ == "__main__":
    sample_hard_negatives()
