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
    # 1. Считаем текущее количество объектов и фоновых изображений
    total_objects = 0
    existing_hns = [] 
    
    for split in ['train', 'valid', 'test']:
        lbl_dir = PROCESSED_DIR / 'labels' / split
        if not lbl_dir.exists():
            continue
            
        for txt_file in lbl_dir.glob("*.txt"):
            if txt_file.stat().st_size == 0:
                existing_hns.append(txt_file)
            else:
                with open(txt_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            total_objects += 1

    if total_objects == 0:
        logging.warning("No objects found in processed directory. Run the pipeline first.")
        return

    # 2. Считаем целевое количество
    target_hn_total = int(total_objects * target_ratio)
    current_hn_total = len(existing_hns)
    
    logging.info(f"Total objects across all classes: {total_objects}")
    logging.info(f"Target hard negatives (7.5%): {target_hn_total}")
    logging.info(f"Current hard negatives: {current_hn_total}")
    
    if current_hn_total > target_hn_total:
        to_remove = current_hn_total - target_hn_total
        logging.info(f"Removing {to_remove} excess hard negatives to reach exactly 7.5%...")
        hns_to_remove = random.sample(existing_hns, to_remove)
        for lbl_path in hns_to_remove:
            img_dir = PROCESSED_DIR / 'images' / lbl_path.parent.name
            img_candidates = list(img_dir.glob(f"{lbl_path.stem}.*"))
            for img in img_candidates:
                img.unlink(missing_ok=True)
            lbl_path.unlink(missing_ok=True)
        logging.info(f"Removed {to_remove} excess hard negatives.")
        return
        
    elif current_hn_total < target_hn_total:
        to_add = target_hn_total - current_hn_total
        logging.info(f"Adding {to_add} hard negatives from Cloud dataset...")
        
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
            
            split_target = int(to_add * (available / total_cloud))
            split_target = min(split_target, available)
            
            sampled_imgs = random.sample(cloud_images[split], split_target)
            
            dest_img_dir = PROCESSED_DIR / 'images' / split
            dest_lbl_dir = PROCESSED_DIR / 'labels' / split
            dest_img_dir.mkdir(parents=True, exist_ok=True)
            dest_lbl_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in sampled_imgs:
                new_name = f"cloud_hn_added_{img_path.name}"
                shutil.copy(img_path, dest_img_dir / new_name)
                
                with open(dest_lbl_dir / f"cloud_hn_added_{img_path.stem}.txt", 'w') as f:
                    pass
                    
            added_hn += split_target
            logging.info(f"Added {split_target} hard negatives to {split} split.")
            
        # Добавляем остаток, который мог потеряться из-за округления
        remaining = to_add - added_hn
        if remaining > 0 and len(cloud_images['train']) > 0:
            sampled_imgs = random.sample(cloud_images['train'], min(remaining, len(cloud_images['train'])))
            for i, img_path in enumerate(sampled_imgs):
                new_name = f"cloud_hn_added_extra_{i}_{img_path.name}"
                shutil.copy(img_path, PROCESSED_DIR / 'images' / 'train' / new_name)
                with open(PROCESSED_DIR / 'labels' / 'train' / f"cloud_hn_added_extra_{i}_{img_path.stem}.txt", 'w') as f:
                    pass
            logging.info(f"Added {remaining} extra hard negatives to train split due to rounding.")
            
        logging.info("Hard negatives sampling complete.")
    else:
        logging.info("Hard negatives are already at exactly 7.5%.")

if __name__ == "__main__":
    sample_hard_negatives()
