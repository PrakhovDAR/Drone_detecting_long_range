import os
import cv2
import yaml
import random
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
YAML_OUTPUT = PROJECT_ROOT / "data" / "data.yaml"

TARGET_RATIO = {0: 0.4, 1: 0.2, 2: 0.2, 3: 0.2}

def count_classes(labels_dir):
    class_counts = defaultdict(int)
    image_classes = defaultdict(set)
    
    for txt_file in labels_dir.glob("*.txt"):
        if txt_file.stat().st_size == 0:
            continue
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    class_counts[cls_id] += 1
                    image_classes[txt_file.stem].add(cls_id)
    return class_counts, image_classes

def augment_image_flip(img_path, lbl_path, new_img_path, new_lbl_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return False
        
    flipped_img = cv2.flip(img, 1) # horizontal
    cv2.imwrite(str(new_img_path), flipped_img)
    
    with open(lbl_path, 'r') as f:
        lines = f.readlines()
        
    with open(new_lbl_path, 'w') as f:
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = parts[0]
                x_center = float(parts[1])
                y_center = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                
                new_x_center = 1.0 - x_center
                f.write(f"{cls_id} {new_x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
    return True

def balance_dataset():
    for split in ['train', 'valid', 'test']:
        split_img_dir = PROCESSED_DIR / 'images' / split
        split_lbl_dir = PROCESSED_DIR / 'labels' / split
        
        if not split_img_dir.exists():
            continue
            
        logging.info(f"Balancing split: {split}")
        class_counts, image_classes = count_classes(split_lbl_dir)
        logging.info(f"Current class distribution: {dict(class_counts)}")
        
        if not class_counts:
            continue
            
        total_objects = sum(class_counts.values())
        target_counts = {c: int(total_objects * ratio) for c, ratio in TARGET_RATIO.items()}
        logging.info(f"Target distribution: {target_counts}")
        
        for cls, target in target_counts.items():
            current = class_counts.get(cls, 0)
            
            if current > target:
                to_remove = current - target
                candidates = [img_stem for img_stem, classes in image_classes.items() if list(classes) == [cls]]
                random.shuffle(candidates)
                
                removed = 0
                for stem in candidates:
                    if removed >= to_remove:
                        break
                    
                    lbl_file = split_lbl_dir / f"{stem}.txt"
                    objs_in_image = 0
                    if lbl_file.exists():
                        with open(lbl_file, 'r') as f:
                            objs_in_image = len([line for line in f if line.strip()])
                        
                    img_file_candidates = list(split_img_dir.glob(f"{stem}.*"))
                    if img_file_candidates:
                        img_file_candidates[0].unlink()
                    lbl_file.unlink(missing_ok=True)
                    
                    removed += objs_in_image
                logging.info(f"Class {cls}: Undersampled {removed} objects.")
                
            elif current < target and current > 0:
                to_add = target - current
                candidates = [img_stem for img_stem, classes in image_classes.items() if cls in classes]
                
                added = 0
                aug_idx = 0
                
                # Prevent infinite loops if candidates run out
                max_attempts = to_add * 2 
                attempts = 0
                
                while added < to_add and candidates and attempts < max_attempts:
                    attempts += 1
                    stem = random.choice(candidates)
                    
                    lbl_file = split_lbl_dir / f"{stem}.txt"
                    img_file_candidates = list(split_img_dir.glob(f"{stem}.*"))
                    
                    if not img_file_candidates or not lbl_file.exists():
                        candidates.remove(stem)
                        continue
                        
                    img_file = img_file_candidates[0]
                    new_stem = f"{stem}_aug_{aug_idx}"
                    new_img_file = split_img_dir / f"{new_stem}{img_file.suffix}"
                    new_lbl_file = split_lbl_dir / f"{new_stem}.txt"
                    
                    if augment_image_flip(img_file, lbl_file, new_img_file, new_lbl_file):
                        objs_in_image = 0
                        with open(new_lbl_file, 'r') as f:
                            for line in f:
                                if line.strip() and int(line.split()[0]) == cls:
                                    objs_in_image += 1
                        added += objs_in_image
                        aug_idx += 1
                    else:
                        candidates.remove(stem)
                        
                logging.info(f"Class {cls}: Oversampled {added} objects using OpenCV flips.")

def generate_yaml():
    YAML_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    data = {
        'path': str(PROCESSED_DIR.resolve()),
        'train': 'images/train',
        'val': 'images/valid',
        'test': 'images/test',
        'nc': 4,
        'names': {
            0: 'DRON',
            1: 'AIRPLANE',
            2: 'HELICOPTER',
            3: 'BIRDS'
        }
    }
    
    with open(YAML_OUTPUT, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    logging.info(f"Generated YAML configuration at {YAML_OUTPUT}")

if __name__ == "__main__":
    balance_dataset()
    generate_yaml()
