import os
import yaml
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTRACT_DIR = PROJECT_ROOT / "data" / "interim" / "extracted"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def get_mapping(dataset_name, data_yaml_content):
    names = data_yaml_content.get('names', [])
    if isinstance(names, dict):
        names_list = [names[k] for k in sorted(names.keys())]
    else:
        names_list = names
        
    name_to_target = {}
    ds_name_lower = dataset_name.lower()
    
    if "cloud" in ds_name_lower:
        # Skip Cloud dataset here; it's handled in 03_hard_negatives.py
        return None
        
    if "birds" in ds_name_lower and "drone-vs-bird" not in ds_name_lower:
        # Birds -> 3
        for i, name in enumerate(names_list):
            name_to_target[i] = 3
    elif "drone-vs-bird" in ds_name_lower:
        # drone -> 0
        for i, name in enumerate(names_list):
            if "drone" in name.lower():
                name_to_target[i] = 0
            elif "bird" in name.lower():
                name_to_target[i] = 3
    elif "flying_object" in ds_name_lower:
        # drone -> 0, c-helicopter -> 2, bird -> 3, airplanes -> 1
        for i, name in enumerate(names_list):
            name_lower = name.lower()
            if "drone" in name_lower:
                name_to_target[i] = 0
            elif "c-helicopter" in name_lower:
                name_to_target[i] = 2
            elif "bird" in name_lower:
                name_to_target[i] = 3
            else:
                name_to_target[i] = 1
    elif "drone detecting" in ds_name_lower or "uavs" in ds_name_lower:
        # drones -> 0
        for i, name in enumerate(names_list):
            name_to_target[i] = 0

    return name_to_target

def process_and_copy():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'valid', 'test']:
        (PROCESSED_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (PROCESSED_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
    if not EXTRACT_DIR.exists():
        logging.error(f"Directory {EXTRACT_DIR} not found.")
        return

    total_files_copied = 0
    total_labels_remapped = 0
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    for ds_path in EXTRACT_DIR.iterdir():
        if not ds_path.is_dir():
            continue
            
        yaml_path = ds_path / "data.yaml"
        if not yaml_path.exists():
            logging.warning(f"data.yaml not found in {ds_path.name}. Skipping dataset.")
            continue
            
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_content = yaml.safe_load(f)
            
        mapping = get_mapping(ds_path.name, yaml_content)
        if mapping is None:
            logging.info(f"Skipping {ds_path.name} in remapping step (handled separately).")
            continue
            
        logging.info(f"Mapping for {ds_path.name}: {mapping}")
        
        for root, _, files in os.walk(ds_path):
            for file in files:
                ext = Path(file).suffix.lower()
                if ext in image_extensions:
                    img_path = Path(root) / file
                    
                    path_parts = list(img_path.parts)
                    split = 'train'
                    if 'valid' in path_parts or 'val' in path_parts:
                        split = 'valid'
                    elif 'test' in path_parts:
                        split = 'test'
                    
                    # Locate the corresponding label file
                    label_path_same = img_path.with_suffix('.txt')
                    label_path_sibling = Path(str(img_path).replace(f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}')).with_suffix('.txt')
                    
                    actual_label_path = None
                    if label_path_sibling.exists():
                        actual_label_path = label_path_sibling
                    elif label_path_same.exists():
                        actual_label_path = label_path_same
                        
                    new_img_name = f"{ds_path.name}_{img_path.name}"
                    dest_img_path = PROCESSED_DIR / 'images' / split / new_img_name
                    shutil.copy(img_path, dest_img_path)
                    total_files_copied += 1
                    
                    dest_label_path = PROCESSED_DIR / 'labels' / split / f"{ds_path.name}_{img_path.stem}.txt"
                    lines_to_write = []
                    
                    if actual_label_path and actual_label_path.exists():
                        with open(actual_label_path, 'r') as f:
                            lines = f.readlines()
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                orig_cls = int(parts[0])
                                if orig_cls in mapping:
                                    new_cls = mapping[orig_cls]
                                    lines_to_write.append(f"{new_cls} " + " ".join(parts[1:]) + "\n")
                                    total_labels_remapped += 1
                    
                    with open(dest_label_path, 'w') as f:
                        f.writelines(lines_to_write)
                        
    logging.info(f"Remapping complete. Images copied: {total_files_copied}. BBoxes remapped: {total_labels_remapped}.")

if __name__ == "__main__":
    process_and_copy()
