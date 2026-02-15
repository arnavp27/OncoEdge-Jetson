"""
OncoEdge Dataset Converter
==========================
Converts Zenodo DOAOC (COCO-like JSON + CSV) to YOLOv11 Segmentation Format.

Features:
1. Handles 'Flat List' JSON polygon format.
2. Maps CSV Clinical Diagnosis to YOLO Classes (0=OCA, 1=OPMD, 2=Benign).
3. INCLUDES 'Healthy' images as Negative Samples (Empty .txt files).
4. Uses shutil.copy2 to preserve metadata.
"""

import json
import csv
import os
import shutil
import random
import yaml
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ================= CONFIGURATION =================
# Update these paths if they differ on your E: drive
RAW_DATA_ROOT = r"E:\4thYear\EMbedding_AI\Project\zenodo_DOAOC"
JSON_PATH = os.path.join(RAW_DATA_ROOT, "Annotation.json")
CSV_PATH = os.path.join(RAW_DATA_ROOT, "Imagewise_Data.csv")
# Analysis confirmed images are in Images/Images
IMG_DIR = os.path.join(RAW_DATA_ROOT, "Images", "Images") 

OUTPUT_DIR = r"E:\4thYear\EMbedding_AI\Project\datasets\oral_cancer_seg"

# Class Mapping
CLASS_MAP = {
    'OCA': 0,     # Oral Cancer
    'OPMD': 1,    # Pre-Cancer
    'Benign': 2,  # Benign Lesions
    # 'Healthy' is handled separately
}

VAL_RATIO = 0.2
RANDOM_SEED = 42

def setup_directories():
    if os.path.exists(OUTPUT_DIR):
        print(f"Note: Output dir {OUTPUT_DIR} exists. Merging/Overwriting...")
    
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

def load_csv_metadata():
    """Maps Image Name -> Category"""
    meta_map = {}
    print("Loading CSV metadata...")
    try:
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row['Image Name'].strip()
                category = row['Category'].strip()
                
                # Ensure extension exists
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_name += '.jpg'
                
                meta_map[img_name] = category
    except Exception as e:
        print(f"Error reading CSV: {e}")
    return meta_map

def main():
    random.seed(RANDOM_SEED)
    setup_directories()
    
    # 1. Load Metadata
    csv_data = load_csv_metadata()
    
    # 2. Load JSON Annotations
    print("Loading Annotations JSON...")
    with open(JSON_PATH, 'r') as f:
        coco = json.load(f)
    
    # Map image_id -> list of annotations
    img_anns = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in img_anns:
            img_anns[img_id] = []
        img_anns[img_id].append(ann)
        
    images_list = coco['images']
    print(f"Total Images in JSON: {len(images_list)}")
    
    # 3. Filter and Prepare List
    valid_samples = []
    
    print("Validating images...")
    for img_info in tqdm(images_list):
        file_name = img_info['file_name']
        img_id = img_info['id']
        
        # Check if file exists on disk
        src_path = os.path.join(IMG_DIR, file_name)
        if not os.path.exists(src_path):
            continue
            
        # Get Category from CSV
        category = csv_data.get(file_name)
        if not category:
            # Try matching without extension if failed
            stem = Path(file_name).stem
            category = csv_data.get(stem)
            
        if not category:
            continue # Skip if we don't know what it is (likely metadata mismatch)
            
        # Add to processing list
        valid_samples.append({
            'info': img_info,
            'category': category,
            'src_path': src_path,
            'anns': img_anns.get(img_id, [])
        })

    # 4. Split Data
    random.shuffle(valid_samples)
    val_count = int(len(valid_samples) * VAL_RATIO)
    val_set = valid_samples[:val_count]
    train_set = valid_samples[val_count:]
    
    print(f"Processing: {len(train_set)} Train, {len(val_set)} Val")
    
    stats = {'train': 0, 'val': 0, 'healthy': 0, 'lesions': 0}

    # 5. Process Loop
    for split, dataset in [('train', train_set), ('val', val_set)]:
        for sample in tqdm(dataset, desc=f"Writing {split}"):
            file_name = sample['info']['file_name']
            category = sample['category']
            src_path = sample['src_path']
            
            # --- LABEL GENERATION ---
            yolo_labels = []
            
            if category == 'Healthy':
                # Healthy = Empty list. 
                # We still copy the image and make an empty text file.
                stats['healthy'] += 1
            elif category in CLASS_MAP:
                # It's a Lesion
                class_id = CLASS_MAP[category]
                
                # Get Image Dimensions for Normalization
                try:
                    with Image.open(src_path) as im:
                        w, h = im.size
                except:
                    print(f"Corrupt image: {file_name}")
                    continue

                for ann in sample['anns']:
                    seg = ann['segmentation']
                    
                    # FIX: Handle flat list vs nested list
                    # JSON analysis showed it might be [x1,y1...] flat
                    if isinstance(seg[0], list):
                        seg = seg[0]
                    
                    if len(seg) < 6: continue # Need at least 3 points (x,y, x,y, x,y)
                    
                    # Normalize to 0-1
                    coords = []
                    for i in range(0, len(seg), 2):
                        # Clamp to ensure 0.0-1.0
                        x_norm = min(max(seg[i] / w, 0.0), 1.0)
                        y_norm = min(max(seg[i+1] / h, 0.0), 1.0)
                        coords.append(f"{x_norm:.6f}")
                        coords.append(f"{y_norm:.6f}")
                    
                    yolo_labels.append(f"{class_id} " + " ".join(coords))
                
                if yolo_labels:
                    stats['lesions'] += 1
            else:
                # Unknown category (shouldn't happen due to filter above)
                continue

            # --- WRITE FILES ---
            # 1. Copy Image (Standard Copy)
            dst_img_path = os.path.join(OUTPUT_DIR, 'images', split, file_name)
            shutil.copy2(src_path, dst_img_path)
            
            # 2. Write Label File (Empty if Healthy)
            label_name = Path(file_name).stem + ".txt"
            label_path = os.path.join(OUTPUT_DIR, 'labels', split, label_name)
            
            with open(label_path, 'w') as f:
                if yolo_labels:
                    f.write("\n".join(yolo_labels))
                # If healthy, file is created but empty (0 bytes) - This is correct for YOLO negative samples
                
            stats[split] += 1

    # 6. Generate YAML
    yaml_content = {
        'path': OUTPUT_DIR,
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'OCA',
            1: 'OPMD',
            2: 'Benign'
        }
    }
    
    with open(os.path.join(OUTPUT_DIR, 'data.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    print("\n" + "="*30)
    print("DONE! Dataset Statistics:")
    print(f"Training Images: {stats['train']}")
    print(f"Validation Images: {stats['val']}")
    print(f"Total Lesion Images: {stats['lesions']}")
    print(f"Total Healthy Images: {stats['healthy']}")
    print(f"Data location: {OUTPUT_DIR}")
    print("="*30)

if __name__ == "__main__":
    main()