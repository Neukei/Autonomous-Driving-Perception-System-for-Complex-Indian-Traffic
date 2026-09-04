import os
import json
import shutil
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# We are mapping both background/road classes and object classes for segmentation
SEG_CLASS_MAPPING = {
    "road": 0,
    "sidewalk": 1,
    "non-drivable": 2,
    "car": 3,
    "bus": 4,
    "truck": 5,
    "autorickshaw": 6,
    "motorcycle": 7,
    "bicycle": 8,
    "person": 9,
    "rider": 10,
    "animal": 11,
}

def normalize_polygon(polygon, img_width, img_height):
    """Converts a polygon to YOLO segmentation format: x1 y1 x2 y2 ... xn yn (normalized)"""
    normalized_coords = []
    for pt in polygon:
        x = max(0.0, min(1.0, pt[0] / img_width))
        y = max(0.0, min(1.0, pt[1] / img_height))
        normalized_coords.extend([f"{x:.6f}", f"{y:.6f}"])
    return " ".join(normalized_coords)

def process_idd_segmentation(idd_root, output_root):
    """Parses IDD JSONs and converts polygons into YOLO-Seg format."""
    idd_path = Path(idd_root)
    out_path = Path(output_root)
    
    for split in ['train', 'val']:
        (out_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
    img_dir_base = idd_path / "leftImg8bit"
    gt_dir_base = idd_path / "gtFine"
    
    total_polygons = 0
    total_images = 0

    for split in ['train', 'val']:
        logger.info(f"Processing Segmentation for {split}...")
        img_split_dir = img_dir_base / split
        gt_split_dir = gt_dir_base / split
        
        if not img_split_dir.exists(): continue
            
        for seq_folder in tqdm(os.listdir(img_split_dir), desc=f"Parsing Seg {split}"):
            seq_img_path = img_split_dir / seq_folder
            seq_gt_path = gt_split_dir / seq_folder
            
            if not seq_img_path.is_dir() or not seq_gt_path.is_dir(): continue
                
            for img_name in os.listdir(seq_img_path):
                if not img_name.endswith(('.jpg', '.png')): continue
                    
                base_name = img_name.replace("_leftImg8bit.jpg", "").replace("_leftImg8bit.png", "")
                json_path = seq_gt_path / f"{base_name}_gtFine_polygons.json"
                
                if not json_path.exists(): continue
                    
                with open(json_path, 'r', encoding='utf-8') as f:
                    try: data = json.load(f)
                    except: continue
                        
                img_w, img_h = data.get("imgWidth", 1920), data.get("imgHeight", 1080)
                yolo_seg_labels = []
                
                for obj in data.get("objects", []):
                    label_name = obj.get("label")
                    if label_name in SEG_CLASS_MAPPING:
                        class_id = SEG_CLASS_MAPPING[label_name]
                        polygon = obj.get("polygon")
                        
                        # Only keep valid polygons
                        if polygon and len(polygon) >= 3:
                            norm_poly = normalize_polygon(polygon, img_w, img_h)
                            yolo_seg_labels.append(f"{class_id} {norm_poly}")
                            total_polygons += 1
                
                if yolo_seg_labels:
                    txt_out = out_path / 'labels' / split / f"{base_name}.txt"
                    img_out = out_path / 'images' / split / img_name
                    
                    with open(txt_out, 'w') as txt_f:
                        txt_f.write("\n".join(yolo_seg_labels))
                    
                    try:
                        if not img_out.exists(): os.link(seq_img_path / img_name, img_out)
                    except OSError:
                        shutil.copy2(seq_img_path / img_name, img_out)
                    total_images += 1

    yaml_content = f"""path: {out_path.absolute()}
train: images/train
val: images/val
names:
  0: road
  1: sidewalk
  2: non-drivable
  3: car
  4: bus
  5: truck
  6: autorickshaw
  7: motorcycle
  8: bicycle
  9: person
  10: rider
  11: animal
"""
    with open(out_path / "seg_data.yaml", "w") as f:
        f.write(yaml_content)

    logger.info(f"Segmentation Parsing Complete! {total_polygons} polygons across {total_images} images.")

if __name__ == "__main__":
    PROJECT_ROOT = Path(os.path.abspath(__file__)).parent
    process_idd_segmentation(PROJECT_ROOT / "idd-20k-II" / "idd20kII", PROJECT_ROOT / "seg_dataset")
