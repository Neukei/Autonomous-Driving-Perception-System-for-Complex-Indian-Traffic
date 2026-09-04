import os
import json
import shutil
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# IDD Class mapping to our custom unified YOLO classes
# We map IDD's specific labels to our 11 core detection classes
CLASS_MAPPING = {
    "car": 0,
    "bus": 1,
    "truck": 2,
    "autorickshaw": 3,
    "motorcycle": 4,
    "bicycle": 5,
    "person": 6,          # pedestrian
    "rider": 7,           # cyclist
    "animal": 8,          # cow/dog/stray
    "traffic sign": 9,
    "traffic light": 10,
    # Ignoring segmentation classes for the detection head for now (road, sky, building, etc.)
}

def polygon_to_yolo_bbox(polygon, img_width, img_height):
    """Converts a polygon to a normalized YOLO bounding box."""
    xs = [pt[0] for pt in polygon]
    ys = [pt[1] for pt in polygon]
    
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    
    # Calculate center, width, and height
    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin
    
    # Normalize values between 0 and 1
    center_x /= img_width
    center_y /= img_height
    width /= img_width
    height /= img_height
    
    # Clamp values to avoid out-of-bounds errors
    center_x = max(0.0, min(1.0, center_x))
    center_y = max(0.0, min(1.0, center_y))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    
    return center_x, center_y, width, height

def process_idd_dataset(idd_root, output_root):
    """Parses IDD JSONs and converts them to YOLO format."""
    idd_path = Path(idd_root)
    out_path = Path(output_root)
    
    # Create YOLO directory structure
    for split in ['train', 'val']:
        (out_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
    img_dir_base = idd_path / "leftImg8bit"
    gt_dir_base = idd_path / "gtFine"
    
    if not img_dir_base.exists() or not gt_dir_base.exists():
        logger.error(f"Could not find leftImg8bit or gtFine in {idd_path}")
        return

    total_objects_parsed = 0
    total_images_processed = 0

    for split in ['train', 'val']:
        logger.info(f"Processing split: {split}...")
        img_split_dir = img_dir_base / split
        gt_split_dir = gt_dir_base / split
        
        if not img_split_dir.exists():
            continue
            
        # Iterate through sequence folders (e.g., 0, 1, 2...)
        for seq_folder in tqdm(os.listdir(img_split_dir), desc=f"Parsing {split}"):
            seq_img_path = img_split_dir / seq_folder
            seq_gt_path = gt_split_dir / seq_folder
            
            if not seq_img_path.is_dir() or not seq_gt_path.is_dir():
                continue
                
            for img_name in os.listdir(seq_img_path):
                if not img_name.endswith('.jpg') and not img_name.endswith('.png'):
                    continue
                    
                # Match image name to JSON annotation name
                # IDD format: <prefix>_leftImg8bit.jpg -> <prefix>_gtFine_polygons.json
                base_name = img_name.replace("_leftImg8bit.jpg", "").replace("_leftImg8bit.png", "")
                json_name = f"{base_name}_gtFine_polygons.json"
                json_path = seq_gt_path / json_name
                
                if not json_path.exists():
                    continue
                    
                # Read JSON and parse objects
                with open(json_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        continue
                        
                img_w = data.get("imgWidth", 1920)
                img_h = data.get("imgHeight", 1080)
                
                yolo_labels = []
                for obj in data.get("objects", []):
                    label_name = obj.get("label")
                    if label_name in CLASS_MAPPING:
                        class_id = CLASS_MAPPING[label_name]
                        polygon = obj.get("polygon")
                        
                        if polygon and len(polygon) >= 3:
                            cx, cy, w, h = polygon_to_yolo_bbox(polygon, img_w, img_h)
                            yolo_labels.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                            total_objects_parsed += 1
                
                # If we found relevant objects, save the YOLO txt and copy/symlink the image
                if yolo_labels:
                    txt_out_path = out_path / 'labels' / split / f"{base_name}.txt"
                    img_out_path = out_path / 'images' / split / img_name
                    
                    # Write label file
                    with open(txt_out_path, 'w') as txt_f:
                        txt_f.write("\n".join(yolo_labels))
                    
                    # Use hardlink or copy for the image to save space
                    try:
                        if not img_out_path.exists():
                            os.link(seq_img_path / img_name, img_out_path)
                    except OSError:
                        shutil.copy2(seq_img_path / img_name, img_out_path)
                        
                    total_images_processed += 1

    # Write a data.yaml file for YOLO training
    yaml_content = f"""path: {out_path.absolute()}
train: images/train
val: images/val

# Classes
names:
  0: car
  1: bus
  2: truck
  3: autorickshaw
  4: motorcycle
  5: bicycle
  6: person
  7: rider
  8: animal
  9: traffic sign
  10: traffic light
"""
    with open(out_path / "data.yaml", "w") as f:
        f.write(yaml_content)

    logger.info("="*50)
    logger.info("IDD to YOLO Conversion Complete!")
    logger.info(f"Total Images parsed: {total_images_processed}")
    logger.info(f"Total Objects mapped: {total_objects_parsed}")
    logger.info(f"Dataset ready at: {out_path.absolute()}/data.yaml")
    logger.info("="*50)

if __name__ == "__main__":
    PROJECT_ROOT = Path(os.path.abspath(__file__)).parent
    IDD_ROOT = PROJECT_ROOT / "idd-20k-II" / "idd20kII"
    OUTPUT_ROOT = PROJECT_ROOT / "unified_dataset"
    
    process_idd_dataset(IDD_ROOT, OUTPUT_ROOT)
