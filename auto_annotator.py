import os
import shutil
import logging
import torch
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# Map COCO classes (YOLO default) to our Unified IDD classes
COCO_TO_IDD_MAP = {
    2: 0,   # car -> car
    5: 1,   # bus -> bus
    7: 2,   # truck -> truck
    # Note: COCO has no auto-rickshaw. It will likely guess car/truck. We accept this for pseudo-labels.
    3: 4,   # motorcycle -> motorcycle
    1: 5,   # bicycle -> bicycle
    0: 6,   # person -> person
    19: 8,  # cow -> animal
    16: 8,  # dog -> animal
    17: 8,  # horse -> animal
    20: 8,  # elephant -> animal
    11: 9,  # stop sign -> traffic sign
    9: 10,  # traffic light -> traffic light
}

def auto_annotate(frames_dir, output_root, conf_threshold=0.35):
    """Generates pseudo-labels for custom frames using a massive Teacher model."""
    frames_path = Path(frames_dir)
    out_path = Path(output_root)
    
    img_out_dir = out_path / 'images' / 'train'
    lbl_out_dir = out_path / 'labels' / 'train'
    
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    # Load the Teacher model (YOLO11x - Extra Large for highest accuracy)
    logger.info("Loading Teacher Model (yolo11x.pt)...")
    model = YOLO("yolo11x.pt")
    
    # Collect all frame paths
    all_frames = []
    for root, dirs, files in os.walk(frames_path):
        for f in files:
            if f.endswith('.jpg'):
                all_frames.append(os.path.join(root, f))
                
    logger.info(f"Found {len(all_frames)} unlabelled frames to auto-annotate.")
    
    total_labels_generated = 0
    total_images_saved = 0
    
    # Batch process frames for speed on the RTX 4070 Super
    batch_size = 16 
    
    for i in tqdm(range(0, len(all_frames), batch_size), desc="Auto-Annotating"):
        batch_paths = all_frames[i:i+batch_size]
        
        # Run inference
        results = model(batch_paths, conf=conf_threshold, verbose=False, imgsz=640)
        
        for frame_path, result in zip(batch_paths, results):
            frame_name = os.path.basename(frame_path)
            base_name = os.path.splitext(frame_name)[0]
            
            # Map detections
            yolo_labels = []
            for box in result.boxes:
                coco_cls = int(box.cls.cpu().numpy()[0])
                if coco_cls in COCO_TO_IDD_MAP:
                    idd_cls = COCO_TO_IDD_MAP[coco_cls]
                    
                    # Normalized bounding box [x_center, y_center, width, height]
                    xywhn = box.xywhn.cpu().numpy()[0]
                    cx, cy, w, h = xywhn
                    
                    yolo_labels.append(f"{idd_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    total_labels_generated += 1
            
            # Save if we found anything useful
            if yolo_labels:
                txt_out_path = lbl_out_dir / f"custom_{base_name}.txt"
                img_out_path = img_out_dir / f"custom_{frame_name}"
                
                with open(txt_out_path, 'w') as txt_f:
                    txt_f.write("\n".join(yolo_labels))
                
                try:
                    if not img_out_path.exists():
                        os.link(frame_path, img_out_path)
                except OSError:
                    shutil.copy2(frame_path, img_out_path)
                    
                total_images_saved += 1
                
    logger.info("="*50)
    logger.info("Auto-Annotation Complete!")
    logger.info(f"Custom Images added to training set: {total_images_saved}")
    logger.info(f"Total Pseudo-labels generated: {total_labels_generated}")
    logger.info("="*50)

if __name__ == "__main__":
    PROJECT_ROOT = Path(os.path.abspath(__file__)).parent
    FRAMES_ROOT = PROJECT_ROOT / "extracted_frames"
    OUTPUT_ROOT = PROJECT_ROOT / "unified_dataset"
    
    auto_annotate(FRAMES_ROOT, OUTPUT_ROOT)
