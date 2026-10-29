"""
Ablation Study: Train YOLO11s WITHOUT synthetic weather images.
Compares against the full dataset to quantify the impact of synthetic augmentation.
"""
import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

def prepare_no_weather_dataset(unified_dir, ablation_dir):
    """Creates a copy of the dataset excluding all synth_ prefixed images and labels."""
    src = Path(unified_dir)
    dst = Path(ablation_dir)
    
    for split in ['train', 'val']:
        (dst / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dst / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        img_dir = src / 'images' / split
        lbl_dir = src / 'labels' / split
        
        copied = 0
        skipped = 0
        
        for img_file in img_dir.iterdir():
            if img_file.name.startswith('synth_'):
                skipped += 1
                continue
            
            # Copy image (use hardlink for speed)
            dst_img = dst / 'images' / split / img_file.name
            if not dst_img.exists():
                try:
                    os.link(str(img_file), str(dst_img))
                except OSError:
                    shutil.copy2(str(img_file), str(dst_img))
            
            # Copy matching label
            lbl_file = lbl_dir / (img_file.stem + '.txt')
            if lbl_file.exists():
                dst_lbl = dst / 'labels' / split / lbl_file.name
                if not dst_lbl.exists():
                    try:
                        os.link(str(lbl_file), str(dst_lbl))
                    except OSError:
                        shutil.copy2(str(lbl_file), str(dst_lbl))
            
            copied += 1
        
        logger.info(f"{split}: Copied {copied} images, skipped {skipped} synthetic weather images")
    
    # Write data.yaml
    yaml_content = f"""path: {dst.absolute()}
train: images/train
val: images/val

# Classes (identical to full dataset)
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
    with open(dst / 'data.yaml', 'w') as f:
        f.write(yaml_content)
    
    logger.info(f"Ablation dataset created at {dst}")

def train_ablation():
    PROJECT_ROOT = Path(os.path.abspath(__file__)).parent
    ABLATION_DIR = PROJECT_ROOT / "ablation_no_weather"
    UNIFIED_DIR = PROJECT_ROOT / "unified_dataset"
    
    # Step 1: Prepare filtered dataset
    logger.info("="*50)
    logger.info("ABLATION STUDY: Impact of Synthetic Weather Augmentation")
    logger.info("="*50)
    
    logger.info("Preparing dataset WITHOUT synthetic weather images...")
    prepare_no_weather_dataset(UNIFIED_DIR, ABLATION_DIR)
    
    # Step 2: Train with identical hyperparameters
    logger.info("Training YOLO11s on the NO-WEATHER dataset...")
    model = YOLO("yolo11s.pt")
    
    results = model.train(
        data=str(ABLATION_DIR / "data.yaml"),
        epochs=300,
        imgsz=640,
        batch=16,
        device=0,
        workers=2,
        name="ablation_no_weather",
        project="runs/detect",
        save=True,
        exist_ok=True,
        patience=30,
        # Same augmentations as the main run (minus the synthetic data)
        mosaic=1.0,
        mixup=0.2,
        scale=0.5,
        perspective=0.0001,
        erasing=0.4,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4
    )
    
    logger.info("="*50)
    logger.info("ABLATION TRAINING COMPLETE!")
    logger.info("Weights: runs/detect/ablation_no_weather/weights/best.pt")
    logger.info("Compare these metrics against the full-dataset run to quantify weather augmentation impact.")
    logger.info("="*50)

if __name__ == "__main__":
    train_ablation()
