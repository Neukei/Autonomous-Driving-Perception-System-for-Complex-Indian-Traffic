import os
from pathlib import Path
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

def train_autonomous_model():
    PROJECT_ROOT = Path(os.path.abspath(__file__)).parent
    DATA_YAML = PROJECT_ROOT / "unified_dataset" / "data.yaml"
    
    if not DATA_YAML.exists():
        logger.error(f"Cannot find dataset configuration at {DATA_YAML}")
        return

    logger.info("Initializing YOLO11s (Small) model for maximum accuracy on occluded traffic...")
    # Upgraded to YOLO11s as it handles dense Indian traffic overlaps significantly better than Nano
    model = YOLO("yolo11s.pt") 
    
    logger.info("Starting Final Object Detection training on Unified Dataset...")
    logger.info("This will utilize your RTX 4070 Super 12GB.")
    
    # Training configuration optimized for RTX 4070 Super (12GB VRAM)
    results = model.train(
        data=str(DATA_YAML),
        epochs=300,                 # 300 epochs gives it time to master the complex augmentations
        imgsz=640,                  # Standard resolution for good accuracy/speed tradeoff
        batch=16,                   # 16 batch size (safe for system RAM/VRAM)
        device=0,                   # Use the primary GPU (RTX 4070 S)
        workers=2,                  # Reduced to 2 to fix Windows multiprocess MemoryError
        name="indian_driving_yolo11s", # Name of the run folder
        project="runs/detect",      # Where to save weights
        save=True,                  # Save best weights
        exist_ok=True,              # Overwrite if we restart
        patience=30,                # Early stopping: if accuracy doesn't improve for 30 epochs, stop automatically
        # ADVANCED AUGMENTATIONS (Varying Frames)
        # These trick the AI into thinking it's seeing millions of different scenarios
        mosaic=1.0,                 # Combines 4 images into 1 (forces it to learn smaller objects)
        mixup=0.2,                  # Overlays translucent images on top of each other
        scale=0.5,                  # Zooms in and out randomly by 50%
        perspective=0.0001,         # Adds 3D distortion (simulates different camera angles)
        erasing=0.4,                # Randomly blocks out parts of the image (forces it to guess hidden cars)
        hsv_h=0.015,                # Color shifting (simulates different times of day)
        hsv_s=0.7,                  
        hsv_v=0.4                   # Brightness shifting (simulates glare and shadows)
    )
    
    logger.info("="*50)
    logger.info("TRAINING COMPLETE!")
    logger.info("The best model weights are saved in: runs/detect/indian_driving_v1/weights/best.pt")
    logger.info("These weights are ready to be exported to TensorRT for the Jetson Orin Nano.")
    logger.info("="*50)

if __name__ == "__main__":
    train_autonomous_model()
