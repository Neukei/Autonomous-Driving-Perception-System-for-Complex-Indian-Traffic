import os
from pathlib import Path
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

def train_segmentation_model():
    PROJECT_ROOT = Path(os.path.abspath(__file__)).parent
    DATA_YAML = PROJECT_ROOT / "seg_dataset" / "seg_data.yaml"
    
    if not DATA_YAML.exists():
        logger.error(f"Cannot find dataset configuration at {DATA_YAML}")
        return

    logger.info("Initializing YOLO11n-seg (Nano Segmentation) model...")
    # Using the latest YOLO11 segmentation variant for pixel-perfect masks
    model = YOLO("yolo11n-seg.pt") 
    
    logger.info("Starting Semantic Segmentation training on IDD Dataset...")
    logger.info("This will utilize your RTX 4070 Super 12GB.")
    
    results = model.train(
        data=str(DATA_YAML),
        epochs=30,                  # 30 epochs for initial convergence
        imgsz=640,                  
        batch=16,                   # 16 batch size (safe for system RAM and 12GB VRAM)
        device=0,                   
        workers=2,                  # 2 workers to avoid Windows multiprocess MemoryError
        name="indian_segmentation_v1", 
        project="runs/segment",     
        save=True,                  
        exist_ok=True,              
        patience=10,                
        # Augmentations for Indian roads
        mosaic=1.0,                 
        mixup=0.1,
        hsv_h=0.015,                
        hsv_s=0.7,                  
        hsv_v=0.4                   
    )
    
    logger.info("="*50)
    logger.info("SEGMENTATION TRAINING COMPLETE!")
    logger.info("Weights saved in: runs/segment/indian_segmentation_v1/weights/best.pt")
    logger.info("="*50)

if __name__ == "__main__":
    train_segmentation_model()
