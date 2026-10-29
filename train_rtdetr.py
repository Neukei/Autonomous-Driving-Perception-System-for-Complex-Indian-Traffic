"""
Train Real-Time Detection Transformer (RT-DETR) on the Unified Indian Driving Dataset.
Benchmark: Vision Transformer vs CNN (YOLO11s).
"""
import os
import torch
from pathlib import Path
from ultralytics import RTDETR
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("RTDETR_Trainer")

def train_rtdetr():
    PROJECT_ROOT = Path(os.path.abspath(__file__)).parent
    DATA_YAML = PROJECT_ROOT / "unified_dataset" / "data.yaml"
    
    logger.info("=" * 60)
    logger.info("STARTING RT-DETR (REAL-TIME DETECTION TRANSFORMER) BENCHMARK")
    logger.info("Dataset: Unified Indian Road Dataset (21,755 images)")
    logger.info("Hardware: NVIDIA GeForce RTX 4070 Super (12GB)")
    logger.info("=" * 60)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        logger.error("CUDA is not available! RT-DETR requires GPU acceleration.")
        return

    # Load RT-DETR model
    # rtdetr-l.pt is Baidu's Real-Time DEtection TRansformer with ResNet backbone + Transformer encoder
    model_weight = "rtdetr-l.pt"
    if not Path(model_weight).exists():
        logger.warning(f"{model_weight} not found locally, waiting for download...")
        
    model = RTDETR(model_weight)
    
    # Hyperparameters tuned for RTX 4070 Super & Windows stability:
    # batch=8 to safely fit Transformer Multi-Head Self-Attention in 12GB VRAM
    # workers=2 to avoid Windows array memory issues
    results = model.train(
        data=str(DATA_YAML),
        epochs=100,               # RT-DETR converges faster than standard CNNs
        imgsz=640,
        batch=8,                  # Attention matrices are memory-intensive
        device=0,
        workers=2,
        name="indian_driving_rtdetr",
        project="runs/detect",
        save=True,
        exist_ok=True,
        patience=25,
        # Standard augmentations for fair comparison against YOLO11s
        mosaic=1.0,
        mixup=0.1,
        scale=0.5,
        perspective=0.0001,
        erasing=0.3
    )
    
    logger.info("=" * 60)
    logger.info("RT-DETR TRAINING COMPLETED!")
    logger.info("Best weights saved at: runs/detect/indian_driving_rtdetr/weights/best.pt")
    logger.info("=" * 60)

if __name__ == "__main__":
    train_rtdetr()
