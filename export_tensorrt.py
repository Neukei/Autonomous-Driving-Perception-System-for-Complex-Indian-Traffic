import logging
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

def export_to_tensorrt(model_path, task_name):
    """
    Exports a PyTorch model to a TensorRT engine optimized for the Jetson.
    NOTE: This script MUST be run directly on the Jetson Orin Nano!
    """
    logger.info(f"Loading {task_name} model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
        
    logger.info("Exporting to TensorRT FP16 format...")
    logger.info("This will take a few minutes as TensorRT profiles the hardware layers.")
    
    # Export parameters optimized for Jetson Orin Nano (8GB)
    # half=True uses FP16 precision, drastically improving FPS on Jetson with negligible accuracy loss
    model.export(
        format="engine",
        half=True,           # Will fallback to FP16 if INT8 isn't used
        int8=True,           # CRITICAL: Crushes math to 8-bit for 45+ FPS!
        data="unified_dataset/data.yaml", # Required for INT8 to "calibrate" the math using real images
        workspace=4,         # Max workspace size in GB
        imgsz=640,           # Inference resolution
        dynamic=False,       # Fixed batch size is faster for Jetson
        simplify=True,       # Simplifies ONNX graph before TRT optimization
        nms=True             # Bakes NMS into the GPU TensorRT engine
    )
    
    logger.info(f"Successfully exported {task_name} model!")
    logger.info("The new file will end in '.engine'. Update your sensor_fusion.py to point to this new file.")

if __name__ == "__main__":
    print("="*60)
    print(" JETSON ORIN NANO TENSORRT EXPORTER")
    print(" IMPORTANT: Ensure you are running this ON the Jetson, not your PC!")
    print("="*60)
    
    # Export Object Detector (YOLO11s with GPU-baked NMS)
    export_to_tensorrt(
        model_path="runs/detect/indian_driving_yolo11s/weights/best.pt",
        task_name="Object Detection (YOLO11s + EfficientNMS_TRT)"
    )
    
    # Export Semantic Segmenter
    export_to_tensorrt(
        model_path="runs/segment/indian_segmentation_v1/weights/best.pt",
        task_name="Semantic Segmentation (YOLO11n-seg)"
    )
