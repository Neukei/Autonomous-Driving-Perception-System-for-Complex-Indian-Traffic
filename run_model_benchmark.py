"""
Model Benchmark Script — Tests top detection models on extracted Indian road frames.
Compares YOLOv8n, YOLOv10n, YOLOv11n on actual dashcam footage.
Measures: inference time, detection count, classes found, GPU memory.
Outputs: visual comparison images + results table.
"""
import os
import sys
import time
import json
import random
import logging
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(PROJECT_ROOT, "extracted_frames")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "benchmark_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Number of frames to benchmark on
NUM_SAMPLE_FRAMES = 50
# Number of warmup frames (not counted in timing)
NUM_WARMUP = 5


def collect_sample_frames(n: int = NUM_SAMPLE_FRAMES) -> list:
    """Collect a diverse random sample of frames from extracted dataset."""
    all_frames = []
    for root, dirs, files in os.walk(FRAMES_DIR):
        for f in files:
            if f.endswith(".jpg"):
                all_frames.append(os.path.join(root, f))
    
    logger.info(f"Found {len(all_frames)} total frames in {FRAMES_DIR}")
    
    # Sample evenly from both routes
    random.seed(42)  # Reproducible
    sample = random.sample(all_frames, min(n + NUM_WARMUP, len(all_frames)))
    return sample


def benchmark_yolo_model(model_name: str, frames: list, device: str = "cuda") -> dict:
    """Benchmark a YOLO model from ultralytics.
    
    Args:
        model_name: Model identifier (e.g., 'yolov8n.pt', 'yolov10n.pt', 'yolo11n.pt')
        frames: List of frame paths
        device: 'cuda' or 'cpu'
    
    Returns:
        Dict with benchmark results
    """
    from ultralytics import YOLO
    import torch
    
    logger.info(f"Loading {model_name}...")
    model = YOLO(model_name)
    model.to(device)
    
    # Get model info
    param_count = sum(p.numel() for p in model.model.parameters())
    model_size_mb = sum(p.numel() * p.element_size() for p in model.model.parameters()) / 1e6
    
    # Warmup
    logger.info(f"  Warming up ({NUM_WARMUP} frames)...")
    for frame_path in frames[:NUM_WARMUP]:
        _ = model(frame_path, verbose=False)
    
    if device == "cuda":
        import torch
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    # Benchmark
    logger.info(f"  Benchmarking on {len(frames) - NUM_WARMUP} frames...")
    timings = []
    all_detections = []
    class_counts = {}
    
    test_frames = frames[NUM_WARMUP:]
    
    for i, frame_path in enumerate(test_frames):
        if device == "cuda":
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        results = model(frame_path, verbose=False, imgsz=640)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        elapsed = (time.perf_counter() - start) * 1000  # ms
        timings.append(elapsed)
        
        # Count detections
        for r in results:
            boxes = r.boxes
            num_dets = len(boxes)
            all_detections.append(num_dets)
            
            for cls_id in boxes.cls.cpu().numpy():
                cls_name = model.names[int(cls_id)]
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
    
    # GPU memory
    gpu_mem_mb = 0
    if device == "cuda":
        gpu_mem_mb = torch.cuda.max_memory_allocated() / 1e6
    
    # Compute stats
    timings_arr = np.array(timings)
    results_dict = {
        "model": model_name,
        "params_millions": round(param_count / 1e6, 2),
        "model_size_mb": round(model_size_mb, 1),
        "device": device,
        "num_frames": len(test_frames),
        "mean_latency_ms": round(float(np.mean(timings_arr)), 2),
        "median_latency_ms": round(float(np.median(timings_arr)), 2),
        "p95_latency_ms": round(float(np.percentile(timings_arr, 95)), 2),
        "p99_latency_ms": round(float(np.percentile(timings_arr, 99)), 2),
        "min_latency_ms": round(float(np.min(timings_arr)), 2),
        "max_latency_ms": round(float(np.max(timings_arr)), 2),
        "fps": round(1000.0 / float(np.mean(timings_arr)), 1),
        "avg_detections_per_frame": round(float(np.mean(all_detections)), 1),
        "gpu_memory_mb": round(gpu_mem_mb, 1),
        "classes_detected": dict(sorted(class_counts.items(), key=lambda x: -x[1])),
        "total_objects_detected": sum(class_counts.values()),
    }
    
    logger.info(f"  ✓ {model_name}: {results_dict['mean_latency_ms']:.1f}ms avg "
               f"({results_dict['fps']:.0f} FPS), "
               f"{results_dict['avg_detections_per_frame']:.1f} dets/frame, "
               f"{results_dict['gpu_memory_mb']:.0f}MB GPU")
    
    return results_dict


def save_visual_comparisons(frames: list, models_results: dict, num_samples: int = 8):
    """Save side-by-side detection visualizations for each model."""
    from ultralytics import YOLO
    
    vis_dir = os.path.join(OUTPUT_DIR, "visual_comparisons")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Pick diverse frames for visualization
    random.seed(123)
    vis_frames = random.sample(frames[NUM_WARMUP:], min(num_samples, len(frames) - NUM_WARMUP))
    
    for model_name in models_results.keys():
        logger.info(f"  Generating visualizations for {model_name}...")
        model = YOLO(model_name)
        model_dir = os.path.join(vis_dir, model_name.replace(".pt", ""))
        os.makedirs(model_dir, exist_ok=True)
        
        for frame_path in vis_frames:
            results = model(frame_path, verbose=False, imgsz=640)
            for r in results:
                annotated = r.plot()  # Get annotated image
                fname = os.path.basename(frame_path)
                cv2.imwrite(os.path.join(model_dir, fname), annotated)


def estimate_jetson_performance(rtx4070_results: dict) -> dict:
    """Estimate Jetson Orin Nano performance from RTX 4070 Super benchmarks.
    
    RTX 4070 Super: ~7168 CUDA cores, 12GB, ~35 TFLOPS FP16
    Jetson Orin Nano 8GB: 1024 CUDA cores, 8GB shared, ~40 TOPS INT8
    
    Scaling factor: RTX 4070S FP16 is roughly ~5-8x faster than Orin Nano FP16.
    But TensorRT on Orin optimizes well, so ~4-6x slower than RTX PyTorch.
    """
    # Conservative scaling: Orin Nano TensorRT FP16 ≈ RTX 4070 PyTorch * 3-5x
    # (TensorRT partially compensates for fewer CUDA cores)
    scale_factor = 4.0  # Conservative estimate
    
    jetson_est = {}
    for model_name, results in rtx4070_results.items():
        jetson_latency = results["median_latency_ms"] * scale_factor
        jetson_est[model_name] = {
            "estimated_latency_ms": round(jetson_latency, 1),
            "estimated_fps": round(1000.0 / jetson_latency, 1),
            "estimated_latency_trt_int8_ms": round(jetson_latency * 0.5, 1),  # INT8 is ~2x faster
            "estimated_fps_trt_int8": round(1000.0 / (jetson_latency * 0.5), 1),
        }
    return jetson_est


def main():
    import torch
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU detected: {gpu_name}")
        device = "cuda"
    else:
        logger.warning("No CUDA GPU found. Running on CPU (will be slower).")
        device = "cpu"
    
    # Collect sample frames
    frames = collect_sample_frames(NUM_SAMPLE_FRAMES)
    logger.info(f"Selected {len(frames)} frames for benchmark")
    
    # Models to benchmark
    models = [
        "yolov8n.pt",   # YOLOv8 Nano — baseline
        "yolov8s.pt",   # YOLOv8 Small — accuracy upgrade
        "yolov10n.pt",  # YOLOv10 Nano — NMS-free
        "yolo11n.pt",   # YOLO11 Nano — latest gen
        "yolo11s.pt",   # YOLO11 Small — latest gen, more accuracy
    ]
    
    all_results = {}
    
    logger.info("=" * 60)
    logger.info("STARTING MODEL BENCHMARK")
    logger.info(f"  Device: {device}")
    logger.info(f"  Frames: {len(frames)} ({NUM_WARMUP} warmup + {len(frames) - NUM_WARMUP} test)")
    logger.info(f"  Models: {len(models)}")
    logger.info("=" * 60)
    
    for model_name in models:
        try:
            result = benchmark_yolo_model(model_name, frames, device=device)
            all_results[model_name] = result
        except Exception as e:
            logger.error(f"  ✗ {model_name} FAILED: {e}")
    
    # Jetson estimates
    logger.info("\nEstimating Jetson Orin Nano performance...")
    jetson_est = estimate_jetson_performance(all_results)
    
    # Visual comparisons
    logger.info("\nGenerating visual comparison images...")
    save_visual_comparisons(frames, all_results, num_samples=8)
    
    # Print summary table
    logger.info("\n" + "=" * 100)
    logger.info("BENCHMARK RESULTS — RTX 4070 Super (PyTorch CUDA)")
    logger.info("=" * 100)
    logger.info(f"{'Model':<15} {'Params':>8} {'Size':>8} {'Avg ms':>8} {'Med ms':>8} {'P95 ms':>8} "
               f"{'FPS':>6} {'Dets/F':>7} {'VRAM':>8} {'Classes':>8}")
    logger.info("-" * 100)
    
    for name, r in all_results.items():
        logger.info(f"{name:<15} {r['params_millions']:>7.1f}M {r['model_size_mb']:>7.1f}M "
                   f"{r['mean_latency_ms']:>7.1f} {r['median_latency_ms']:>7.1f} "
                   f"{r['p95_latency_ms']:>7.1f} {r['fps']:>5.0f} "
                   f"{r['avg_detections_per_frame']:>6.1f} {r['gpu_memory_mb']:>7.0f}M "
                   f"{len(r['classes_detected']):>7}")
    
    logger.info("\n" + "=" * 100)
    logger.info("ESTIMATED JETSON ORIN NANO PERFORMANCE")
    logger.info("=" * 100)
    logger.info(f"{'Model':<15} {'FP16 Latency':>14} {'FP16 FPS':>10} {'INT8 Latency':>14} {'INT8 FPS':>10}")
    logger.info("-" * 100)
    
    for name, est in jetson_est.items():
        logger.info(f"{name:<15} {est['estimated_latency_ms']:>12.1f}ms {est['estimated_fps']:>9.1f} "
                   f"{est['estimated_latency_trt_int8_ms']:>12.1f}ms {est['estimated_fps_trt_int8']:>9.1f}")
    
    # India-specific class analysis
    logger.info("\n" + "=" * 100)
    logger.info("INDIA-SPECIFIC CLASS DETECTION ANALYSIS")
    logger.info("=" * 100)
    
    india_classes = ["car", "truck", "bus", "motorcycle", "bicycle", "person", "cow", "dog", 
                     "traffic light", "stop sign", "cat", "horse", "sheep"]
    
    for name, r in all_results.items():
        detected = r["classes_detected"]
        india_found = {k: v for k, v in detected.items() if k in india_classes}
        logger.info(f"\n  {name}:")
        logger.info(f"    Total objects: {r['total_objects_detected']}")
        logger.info(f"    Top classes: {dict(list(detected.items())[:10])}")
        logger.info(f"    India-relevant: {india_found}")
    
    # Save full results
    output = {
        "benchmark_config": {
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
            "num_test_frames": len(frames) - NUM_WARMUP,
            "input_size": 640,
        },
        "rtx4070_results": all_results,
        "jetson_estimates": jetson_est,
    }
    
    results_path = os.path.join(OUTPUT_DIR, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    logger.info(f"\nFull results saved → {results_path}")
    logger.info(f"Visual comparisons saved → {os.path.join(OUTPUT_DIR, 'visual_comparisons')}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
