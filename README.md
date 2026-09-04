# Dreive: Autonomous Driving Perception & Edge AI Pipeline

An end-to-end, edge-optimized autonomous driving perception and sensor fusion pipeline designed specifically for unstructured and complex road environments (trained and validated on the Indian Driving Dataset (IDD) and real-world dashcam video). Optimized for real-time edge deployment on the **Nvidia Jetson Orin Nano (8GB)**.

---

## Key Features

- **Intelligent Hybrid Frame Extraction (`frame_extractor/`)**:
  - Adaptive frame sampling combining fixed-interval and scene change detection.
  - Quality filtering to eliminate blurry, over/underexposed frames.
  - Perceptual deduplication using image hashing algorithms.
  - Multiprocessing-enabled pipeline with JSON manifest tracking.

- **Synthetic Weather Simulation Engine (`synthetic_weather_engine.py`)**:
  - Augments dataset with realistic rain streaks, atmospheric fog, and low-light dusk/night effects.
  - Drastically mitigates domain gap across challenging environmental conditions.

- **Indian Driving Dataset (IDD) Parsers & Processing**:
  - Complete parsing and conversion utilities for IDD bounding box labels (`idd_parser.py`) and segmentation polygon masks (`idd_seg_parser.py`).
  - Automated labeling via YOLO pseudo-labeling (`auto_annotator.py`) and diversity frame sampling (`diverse_frame_sampler.py`).

- **Object Detection & Semantic Segmentation**:
  - NMS-free object detection leveraging **YOLOv10n** for low-latency edge inference.
  - High-precision semantic segmentation using **YOLO11n-seg** for pixel-level drivable surface mapping (94.4% mAP@50 for road segmentation).

- **Multi-Model Edge Benchmarking & TensorRT Compilation**:
  - Comprehensive benchmarking suite (`run_model_benchmark.py`) comparing YOLOv8, YOLOv10, and YOLO11 across FPS, latency, and memory footprints.
  - Automated FP16 TensorRT engine export (`export_tensorrt.py`) targeting 30+ FPS on edge GPUs.

- **Asynchronous Sensor Fusion & AEB System (`sensor_fusion.py`)**:
  - Thread-decoupled asynchronous fusion of high-framerate vision (30 Hz) and 2D RPLiDAR A1 (5 Hz).
  - Autonomous Emergency Braking (AEB) safety logic with dynamic distance thresholds.

---

## Directory Structure

```text
├── frame_extractor/             # Modular video extraction & quality filtering package
│   ├── extractors/              # Fixed-interval, scene-change, and hybrid extractors
│   ├── processors/              # Blur detection, deduplication, and quality filters
│   └── utils/                   # Video/file manipulation helpers
├── metrics_and_results/         # Detailed benchmark comparisons and performance reports
│   ├── FINAL_PERFORMANCE_METRICS.md
│   └── YOLO11s_vs_YOLOv10n_COMPARISON.md
├── benchmark_results/           # Visual comparison outputs and benchmark JSON logs
├── auto_annotator.py            # Automated YOLO pseudo-labeling pipeline
├── diverse_frame_sampler.py     # Diversity frame selection using visual embeddings/clustering
├── export_tensorrt.py           # PyTorch to TensorRT FP16 compilation script
├── idd_parser.py                # IDD bounding box annotation converter
├── idd_seg_parser.py            # IDD semantic segmentation mask converter
├── run_full_extraction.py       # Multi-video pipeline orchestrator
├── run_model_benchmark.py       # Latency, FPS, and detection benchmark harness
├── sensor_fusion.py             # Dual-threaded Camera + RPLiDAR sensor fusion & AEB logic
├── synthetic_weather_engine.py  # Rain, fog, and nighttime weather augmentation engine
├── train_model.py               # Object detection training harness
├── train_seg_model.py           # Semantic segmentation training harness
├── .gitignore                   # Excludes raw datasets, video archives, and heavy weights
└── README.md                    # Project documentation
```

---

## Getting Started

### 1. Prerequisites

- Python 3.10+
- PyTorch with CUDA support
- Ultralytics (`pip install ultralytics`)
- OpenCV (`pip install opencv-python`)

Install core dependencies:
```bash
pip install -r requirements.txt
```

### 2. Video Frame Extraction

Run the hybrid extractor with blur detection and deduplication:
```bash
python run_full_extraction.py
```
Or use the CLI in `frame_extractor/`:
```bash
python frame_extractor/main.py extract --input-dir ./videos --output-dir ./dataset --workers 8 --blur-threshold 120.0
```

### 3. Synthetic Weather Engine

Augment frames with rain, fog, and low-light transformations:
```bash
python synthetic_weather_engine.py
```

### 4. TensorRT Export & Benchmarking

Compile models to TensorRT FP16 engines for Nvidia Jetson:
```bash
python export_tensorrt.py --weights yolov10n.pt --device 0
```
Run comprehensive benchmark comparisons:
```bash
python run_model_benchmark.py
```

### 5. Running Sensor Fusion & AEB

To launch the asynchronous camera and RPLiDAR sensor fusion pipeline:
```bash
python sensor_fusion.py
```

---

## Performance Summary

| Task | Model | Key Target | Validation Result |
|---|---|---|---|
| **Semantic Segmentation** | YOLO11n-seg | Road Surface | **94.4% mAP@50** |
| **Object Detection** | YOLOv10n | Auto-rickshaw | **55.9% mAP@50** |
| **Edge Optimization** | TensorRT FP16 | Jetson Orin Nano | **30+ FPS (Real-Time)** |

Detailed evaluations and comparison breakdowns are available in [`metrics_and_results/`](metrics_and_results/FINAL_PERFORMANCE_METRICS.md).

---

## License

This project is developed for research and development of autonomous perception systems on unstructured roads.
