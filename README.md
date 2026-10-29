# 🚗 DREIVE — Deep Real-time Edge Inference for Vehicle Environment

> An end-to-end autonomous driving perception and decision-making system purpose-built for Indian road conditions, powered by multi-task deep learning and real-time sensor fusion, deployable on edge hardware (NVIDIA Jetson Orin Nano 8GB).

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-red?logo=pytorch)
![YOLO](https://img.shields.io/badge/Ultralytics-YOLO11-green?logo=yolo)
![TensorRT](https://img.shields.io/badge/NVIDIA-TensorRT_FP16/INT8-76B900?logo=nvidia)
![Jetson](https://img.shields.io/badge/Edge-Jetson_Orin_Nano_8GB-76B900?logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Abstract

Indian roads present some of the most challenging driving environments globally — unmarked lanes, heterogeneous traffic (auto-rickshaws, cattle, two-wheelers), aggressive lane-changing, and minimal infrastructure. Existing autonomous driving systems trained on Western datasets (COCO, BDD100K) fail catastrophically in these conditions.

**DREIVE** addresses this by building a custom perception pipeline from scratch:
- A **multi-task AI system** combining real-time object detection and semantic segmentation
- Trained on a **21,000+ image unified dataset** merging the Indian Driving Dataset (IDD) with custom dashcam footage and synthetically generated adverse weather conditions
- Fused with a **2D LiDAR** for hardware-level collision avoidance
- Optimized for **30–45 FPS** on a $249 edge device via TensorRT quantization

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DREIVE — System Architecture                  │
│                                                                 │
│  ┌──────────┐    ┌──────────────────────────────────────────┐   │
│  │  Camera   │───▶│  YOLO11s Object Detection (TensorRT)    │   │
│  │  (30 Hz)  │    │  11 classes · 40.5% mAP · GPU NMS      │   │
│  └──────────┘    └──────────────┬───────────────────────────┘   │
│                                 │                               │
│  ┌──────────┐    ┌──────────────▼───────────────────────────┐   │
│  │  Camera   │───▶│  YOLO11n-seg Segmentation (TensorRT)    │   │
│  │  (shared) │    │  12 classes · 94.4% road mAP · FP16     │   │
│  └──────────┘    └──────────────┬───────────────────────────┘   │
│                                 │                               │
│  ┌──────────┐    ┌──────────────▼───────────────────────────┐   │
│  │ RPLiDAR  │───▶│  Sensor Fusion Engine                    │   │
│  │ A1 (5Hz) │    │  Async threads · AEB logic · Dashboard   │   │
│  └──────────┘    └──────────────────────────────────────────┘   │
│                                                                 │
│  Target Hardware: NVIDIA Jetson Orin Nano 8GB (40 TOPS)         │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

| Feature | Description |
|:---|:---|
| **India-Specific Detection** | Detects auto-rickshaws, riders, animals, and other India-unique classes that Western models completely miss |
| **Dual AI Heads** | Simultaneous object detection (bounding boxes) + semantic segmentation (pixel-level road mapping) |
| **NMS-Free GPU Inference** | TensorRT `EfficientNMS_TRT` plugin bakes Non-Maximum Suppression into the GPU, eliminating CPU bottlenecks |
| **Synthetic Weather Augmentation** | Algorithmically generated rain, fog, and night conditions to guarantee robustness without real adverse-weather data |
| **Adaptive Frame Extraction** | Intelligent video processing that extracts more frames from complex scenes (intersections) and fewer from simple ones (highways) |
| **Sensor Fusion (Camera + LiDAR)** | Asynchronous multi-threaded architecture fusing 30Hz camera with 5Hz LiDAR for hardware-level collision avoidance |
| **Edge-Optimized** | FP16/INT8 TensorRT quantization achieving 30–45 FPS on a $249 Jetson Orin Nano |

---

## 📊 Performance Metrics

### Object Detection (YOLO11s)

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|:---|:---:|:---:|:---:|:---:|
| Auto-rickshaw | 81.4% | 55.1% | **61.5%** | 43.0% |
| Car | 77.8% | 53.1% | **58.5%** | 40.8% |
| Motorcycle | 74.9% | 52.8% | **57.1%** | 32.3% |
| Bus | 65.9% | 45.6% | **48.6%** | 35.8% |
| Truck | 66.7% | 43.4% | **48.1%** | 33.8% |
| Rider | 75.7% | 43.1% | **47.5%** | 27.3% |
| Person | 73.7% | 34.9% | **40.9%** | 23.0% |
| Bicycle | 61.9% | 31.2% | **29.1%** | 16.7% |
| Traffic Sign | 56.2% | 23.9% | **23.1%** | 12.7% |
| Animal | 62.9% | 12.3% | **15.5%** | 8.2% |
| Traffic Light | 80.0% | 12.4% | **15.3%** | 6.3% |
| **Overall** | **70.6%** | **37.1%** | **40.5%** | **25.4%** |

### Semantic Segmentation (YOLO11n-seg)

| Class | mAP@50 (Box) | mAP@50 (Mask) |
|:---|:---:|:---:|
| **Road** | **94.4%** | **92.1%** |
| Auto-rickshaw | 72.2% | 43.4% |
| Car | 66.1% | 41.4% |
| Motorcycle | 61.3% | 38.4% |
| Sidewalk | 52.7% | 27.1% |

### Model Comparison (YOLOv10n vs YOLO11s)

| Metric | YOLOv10n (Nano) | YOLO11s (Small) | Δ Improvement |
|:---|:---:|:---:|:---:|
| Parameters | 2.27M | 9.42M | 4.1x larger |
| Overall mAP@50 | 34.1% | **40.5%** | **+18.8%** |
| Auto-rickshaw | 55.9% | **61.5%** | +10.0% |
| Traffic Light | 3.9% | **15.3%** | +292.3% |
| Inference (Jetson) | 45+ FPS | **30–40 FPS** | Acceptable |

### Inference Speed

| Platform | Precision | FPS | Latency |
|:---|:---:|:---:|:---:|
| RTX 4070 Super (Desktop) | FP32 | ~70 | 14ms |
| Jetson Orin Nano (Edge) | FP16 | 30–35 | 30ms |
| Jetson Orin Nano (Edge) | INT8 | 40–45 | 22ms |

---

## 🗂️ Dataset

### Unified Training Dataset: 21,000+ Images

| Source | Images | Description |
|:---|:---:|:---|
| **IDD (Indian Driving Dataset)** | 8,024 | Ground-truth bounding boxes converted from JSON polygon annotations |
| **Custom Dashcam Footage** | 10,731 | 46 videos from Mumbai/Karjat highways, pseudo-labeled via YOLO11x Teacher model |
| **Synthetic Weather** | 3,000 | Algorithmically generated rain, fog, and night conditions using OpenCV |
| **Total** | **21,755** | Covers urban, suburban, highway, day, night, rain, and fog |

### Detection Classes (11)
```
car · bus · truck · autorickshaw · motorcycle · bicycle
person · rider · animal · traffic sign · traffic light
```

### Segmentation Classes (12)
```
road · sidewalk · non-drivable · car · bus · truck
autorickshaw · motorcycle · bicycle · person · rider · animal
```

---

## 🛠️ Technical Pipeline

### Phase 1: Intelligent Frame Extraction
- Built a hybrid adaptive frame extractor combining SSIM-based scene change detection, optical flow complexity scoring, and Laplacian blur filtering
- Extracted **12,677 high-quality frames** from 46 dashcam videos (2.83 GB)
- Discovered and auto-corrected inverted dashcam footage (180° rotation)

### Phase 2: Data Pipeline & Annotation
- Parsed IDD dataset JSON polygon annotations → YOLO bounding box format (8,024 images)
- Parsed IDD polygon masks → YOLO segmentation format (198,801 polygons across 8,065 images)
- Built a Teacher-Student pseudo-labeling pipeline using YOLO11x to auto-annotate custom frames
- Generated 3,000 synthetic weather-augmented images (rain, fog, night)

### Phase 3: Multi-Task Model Training
- **Object Detection:** YOLO11s trained for 141 epochs (early stopping at epoch 111) with advanced augmentations (mosaic, mixup, perspective, erasing, scale, HSV shifts)
- **Semantic Segmentation:** YOLO11n-seg trained for 30 epochs on IDD polygon masks
- Resolved Domain Gap between custom dashcam and IDD by aggressive dataset mixing

### Phase 4: Sensor Fusion
- Designed asynchronous multi-threaded fusion engine (`sensor_fusion.py`)
- Camera thread (30Hz) and LiDAR thread (5Hz) run independently
- Fusion logic: Camera classifies objects + LiDAR measures distance → AEB decision
- Emergency braking threshold: Object in path AND distance < 3 meters

### Phase 5: Edge Deployment
- TensorRT FP16/INT8 quantization with automatic calibration
- `EfficientNMS_TRT` GPU plugin eliminates CPU-bound NMS bottleneck
- Compiled `.engine` files optimized specifically for Jetson Orin Nano hardware

---

## 📁 Project Structure

```
DREIVE/
├── frame_extractor/              # Intelligent frame extraction module
│   ├── extractors/
│   │   ├── hybrid.py             # Adaptive extractor (SSIM + complexity + fixed interval)
│   │   ├── scene_change.py       # SSIM & histogram-based scene change detection
│   │   ├── complexity_scorer.py  # Edge density + optical flow scoring
│   │   └── fixed_interval.py     # Baseline fixed-rate extractor
│   ├── processors/
│   │   ├── blur_detector.py      # Laplacian variance blur filtering
│   │   └── dedup.py              # Perceptual hash deduplication
│   ├── utils/
│   │   ├── video_utils.py        # OpenCV video I/O utilities
│   │   └── file_utils.py         # File management & manifest generation
│   └── config.py                 # Dataclass-based pipeline configuration
│
├── unified_dataset/              # Combined training dataset
│   ├── images/                   # Train & val splits (~21,000 images)
│   ├── labels/                   # YOLO-format bounding box annotations
│   └── data.yaml                 # Class definitions (11 classes)
│
├── seg_dataset/                  # Segmentation dataset
│   ├── images/                   # IDD images
│   ├── labels/                   # YOLO polygon mask annotations
│   └── seg_data.yaml             # Class definitions (12 classes)
│
├── runs/                         # Training outputs
│   ├── detect/indian_driving_yolo11s/weights/best.pt
│   └── segment/indian_segmentation_v1/weights/best.pt
│
├── metrics_and_results/          # Performance reports & comparisons
│
├── train_model.py                # YOLO11s detection training script
├── train_seg_model.py            # YOLO11n-seg segmentation training script
├── idd_parser.py                 # IDD → YOLO bounding box converter
├── idd_seg_parser.py             # IDD → YOLO segmentation polygon converter
├── auto_annotator.py             # Teacher-Student pseudo-labeling pipeline
├── synthetic_weather_engine.py   # Rain/fog/night synthetic augmentation
├── diverse_frame_sampler.py      # K-Means clustering for frame diversity
├── sensor_fusion.py              # Camera + LiDAR fusion engine (Jetson)
├── export_tensorrt.py            # TensorRT FP16/INT8 + GPU NMS exporter
└── run_full_extraction.py        # Batch video processing orchestrator
```

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install ultralytics opencv-python numpy scikit-learn tqdm rplidar-roboticia
```

### 1. Train Object Detection
```bash
python train_model.py
```

### 2. Train Semantic Segmentation
```bash
python train_seg_model.py
```

### 3. Export to TensorRT (Run on Jetson)
```bash
python export_tensorrt.py
```

### 4. Run Sensor Fusion (Live Inference)
```bash
python sensor_fusion.py
```

---

## 🔧 Hardware

| Component | Specification | Role |
|:---|:---|:---|
| **Training GPU** | NVIDIA GeForce RTX 4070 Super (12GB VRAM) | Model training & benchmarking |
| **Edge Device** | NVIDIA Jetson Orin Nano 8GB (40 TOPS) | Real-time inference & deployment |
| **Camera** | Dashboard-mounted forward-facing webcam | Visual perception |
| **LiDAR** | RPLiDAR A1 (2D, 360°, 12m range) | Distance measurement & collision avoidance |

---

## 🧪 Key Technical Decisions

| Decision | Rationale |
|:---|:---|
| **YOLO11s over YOLOv10n** | 18.8% higher accuracy; NMS bottleneck eliminated via TensorRT GPU plugin |
| **YOLO11n-seg for segmentation** | Lightweight enough for Jetson while providing pixel-level road mapping |
| **Teacher-Student pseudo-labeling** | Enabled annotation of 10,731 custom frames without manual labeling |
| **Synthetic weather augmentation** | Eliminated the need to collect real rain/fog/night driving footage |
| **Async sensor fusion** | Prevents slow 5Hz LiDAR from throttling 30Hz camera inference |
| **INT8 quantization** | 2x speed boost over FP16 with <2% accuracy loss via calibration |

---

## 📈 Future Roadmap

- [ ] Add pothole and speed breaker detection classes
- [ ] Integrate multi-object tracking (ByteTrack / DeepSORT)
- [ ] Build lane departure warning using segmentation masks
- [ ] Upgrade from 2D LiDAR to 3D LiDAR for vertical obstacle detection
- [ ] Implement PID/MPC controller for autonomous steering
- [ ] Deploy as passive ADAS dashboard for real-world data collection

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Indian Driving Dataset (IDD)](https://idd.insaan.iiit.ac.in/) — IIIT Hyderabad
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — Model architecture & training framework
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) — Edge inference optimization
- [RPLiDAR](https://www.slamtec.com/en/Lidar/A1) — 2D LiDAR SDK
