# DREIVE System Knowledge Base Index

Welcome to the comprehensive technical documentation and system knowledge base for **DREIVE** (*Deep Real-time Edge Inference for Vehicle Environment*).

This knowledge base serves as the formal engineering reference for recruiters, academic reviewers, and system integrators deploying multi-task perception and 2D LiDAR fusion on edge platforms (NVIDIA Jetson Orin Nano).

---

## 📚 Table of Contents

### [01. System Architecture & Hardware Stack](file:///c:/Users/ADMIN/Desktop/MAJOR%20PROJECT/knowledge_base/01_system_architecture.md)
- Executive project overview and Indian roadway challenges.
- Hardware topology & sensor mounting geometry (Interior Cabin Camera + Exterior Roof RPLiDAR A1M8).
- NVIDIA Jetson Orin Nano 8GB specifications, power profile, and LPDDR5 shared memory constraints.

### [02. Dataset Curation & Synthetic Augmentation Pipeline](file:///c:/Users/ADMIN/Desktop/MAJOR%20PROJECT/knowledge_base/02_dataset_curation.md)
- 21,755-image unified dataset breakdown (IDD + Custom Dashcam + Synthetic Weather).
- The 11 Indian Driving Dataset (IDD) detection classes.
- Adaptive SSIM + Complexity scoring frame extractor (`frame_extractor/`).
- Synthetic rain, fog, and night simulation engine (`synthetic_weather_engine.py`).

### [03. Model Zoo & Offline Verification Benchmarks](file:///c:/Users/ADMIN/Desktop/MAJOR%20PROJECT/knowledge_base/03_model_zoo_benchmarks.md)
- Model comparison matrix: YOLOv10n vs YOLO11s vs YOLO11n-seg vs RT-DETR.
- Production YOLO11s per-class precision, recall, and mAP@50 results (Auto-rickshaws: **61.5%**, Cars: **58.5%**, Bikes: **57.1%**).
- YOLOv10n vs YOLO11s head-to-head gains (+18.8% overall mAP, +292% traffic lights).
- Vision Transformers (RT-DETR) vs Convolutional Neural Networks for occlusion reasoning.

### [04. Edge Optimization & TensorRT INT8 Quantization](file:///c:/Users/ADMIN/Desktop/MAJOR%20PROJECT/knowledge_base/04_edge_optimization.md)
- Embedded Jetson Orin Nano bottlenecks: 68 GB/s memory bandwidth & 5GB VRAM ceiling.
- Eliminating CPU post-processing: GPU `EfficientNMS_TRT` TensorRT plugin.
- INT8 quantization via KL-Divergence calibration batches.
- Frame Alternating dual-head pipeline for sustained 30+ FPS edge execution.

### [05. Spatial 2D LiDAR Fusion & Active AEB Decision Engine](file:///c:/Users/ADMIN/Desktop/MAJOR%20PROJECT/knowledge_base/05_sensor_fusion_aeb.md)
- Mathematical derivation of 3D-to-2D pinhole coordinate projection.
- Bounding box association & median cluster distance filtering.
- Dynamic ground-to-object laser tracer vectors & color-coded proximity gradients.
- Autonomous Emergency Braking (AEB) decision thresholds (< 3.0m critical halt).

### [06. Ragnarök ADAS Command Center & Operations](file:///c:/Users/ADMIN/Desktop/MAJOR%20PROJECT/knowledge_base/06_dashboard_operations.md)
- God of War Ragnarök design system mapping (Level 8 crest, attribute bars, skill trees).
- The 4 Mission View modes: Combined HUD, Camera Vision, 2D LiDAR Polar Radar, and Split View.
- Operator guide: Live RPLiDAR COM auto-detection, video switcher, snapshots, and MP4 session recording.
