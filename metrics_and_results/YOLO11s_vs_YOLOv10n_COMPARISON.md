# YOLO11s vs YOLOv10n — Head-to-Head Performance Comparison

Both models were trained on the identical 21,000+ image unified dataset (IDD + Custom Dashcam + Synthetic Weather).

---

## Per-Class Accuracy (mAP@50)

| Class | YOLOv10n (Nano) | YOLO11s (Small) | Improvement |
|:---|:---:|:---:|:---:|
| **Auto-rickshaw** | 55.9% | **61.5%** | +10.0% ↑ |
| **Car** | 53.3% | **58.5%** | +9.8% ↑ |
| **Motorcycle** | 51.8% | **57.1%** | +10.2% ↑ |
| **Bus** | 40.7% | **48.6%** | +19.4% ↑ |
| **Truck** | 41.2% | **48.1%** | +16.7% ↑ |
| **Person** | 35.4% | **40.9%** | +15.5% ↑ |
| **Rider** | 43.4% | **47.5%** | +9.4% ↑ |
| **Bicycle** | 21.6% | **29.1%** | +34.7% ↑ |
| **Animal** | 9.8% | **15.5%** | +58.2% ↑ |
| **Traffic Sign** | 18.5% | **23.1%** | +24.9% ↑ |
| **Traffic Light** | 3.9% | **15.3%** | +292.3% ↑ |
| **Overall (mAP@50)** | 34.1% | **40.5%** | +18.8% ↑ |
| **Overall (mAP@50-95)** | 21.1% | **25.4%** | +20.4% ↑ |

---

## Training Summary

| Metric | YOLOv10n | YOLO11s |
|:---|:---:|:---:|
| Parameters | 2.27M | **9.42M** |
| GFLOPs | 6.6 | 21.4 |
| Best Epoch | 219 | **111** (learned faster) |
| Total Training Time | 7.1 hours | **4.1 hours** |
| Model File Size | 5.7 MB | 19.2 MB |
| Precision (P) | 62.2% | **70.6%** |
| Recall (R) | 30.7% | **37.1%** |

---

## Key Takeaways

### 1. YOLO11s learns Indian traffic faster
Despite having 4x more parameters, YOLO11s peaked at **Epoch 111** vs YOLOv10n's Epoch 219. The larger brain understood the complex overlapping patterns of Indian traffic much more efficiently.

### 2. Traffic Lights: The biggest win (+292%)
YOLOv10n essentially failed at traffic lights (3.9%). YOLO11s jumped to 15.3%. This is because traffic lights are tiny objects that require deeper feature extraction layers — exactly what the "Small" model provides.

### 3. The Occlusion Victory
Person detection improved from 35.4% → 40.9%, and Rider detection from 43.4% → 47.5%. These are the hardest classes because people and riders are frequently hidden behind vehicles. YOLO11s's deeper network can "see through" the occlusion better.

### 4. NMS Bottleneck: Solved
The `export_tensorrt.py` script has `nms=True`, which bakes NVIDIA's `EfficientNMS_TRT` C++ plugin directly into the GPU engine. The Jetson's CPU will never touch the raw bounding boxes.

---

## Deployment Readiness

| Component | Status | File |
|:---|:---:|:---|
| Object Detection (YOLO11s) | ✅ Ready | `runs/detect/indian_driving_yolo11s/weights/best.pt` |
| Semantic Segmentation (YOLO11n-seg) | ✅ Ready | `runs/segment/indian_segmentation_v1/weights/best.pt` |
| TensorRT Export Script | ✅ Ready | `export_tensorrt.py` (FP16 + INT8 + GPU NMS) |
| Sensor Fusion Engine | ✅ Ready | `sensor_fusion.py` (Camera + LiDAR threading) |
