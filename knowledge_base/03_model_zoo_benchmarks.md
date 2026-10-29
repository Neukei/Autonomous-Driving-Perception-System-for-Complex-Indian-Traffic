# Chapter 3: Model Zoo & Offline Verification Benchmarks

## 1. Model Zoo Comparison Matrix

DREIVE benchmarked multiple architectures across accuracy, model size, and edge throughput:

| Architecture | Paradigm | Parameters | Overall mAP@50 | Orin Nano FPS (FP16) | NMS Handling |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **YOLOv10n** | Dual-label CNN | 2.27M | 34.1% | 45+ FPS | NMS-Free (Architecture) |
| **YOLO11s (Production)** | Modern CNN + C3k2 | 9.42M | **40.5%** | **30–35 FPS** | GPU `EfficientNMS_TRT` Plugin |
| **YOLO11n-seg** | Instance Segmentation | 2.87M | **94.4% (Road)** | 45+ FPS | Mask Proto-Net |
| **RT-DETR-L** | Vision Transformer | 32.9M | *In Benchmarking* | ~25 FPS | Bipartite Set Matching (NMS-Free) |

---

## 2. Production Model Per-Class Breakdown (YOLO11s)

*Trained for 141 epochs (Early stopped at epoch 111 with patience=30) on 21,755 unified images.*

| Class | Precision | Recall | mAP@50 | mAP@50-95 | Engineering Significance |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Auto-rickshaw** | 81.4% | 55.1% | **61.5%** | 43.0% | **Highest scoring vehicle.** Successfully captures non-standard 3-wheel geometry. |
| **Car** | 77.8% | 53.1% | **58.5%** | 40.8% | Standard passenger vehicle baseline. |
| **Motorcycle** | 74.9% | 52.8% | **57.1%** | 32.3% | High-density class; successfully discriminates clustered bikes. |
| **Bus** | 65.9% | 45.6% | **48.6%** | 35.8% | Heavy passenger transport. |
| **Truck** | 66.7% | 43.4% | **48.1%** | 33.8% | Multi-axle commercial transport. |
| **Rider** | 75.7% | 43.1% | **47.5%** | 27.3% | Distinguishes operator sitting on bike vs walking pedestrian. |
| **Person** | 73.7% | 34.9% | **40.9%** | 23.0% | Pedestrian safety barrier. |
| **Bicycle** | 61.9% | 31.2% | **29.1%** | 16.7% | Vulnerable road user. |
| **Traffic Sign** | 56.2% | 23.9% | **23.1%** | 12.7% | Regulatory signage. |
| **Animal** | 62.9% | 12.3% | **15.5%** | 8.2% | Stray cows, dogs on roadway. |
| **Traffic Light** | 80.0% | 12.4% | **15.3%** | 6.3% | Small bounding boxes. Improved **+292%** over YOLOv10n (3.9%). |
| **OVERALL** | **70.6%** | **37.1%** | **40.5%** | **25.4%** | **Production Baseline** |

---

## 3. YOLOv10n vs YOLO11s Head-to-Head Delta

```
mAP@50 Improvement:
YOLOv10n: [=========================>              ] 34.1%
YOLO11s:  [===============================>        ] 40.5%  (+18.8% Relative Boost)

Traffic Light Detection:
YOLOv10n: [==>                                     ] 3.9%
YOLO11s:  [===========>                            ] 15.3% (+292.3% Relative Boost)
```

The 4.1x larger parameter capacity of YOLO11s (9.42M vs 2.27M) allows the network to learn subtle optical cues (traffic light color signals, distant auto-rickshaw canvas tops) that smaller nano models fail to capture.

---

## 4. Vision Transformers vs CNNs: The RT-DETR Rationale

Convolutional networks (YOLO) process images through localized sliding receptive fields, which creates vulnerability when objects heavily occlude each other (e.g. an auto-rickshaw partially hidden behind a bus).

**RT-DETR (Real-Time Detection Transformer)** replaces localized convolution with **Multi-Head Self-Attention**:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
- Every pixel directly queries every other pixel across the entire image.
- Global spatial reasoning resolves overlapping vehicles without candidate box overlap ambiguities.
- NMS is completely bypassed using **Hungarian Bipartite Loss**.
