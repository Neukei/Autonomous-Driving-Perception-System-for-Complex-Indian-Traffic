# Chapter 2: Dataset Curation & Synthetic Augmentation Pipeline

## 1. Unified Dataset Architecture (21,755 Images)

A critical failure point of off-the-shelf vision models (COCO, KITTI) on Indian roads is **domain gap** — standard models fail to recognize three-wheeled auto-rickshaws, overloaded trucks, stray animals, and dense two-wheeler swarms.

To solve this, DREIVE unifies three complementary data streams into a single standardized YOLO-format training corpus:

```
┌────────────────────────────────────────────────────────────────────────┐
│               DREIVE UNIFIED DATASET: 21,755 IMAGES                     │
│                                                                        │
│   1. Indian Driving Dataset (IDD)           8,024 images (36.9%)       │
│      • Gold-standard ground truth polygons converted to YOLO boxes     │
│                                                                        │
│   2. Real-World Dashcam Extraction          10,731 images (49.3%)      │
│      • 46 raw highway videos (Mumbai-Karjat corridor)                  │
│      • Adaptive SSIM + Complexity filtering                            │
│      • Pseudo-labeled via Teacher YOLO11x model                        │
│                                                                        │
│   3. Synthetic Weather Engine               3,000 images (13.8%)       │
│      • Algorithmically generated Rain, Fog, and Night augmentation     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. The 11 Detection Classes

The class schema covers all major obstacle categories on Indian roadways:

| Class ID | Name | Relevance / Specificity |
| :---: | :--- | :--- |
| `0` | `car` | Standard sedans, hatchbacks, SUVs |
| `1` | `bus` | Private and public municipal buses (MSRTC, BEST) |
| `2` | `truck` | Commercial carriers, multi-axle freight vehicles |
| `3` | `autorickshaw` | **India-Unique:** 3-wheeled passenger vehicles |
| `4` | `motorcycle` | Motorcycles, mopeds, scooters (high collision vulnerability) |
| `5` | `bicycle` | Non-motorized human transport |
| `6` | `person` | Pedestrians crossing or walking along road shoulders |
| `7` | `rider` | Operator/passenger on motorcycle or bicycle |
| `8` | `animal` | Cattle, stray dogs, and horses obstructing roadways |
| `9` | `traffic sign` | Regulatory, warning, and informational boards |
| `10` | `traffic light`| Standard red/yellow/green signal lamps |

---

## 3. Intelligent Frame Extraction Pipeline (`frame_extractor/`)

Extracting video at a naive fixed frame rate (e.g. 1 frame every second) produces massive data redundancy during highway cruising and misses fast transient events at crowded intersections.

DREIVE utilizes a **Hybrid Adaptive Extractor**:
1. **Scene Change Detection (SSIM)**: Compares Structural Similarity Index between consecutive frames. When $\text{SSIM} < 0.65$, a major scene change is flagged.
2. **Optical Flow & Edge Complexity Scoring**: Uses Sobel/Laplacian gradient density to calculate visual entropy:
   $$\text{Complexity} = \frac{1}{N} \sum |\nabla I(x, y)|$$
   High-complexity scenes (busy markets, junctions) extract frames at **3–5 FPS**, while static highway cruising drops to **0.5 FPS**.
3. **Blur & Inversion Auto-Correction**: Frames with Laplacian variance $\sigma^2 < 100$ are rejected as motion-blurred. Upside-down dashcam footage is detected and automatically rotated $180^\circ$.

---

## 4. Synthetic Weather Augmentation Engine (`synthetic_weather_engine.py`)

Adverse weather driving footage (heavy monsoon rain, dense fog, night driving) is notoriously difficult to capture safely. DREIVE uses an algorithmic OpenCV synthetic augmentation engine:
- **Synthetic Rain**: Generates directional motion-blurred raindrops using 2D line kernels with specular reflection highlights.
- **Atmospheric Fog**: Implements optical depth transmission maps using Koschmieder's Law:
  $$I(x) = J(x) e^{-\beta d(x)} + A(1 - e^{-\beta d(x)})$$
  where $\beta$ is the scattering coefficient and $A$ is the atmospheric light.
- **Night Simulation**: Non-linear gamma curve shifting with simulated headlight flare cones.

> **Offline Verification Result:** Removing these 3,000 synthetic weather images causes overall mAP to drop from **40.5%** to **38.9%**, empirically proving a **+1.6% accuracy boost** from synthetic augmentation.
