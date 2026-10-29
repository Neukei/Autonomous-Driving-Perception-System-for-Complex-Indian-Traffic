# Chapter 1: DREIVE System Architecture & Calibrated Hardware Stack

## 1. Executive Hardware Specification

The DREIVE perception pipeline is calibrated specifically for the following production hardware components:

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                         PRODUCTION SENSOR TOPOLOGY                             │
│                                                                                │
│   [ Exterior Roof Mount ]                                                      │
│   Slamtec RPLiDAR A1 / A1M8 (2D 360° Laser Scanner) ──(USB UART CP2102)─┐      │
│   • 0.15m – 12.0m Range, 8000 samples/sec, 5.5–10 Hz                     │      │
│   • Elevation: Z_lidar = 1.50m (Ground reference)                        │      │
│                                                                          ▼      │
│   [ Interior Cabin Mount (Rearview Mirror) ]                   ┌──────────────┐│
│   Logitech C930e Professional HD Webcam                       │ EXTRINSIC    ││
│   • Exact 16:9 FOV: 82.1° Horizontal, 52.2° Vertical          │ CALIBRATION  ││
│   • 1280x720 @ 30 FPS                                         │ TRANSFORM    ││
│   • Elevation: Z_cam = 1.25m (Ground reference)               └──────┬───────┘│
│                                                                      │        │
│   Rigid Spatial Extrinsic Offsets:                                   │        │
│   • ΔZ (Height delta)  = +0.25 m (LiDAR higher than camera)         │        │
│   • ΔY (Forward delta) = +0.15 m (LiDAR forward of camera)          ▼        │
│   • ΔX (Lateral delta) =  0.00 m (Centered along vehicle centerline)         │
│                                                                               │
│                 ┌───────────────────────────────────────────────┐             │
│                 │ NVIDIA JETSON ORIN NANO (8GB LPDDR5)          │             │
│                 │ • 40 TOPS INT8 / 20 TFLOPS FP16 Tensor Cores │             │
│                 │ • 68 GB/s Unified LPDDR5 Memory Bus           │             │
│                 └───────────────────────────────────────────────┘             │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Does the System Auto-Compensate for the Distance Between Sensors?

### **Yes! The Extrinsic Transformation Matrix is actively calculated:**

Because the **Logitech C930e** is inside the cabin (rearview mirror) and the **Slamtec RPLiDAR A1M8** is outside on the roof, they do not share the exact same physical coordinates. A target measured at $5.00\text{ meters}$ by the roof LiDAR is actually slightly farther from the camera lens inside the cabin.

### The Rigid Extrinsic Spatial Transform
In [`lidar_projection.py`](file:///c:/Users/ADMIN/Desktop/MAJOR%20PROJECT/lidar_projection.py) and [`sensor_rig_config.py`](file:///c:/Users/ADMIN/Desktop/MAJOR%20PROJECT/sensor_rig_config.py), each raw polar point $(r, \theta)$ from the Slamtec A1 is converted into the **Logitech C930e Optical Frame** via rigid 3D translation:

$$\begin{bmatrix} X_{\text{cam}} \\ Y_{\text{cam}} \\ Z_{\text{cam}} \end{bmatrix} = \begin{bmatrix} r \sin(\theta + \theta_{\text{trim}}) + \Delta X \\ -\Delta Z \\ r \cos(\theta + \theta_{\text{trim}}) + \Delta Y \end{bmatrix}$$

Where:
- $\Delta X = 0.00\text{ m}$: Both sensors aligned with the vehicle centerline.
- $\Delta Y = +0.15\text{ m}$: Slamtec A1 is mounted $15\text{ cm}$ farther forward toward the windshield front edge.
- $\Delta Z = +0.25\text{ m}$: Slamtec A1 is mounted $25\text{ cm}$ higher on the roof than the camera lens inside.

### True Optical Distance Calculation
The distance metric displayed on the bounding box and tracer lines is the **true Euclidean distance** to the vehicle:
$$D_{\text{true}} = \sqrt{X_{\text{cam}}^2 + Y_{\text{cam}}^2 + Z_{\text{cam}}^2}$$

---

## 3. Logitech C930e Specific Optical Calibration

The Logitech C930e has an ultra-wide $90^\circ$ diagonal Field of View. In standard $16:9$ aspect ratio ($1280 \times 720$), its exact optical characteristics are:
- **Horizontal FOV ($\text{FOV}_h$)**: **$82.1^\circ$** ($1.4329\text{ rad}$)
- **Vertical FOV ($\text{FOV}_v$)**: **$52.2^\circ$** ($0.9110\text{ rad}$)
- **Focal Length in Pixels ($f_x$)**:
  $$f_x = \frac{W / 2}{\tan(\text{FOV}_h / 2)} = \frac{640}{\tan(41.05^\circ)} \approx \mathbf{734.9\text{ px}}$$
- **Focal Length in Pixels ($f_y$)**:
  $$f_y = \frac{H / 2}{\tan(\text{FOV}_v / 2)} = \frac{360}{\tan(26.1^\circ)} \approx \mathbf{734.8\text{ px}}$$

With $f_x \approx f_y$, the C930e possesses square pixels ($1.0$ pixel aspect ratio), ensuring zero anamorphic distortion during 3D LiDAR point ray-casting.
