# Chapter 6: Ragnarök ADAS Command Center & Operations

## 1. Design Philosophy
The **DREIVE Command Center** is built using a modern **Norse RPG design system** inspired by *God of War Ragnarök*. It maps game UI concepts directly to mission-critical automotive perception:

| Ragnarök Game Concept | DREIVE ADAS Mapping |
| :--- | :--- |
| **Kratos Level Diamond (Level 8)** | Edge Deployment Tier Indicator |
| **Character Attributes (Strength, Defense, Runic)** | Hardware Telemetry (Throughput, Latency Budget, VRAM Pool) |
| **Weapon Loadout (Axe, Blades, Spear)** | Perception Sensors (Camera, RPLiDAR, YOLO11s, YOLO11n-seg) |
| **Skill Trees (Technique, Ranged, Melee)** | ADAS Feature Hierarchy (Optical, Spatial Fusion, Optimization) |
| **Codex** | Model Benchmarks & Offline Verification Tables |

---

## 2. The 4 Mission Viewport Modes

In **Tab 1: Mission Control**, operators can instantly toggle between four rendering modes without restarting inference:

### Mode 1: Combined HUD (`'combined'`)
- The standard operational ADAS view.
- Overlays bounding boxes, ByteTrack persistent IDs, drivable road carpet (emerald green), ground LiDAR ray intercepts, and **dynamic distance tracer lines**.

### Mode 2: Camera Vision Only (`'camera'`)
- Clean visual feed without LiDAR laser points or tracer lines.
- Used for evaluating pure optical detection and classification boundaries.

### Mode 3: 2D Polar LiDAR Radar (`'lidar_radar'`)
- Dedicated dark-slate polar radar display.
- Visualizes concentric distance rings ($3\text{m}, 6\text{m}, 9\text{m}, 12\text{m}$), forward camera $78^\circ$ FOV cone, and $360^\circ$ obstacle point echoes.
- Maps detected vehicles as colored radar blips with distance labels.

### Mode 4: Split View (`'split'`)
- Horizontal side-by-side display:
  - **Left Half**: Combined Camera Perception View.
  - **Right Half**: Synchronized 2D Polar LiDAR Radar HUD.

---

## 3. Operations & Quick-Start Guide

### Starting the Command Center
Execute in the project root:
```powershell
python app.py
```
Open a browser to **`http://127.0.0.1:5000`**.

### Connecting Physical RPLiDAR A1M8
1. Plug the LiDAR USB cable into any USB 3.0 port on your computer or Jetson.
2. In the right-hand panel under **Feed & Sensor Controls**, click **`Scan COM`**.
3. Select your device (e.g., `COM3` or `/dev/ttyUSB0`) and click **`Connect`**.
4. The status pill will switch from **`SIMULATED LIDAR`** to **`LIDAR: CONNECTED (PORT)`**.

### Capturing Proof of Performance
- **Instant Frame Snapshot**: Click the camera icon in the top right to save an annotated frame to `metrics_and_results/snapshots/`.
- **Session Video Recording**: Click **`REC SESSION`** to record real-time driving sessions directly to `metrics_and_results/recordings/session_<timestamp>.mp4`.
