"""
DREIVE Edge ADAS Mission Control Dashboard Backend.
Engineered for reliable live camera grab, video replay, model switching, and calibration.
"""
import os
import cv2
import time
import json
import math
import threading
import numpy as np
from pathlib import Path
from flask import Flask, Response, jsonify, request, render_template

from fusion_pipeline import FusionPipeline
from rplidar_driver import scan_available_ports, autodetect_rplidar_port
from auto_calibrator import AutoExtrinsicCalibrator
from sensor_rig_config import load_rig_config, save_rig_config

app = Flask(__name__, template_folder="templates", static_folder="static")

# Pipeline instance & calibrator
pipeline = None
pipeline_lock = threading.Lock()
calibrator = AutoExtrinsicCalibrator()

# Streaming state
video_capture = None
is_streaming = False
stream_mode = "video"  # "video" or "live"
current_video_file = None
active_telemetry = {}
available_videos = []

# Recording state
is_recording = False
record_writer = None
record_filename = None

def scan_available_videos():
    global available_videos
    project_root = Path(__file__).parent
    videos = []
    
    search_dirs = [
        project_root / "Towards Karjat-20260815T083400Z-1-001" / "Towards Karjat",
        project_root / "Towards Home-20260815T083359Z-1-001" / "Towards Home",
        project_root / "Towards Home",
        project_root / "Towards Karjat"
    ]
    
    for sdir in search_dirs:
        if sdir.exists():
            for f in sdir.glob("*.mp4"):
                if f.stat().st_size > 10_000_000:
                    videos.append({
                        "name": f.name,
                        "path": str(f.resolve()),
                        "size_mb": round(f.stat().st_size / (1024 * 1024), 1)
                    })
    
    available_videos = videos[:20]
    return available_videos

def open_video_source(mode, path_or_index=0):
    """
    Robust VideoCapture opener with fallback mechanisms.
    For live camera on Windows, tries DirectShow (CAP_DSHOW), MSMF, and standard default.
    """
    global video_capture
    if video_capture is not None:
        video_capture.release()
        video_capture = None
        time.sleep(0.05)

    if mode == "live":
        cam_idx = int(path_or_index)
        print(f"Attempting to open Live USB Camera index {cam_idx}...")
        
        # Try DirectShow first on Windows
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            # Fallback to default
            cap = cv2.VideoCapture(cam_idx)
            
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            print(f"Successfully opened Live Camera on index {cam_idx}")
            video_capture = cap
            return True
        else:
            print(f"Warning: Could not open Live Camera on index {cam_idx}")
            video_capture = None
            return False
    else:
        vpath = str(path_or_index)
        print(f"Loading Prerecorded Video: {vpath}")
        cap = cv2.VideoCapture(vpath)
        if cap.isOpened():
            video_capture = cap
            return True
        else:
            print(f"Failed to open video file: {vpath}")
            video_capture = None
            return False

def stream_worker():
    """Continuous perception background thread."""
    global video_capture, is_streaming, active_telemetry, pipeline, is_recording, record_writer
    print("Perception Video Worker started.")
    
    blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(blank_frame, "CONNECTING TO CAMERA SOURCE...", (320, 360), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
    
    while is_streaming:
        if video_capture is None or not video_capture.isOpened():
            with pipeline_lock:
                pipeline.latest_frame = blank_frame
            time.sleep(0.1)
            continue

        ret, frame = video_capture.read()
        if not ret:
            if stream_mode == "video":
                # Seamless loop for recorded video
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.01)
                continue
            else:
                time.sleep(0.03)
                continue

        # Execute through DREIVE Pipeline
        output_frame, detections, duration = pipeline.process_frame(frame)
        
        # Calibration observation accumulator
        if calibrator.is_calibrating and detections:
            for d in detections:
                if d.get("distance_m") and d.get("box"):
                    x1, _, x2, _ = d["box"]
                    cx = (x1 + x2) / 2.0
                    norm_x = (cx - (1280 / 2.0)) / (1280 / 2.0)
                    ang_rel = math.degrees(math.atan(norm_x * math.tan(math.radians(82.1 / 2.0))))
                    ang_lidar = (ang_rel + 360.0) % 360.0
                    calibrator.add_observation(cx, ang_lidar, d["distance_m"])

        # Recording
        if is_recording and record_writer is not None:
            record_writer.write(output_frame)

        # Telemetry
        telem = pipeline.emulator.get_hardware_telemetry()
        telem["total_detections"] = len(detections)
        telem["active_objects"] = [
            {"id": d.get("track_id"), "class": d["class_name"], "dist": d["distance_m"], "danger": d["danger_level"]}
            for d in detections
        ]
        telem["source_mode"] = stream_mode
        telem["view_mode"] = pipeline.view_mode
        telem["current_model"] = Path(pipeline.det_model_path).name
        telem["lidar_connected"] = bool(pipeline.lidar_driver and pipeline.lidar_driver.scanning)
        
        with pipeline_lock:
            pipeline.latest_frame = output_frame
            active_telemetry = telem

def generate_mjpeg():
    global pipeline
    while is_streaming:
        if pipeline is None or pipeline.latest_frame is None:
            time.sleep(0.02)
            continue

        with pipeline_lock:
            frame = pipeline.latest_frame

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        time.sleep(0.025)

@app.route('/')
def index():
    return render_template("dashboard.html")

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/telemetry')
def get_telemetry():
    with pipeline_lock:
        return jsonify(active_telemetry)

@app.route('/api/videos')
def get_videos():
    return jsonify({
        "videos": available_videos,
        "current_video": current_video_file
    })

@app.route('/api/ports')
def get_serial_ports():
    ports = scan_available_ports()
    detected = autodetect_rplidar_port()
    return jsonify({
        "ports": ports,
        "recommended_port": detected
    })

@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    if request.method == 'POST':
        new_cfg = request.json or {}
        save_rig_config(new_cfg)
        # Re-apply to active projector
        pipeline.projector.__init__(new_cfg)
        return jsonify({"status": "success", "config": new_cfg})
    else:
        return jsonify(load_rig_config())

@app.route('/api/control', methods=['POST'])
def update_control():
    global stream_mode, current_video_file, is_recording, record_writer, record_filename
    data = request.json or {}
    
    # 1. View Mode
    if "view_mode" in data:
        pipeline.view_mode = str(data["view_mode"])

    # 2. Display Layers
    if "show_tracers" in data:
        pipeline.show_tracers = bool(data["show_tracers"])
    if "show_segmentation" in data:
        pipeline.show_segmentation = bool(data["show_segmentation"])
    if "show_boxes" in data:
        pipeline.show_boxes = bool(data["show_boxes"])
    if "show_lidar_points" in data:
        pipeline.show_lidar_points = bool(data["show_lidar_points"])
    if "invert_camera" in data:
        pipeline.invert_camera = bool(data["invert_camera"])

    # 3. Model Switcher
    if "model_name" in data:
        mname = data["model_name"]
        model_map = {
            "yolo11s": "runs/detect/runs/detect/indian_driving_yolo11s/weights/best.pt",
            "yolo26n": "models_inventory/yolo26n_best.pt",
            "yolov10n": "runs/detect/runs/detect/indian_driving_v1/weights/best.pt"
        }
        target_path = model_map.get(mname, mname)
        if Path(target_path).exists():
            pipeline.load_models(target_path)
            return jsonify({"status": "success", "loaded_model": mname})
        else:
            return jsonify({"status": "error", "message": f"Model weight not found: {target_path}"})

    # 4. Engine Optimizations
    if "emulate_orin" in data:
        pipeline.emulator.emulate_throttling = bool(data["emulate_orin"])
        pipeline.emulate_orin = bool(data["emulate_orin"])
    if "alternate_frames" in data:
        pipeline.alternate_frames = bool(data["alternate_frames"])
    if "conf_threshold" in data:
        pipeline.conf_threshold = float(data["conf_threshold"])

    # 5. Input Source Switching (Live Camera vs Prerecorded Video)
    if "source" in data:
        req_source = data["source"]
        if req_source == "live":
            cam_idx = int(data.get("camera_index", 0))
            success = open_video_source("live", cam_idx)
            if success:
                stream_mode = "live"
                return jsonify({"status": "success", "source": "live", "camera_index": cam_idx})
            else:
                return jsonify({"status": "error", "message": f"Could not connect to USB camera at index {cam_idx}. Check USB plug."})
        elif req_source == "video":
            vpath = data.get("video_path") or current_video_file
            if not vpath and available_videos:
                vpath = available_videos[0]["path"]
            success = open_video_source("video", vpath)
            if success:
                stream_mode = "video"
                current_video_file = vpath
                return jsonify({"status": "success", "source": "video", "video_path": vpath})
            else:
                return jsonify({"status": "error", "message": f"Failed to load video file: {vpath}"})

    # 6. Physical LiDAR Connection
    if data.get("connect_lidar"):
        port = data.get("port")
        res = pipeline.connect_lidar(port)
        return jsonify({"status": "success", "lidar_connected": res})

    # 7. Snapshots & Recordings
    if data.get("take_snapshot"):
        out_dir = Path(__file__).parent / "metrics_and_results" / "snapshots"
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = f"snapshot_{int(time.time())}.jpg"
        with pipeline_lock:
            if pipeline.latest_frame is not None:
                cv2.imwrite(str(out_dir / filename), pipeline.latest_frame)
                return jsonify({"status": "success", "file": filename})

    if "recording" in data:
        start_rec = bool(data["recording"])
        if start_rec and not is_recording:
            out_dir = Path(__file__).parent / "metrics_and_results" / "recordings"
            out_dir.mkdir(parents=True, exist_ok=True)
            record_filename = f"session_{int(time.time())}.mp4"
            record_writer = cv2.VideoWriter(str(out_dir / record_filename), cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (1280, 720))
            is_recording = True
            return jsonify({"status": "success", "recording": True, "file": record_filename})
        elif not start_rec and is_recording:
            is_recording = False
            if record_writer:
                record_writer.release()
                record_writer = None
            return jsonify({"status": "success", "recording": False})

    # 8. Auto Extrinsic Calibration
    if data.get("start_auto_calibration"):
        calibrator.observations.clear()
        calibrator.is_calibrating = True
        return jsonify({"status": "started", "message": "Collecting target correspondence vectors..."})

    if data.get("solve_auto_calibration"):
        calibrator.is_calibrating = False
        res = calibrator.run_auto_calibration()
        if res.get("success"):
            pipeline.projector.update_extrinsics(
                delta_x_m=res["delta_x_m"],
                delta_y_m=res["delta_y_m"],
                yaw_trim_deg=res["yaw_trim_deg"]
            )
        return jsonify(res)

    return jsonify({"status": "success"})

def init_app():
    global pipeline, current_video_file, is_streaming
    print("Scanning available video sequences...")
    scan_available_videos()
    
    print("Initializing DREIVE Pipeline...")
    pipeline = FusionPipeline(emulate_orin=True)
    
    if available_videos:
        current_video_file = available_videos[0]["path"]
        open_video_source("video", current_video_file)
    else:
        open_video_source("live", 0)
        
    is_streaming = True
    worker = threading.Thread(target=stream_worker, daemon=True)
    worker.start()

if __name__ == '__main__':
    init_app()
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
