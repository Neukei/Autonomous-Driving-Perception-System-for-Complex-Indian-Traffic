import cv2
import time
import math
import threading
import logging
import numpy as np
from ultralytics import YOLO

# Try to import RPLiDAR, fallback if not installed on dev machine
try:
    from rplidar import RPLidar
except ImportError:
    RPLidar = None
    print("Warning: rplidar package not found. Run 'pip install rplidar-roboticia'")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class SensorFusionEngine:
    def __init__(self, model_path, lidar_port='/dev/ttyUSB0', camera_index=0):
        # AI Model
        self.model = YOLO(model_path)
        
        # Hardware params
        self.lidar_port = lidar_port
        self.camera_index = camera_index
        
        # State variables (Thread safe)
        self.latest_frame = None
        self.detections = []
        self.front_distance_mm = float('inf')
        
        self.running = False
        self.lock = threading.Lock()

        # Emergency Thresholds
        self.DANGER_ZONES = [0, 1, 2, 3, 4, 6, 8]  # Cars, Trucks, Rickshaws, Bikes, People, Animals
        self.CRITICAL_DISTANCE_MM = 3000           # 3 meters (Trigger Emergency Brake)
        self.WARNING_DISTANCE_MM = 7000            # 7 meters (Dashboard Warning)

    def _lidar_worker(self):
        """Background thread reading 360-degree scans from RPLiDAR A1"""
        if RPLidar is None:
            logging.error("LiDAR disabled (package missing).")
            return
            
        try:
            lidar = RPLidar(self.lidar_port)
            logging.info("LiDAR connected successfully.")
            
            for scan in lidar.iter_scans():
                if not self.running:
                    break
                    
                min_dist = float('inf')
                # scan format: (quality, angle, distance_mm)
                for (_, angle, distance) in scan:
                    # Look at a 30-degree cone directly in front of the car
                    # Assuming 0 degrees is dead ahead
                    if (angle < 15 or angle > 345) and distance > 0:
                        if distance < min_dist:
                            min_dist = distance
                            
                with self.lock:
                    self.front_distance_mm = min_dist
                    
        except Exception as e:
            logging.error(f"LiDAR Error: {e}")
            
    def _camera_worker(self):
        """Background thread reading webcam and running YOLOv10n inference"""
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Run inference on Jetson GPU
            results = self.model(frame, verbose=False, device=0)
            
            current_detections = []
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                # Normalized coordinates
                cx, cy, w, h = box.xywhn[0]
                
                # Check if object is in the center column of the frame (in our path)
                in_path = (cx > 0.35 and cx < 0.65)
                
                current_detections.append({
                    'class': cls_id,
                    'confidence': conf,
                    'in_path': in_path,
                    'size': w * h  # Area proxy
                })
                
            with self.lock:
                self.latest_frame = results[0].plot() # Image with drawn boxes
                self.detections = current_detections

        cap.release()

    def start(self):
        self.running = True
        self.lidar_thread = threading.Thread(target=self._lidar_worker, daemon=True)
        self.camera_thread = threading.Thread(target=self._camera_worker, daemon=True)
        
        self.lidar_thread.start()
        self.camera_thread.start()
        
        self._fusion_loop()

    def stop(self):
        self.running = False
        self.lidar_thread.join()
        self.camera_thread.join()

    def _fusion_loop(self):
        """Main loop that evaluates data from both sensors to make driving decisions"""
        logging.info("Sensor Fusion Engine started...")
        
        while self.running:
            with self.lock:
                dist = self.front_distance_mm
                dets = self.detections
                frame = self.latest_frame
                
            if frame is None:
                time.sleep(0.01)
                continue

            # Default state
            decision = "🟢 SAFE TO DRIVE"
            color = (0, 255, 0)
            
            # Evaluate Threat Logic
            threat_in_path = any((d['class'] in self.DANGER_ZONES and d['in_path']) for d in dets)
            
            if threat_in_path:
                if dist < self.CRITICAL_DISTANCE_MM:
                    decision = "🚨 EMERGENCY BRAKE ENGAGED"
                    color = (0, 0, 255)
                    # Trigger physical brake relay via GPIO here
                elif dist < self.WARNING_DISTANCE_MM:
                    decision = "⚠️ WARNING: TAILGATING"
                    color = (0, 165, 255)

            # Display UI Dashboard
            cv2.putText(frame, f"LiDAR: {dist/1000:.2f}m", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Status: {decision}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            
            cv2.imshow("Autonomous Perception Dashboard", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break

if __name__ == "__main__":
    # Pointing to the newly trained model weights
    engine = SensorFusionEngine(
        model_path="runs/detect/indian_driving_v1/weights/best.pt",
        lidar_port="COM3",  # Change to /dev/ttyUSB0 on Jetson
        camera_index=0
    )
    engine.start()
