"""
Core Perception & Sensor Fusion Pipeline (Standard Clean Engineering Architecture).
Supports:
  - Dynamic Model Swapping (YOLO11s, YOLO26n, YOLOv10n, RT-DETR)
  - Seamless Live USB Camera vs Prerecorded Video Replay
  - Configurable Dashcam Orientation (Normal / 180° Inverted)
  - 4 View Modes: Combined HUD, Pure Camera, 2D Polar LiDAR Radar, Split View
  - Dynamic Ground-to-Object Laser Distance Tracers
  - Real-time Extrinsic Tuning (dX, dY, dZ, yaw trim)
  - Orin Nano Hardware Throttling Emulator
"""
import cv2
import time
import math
import random
import threading
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from rplidar_driver import SimpleRPLidar, autodetect_rplidar_port, scan_available_ports
from lidar_projection import LidarCameraProjector
from orin_emulator import JetsonOrinNanoEmulator

# 11 Indian Driving Dataset classes
IDD_CLASSES = {
    0: 'car', 1: 'bus', 2: 'truck', 3: 'autorickshaw',
    4: 'motorcycle', 5: 'bicycle', 6: 'person', 7: 'rider',
    8: 'animal', 9: 'traffic sign', 10: 'traffic light'
}

CLASS_COLORS = {
    'car': (255, 178, 50),
    'bus': (0, 165, 255),
    'truck': (205, 90, 106),
    'autorickshaw': (0, 215, 255),
    'motorcycle': (0, 255, 127),
    'bicycle': (255, 255, 0),
    'person': (0, 69, 255),
    'rider': (147, 20, 255),
    'animal': (180, 105, 255),
    'traffic sign': (200, 200, 200),
    'traffic light': (50, 220, 220)
}

class FusionPipeline:
    def __init__(self, 
                 det_model_path="runs/detect/runs/detect/indian_driving_yolo11s/weights/best.pt",
                 seg_model_path="runs/segment/runs/segment/indian_segmentation_v1/weights/best.pt",
                 emulate_orin=True):
        
        self.emulate_orin = emulate_orin
        self.emulator = JetsonOrinNanoEmulator(emulate_throttling=emulate_orin)
        self.projector = LidarCameraProjector()
        
        # Pipeline State & Locks (Must be initialized before load_models)
        self.running = True
        self.paused = False
        self.lock = threading.Lock()
        
        # Models
        self.det_model_path = det_model_path
        self.seg_model_path = seg_model_path
        self.det_model = None
        self.seg_model = None
        self.load_models(det_model_path, seg_model_path)
        
        # View & Display Toggles
        self.view_mode = "combined"      # 'combined' | 'camera' | 'lidar_radar' | 'split'
        self.show_tracers = True
        self.show_segmentation = True
        self.show_boxes = True
        self.show_lidar_points = True
        
        # Live Buffers
        self.latest_lidar_points = []
        self.latest_frame = None
        self.latest_detections = []
        self.latest_road_mask = None
        self.cached_road_mask = None
        
        # Hardware & Optimization Controls
        self.frame_idx = 0
        self.alternate_frames = True
        self.enable_tracking = True
        self.conf_threshold = 0.35
        self.iou_threshold = 0.45
        self.invert_camera = True       # Dashcam inverted mount fix
        
        # LiDAR Driver
        self.lidar_port = None
        self.lidar_driver = None
        self.lidar_thread = None

    def load_models(self, det_path, seg_path=None):
        """Allows hot-swapping detection or segmentation weights on the fly."""
        with self.lock:
            try:
                print(f"Loading Detection Model: {det_path}")
                self.det_model = YOLO(det_path)
                self.det_model_path = det_path
            except Exception as e:
                print(f"Failed to load detection model {det_path}: {e}")

            if seg_path:
                try:
                    print(f"Loading Segmentation Model: {seg_path}")
                    self.seg_model = YOLO(seg_path)
                    self.seg_model_path = seg_path
                except Exception as e:
                    print(f"Failed to load segmentation model {seg_path}: {e}")

    def connect_lidar(self, port=None):
        """Connects to physical Slamtec RPLiDAR A1."""
        if port is None:
            port = autodetect_rplidar_port()
        if not port:
            print("No physical LiDAR found. Retaining simulated ray mode.")
            return False
            
        try:
            self.lidar_driver = SimpleRPLidar(port=port)
            if self.lidar_driver.connect():
                self.lidar_port = port
                self.lidar_thread = threading.Thread(target=self._lidar_worker, daemon=True)
                self.lidar_thread.start()
                return True
        except Exception as e:
            print(f"LiDAR Connection Error: {e}")
        return False

    def _lidar_worker(self):
        """Background scan loop for physical LiDAR."""
        scan_bucket = []
        last_flush = time.time()
        
        for quality, angle, dist_m in self.lidar_driver.iter_measurements():
            if not self.running:
                break
            scan_bucket.append((angle, dist_m))
            if time.time() - last_flush > 0.05:
                with self.lock:
                    self.latest_lidar_points = list(scan_bucket)
                scan_bucket = []
                last_flush = time.time()

    def generate_simulated_lidar(self, detections, img_w=1280):
        """Synthesizes high-fidelity 2D laser sweeps matching visible obstacles."""
        sim_points = []
        # Ambient 360-degree boundary returns
        for ang in range(0, 360, 3):
            rel = (ang + 180) % 360 - 180
            if abs(rel) < 42:
                sim_points.append((float(ang), random.uniform(9.0, 12.0)))
            elif abs(rel) > 135:
                sim_points.append((float(ang), random.uniform(1.2, 3.8)))
            else:
                sim_points.append((float(ang), random.uniform(3.5, 8.0)))
        
        # Ray clusters corresponding to detected bounding boxes
        for det in detections:
            x1, y1, x2, y2 = det['box']
            box_h = max(10, y2 - y1)
            dist_approx = max(1.2, min(12.0, (720.0 / box_h) * 1.5))
            
            cx = (x1 + x2) / 2.0
            norm_x = (cx - (img_w / 2.0)) / (img_w / 2.0)
            angle_rel = math.degrees(math.atan(norm_x * math.tan(math.radians(41.0))))
            angle_lidar = (angle_rel + 360.0) % 360.0
            
            for da in [-1.5, -0.7, 0.0, 0.7, 1.5]:
                sim_points.append(((angle_lidar + da) % 360.0, dist_approx + random.uniform(-0.06, 0.06)))
                
        return sim_points

    def process_frame(self, frame):
        """Main inference and sensor fusion pass on incoming video frame."""
        t_start = time.time()
        
        # Apply orientation correction if mounted upside down
        if self.invert_camera:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            
        h, w = frame.shape[:2]
        self.frame_idx += 1
        
        run_det = True
        run_seg = (not self.alternate_frames) or (self.frame_idx % 2 == 1) or (self.cached_road_mask is None)

        detections = []
        road_mask = self.cached_road_mask

        # --- A. ROAD & DRIVABLE SEGMENTATION ---
        if run_seg and self.seg_model is not None:
            try:
                seg_results = self.seg_model(frame, verbose=False, conf=0.3, imgsz=480, device=0)
                if seg_results and len(seg_results) > 0 and seg_results[0].masks is not None:
                    masks = seg_results[0].masks.data.cpu().numpy()
                    classes = seg_results[0].boxes.cls.cpu().numpy()
                    
                    road_indices = np.where(classes == 0)[0]
                    if len(road_indices) > 0:
                        combined_mask = np.zeros((masks.shape[1], masks.shape[2]), dtype=np.uint8)
                        for idx in road_indices:
                            combined_mask = np.maximum(combined_mask, (masks[idx] * 255).astype(np.uint8))
                        
                        road_mask = cv2.resize(combined_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        self.cached_road_mask = road_mask
            except Exception as e:
                pass

        # --- B. OBJECT DETECTION & BYTETRACK ---
        if run_det and self.det_model is not None:
            try:
                if self.enable_tracking:
                    det_results = self.det_model.track(
                        frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        conf=self.conf_threshold,
                        iou=self.iou_threshold,
                        device=0,
                        verbose=False
                    )
                else:
                    det_results = self.det_model(frame, conf=self.conf_threshold, iou=self.iou_threshold, device=0, verbose=False)

                if det_results and len(det_results) > 0:
                    res = det_results[0]
                    names_dict = res.names if hasattr(res, 'names') and res.names else IDD_CLASSES
                    if res.boxes is not None:
                        for box in res.boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            xyxy = [int(v) for v in box.xyxy[0]]
                            track_id = int(box.id[0]) if (box.id is not None) else None
                            
                            class_name = names_dict.get(cls_id, f"obj_{cls_id}")
                            detections.append({
                                "box": xyxy,
                                "cls_id": cls_id,
                                "class_name": class_name,
                                "conf": conf,
                                "track_id": track_id
                            })
            except Exception as e:
                pass

        # --- C. 2D LIDAR CORRESPONDENCE & EXTRINSIC RANGING ---
        with self.lock:
            lidar_pts = list(self.latest_lidar_points)
        
        if not lidar_pts:
            lidar_pts = self.generate_simulated_lidar(detections, img_w=w)
            
        fused_detections, proj_pts = self.projector.associate_lidar_with_detections(lidar_pts, detections)

        # --- D. ORIN NANO LATENCY INJECTION ---
        duration = self.emulator.profile_and_throttle(t_start)

        # --- E. MULTI-VIEW RENDERING ---
        output_frame = self._render_view(frame, fused_detections, road_mask, proj_pts, lidar_pts)
        
        return output_frame, fused_detections, duration

    def _render_view(self, frame, detections, road_mask, proj_pts, lidar_pts):
        if self.view_mode == "camera":
            return self._render_camera_only(frame, detections)
        elif self.view_mode == "lidar_radar":
            return self._render_lidar_radar_fullscreen(frame.shape[1], frame.shape[0], lidar_pts, detections)
        elif self.view_mode == "split":
            return self._render_split_view(frame, detections, road_mask, proj_pts, lidar_pts)
        else:
            return self._render_combined_view(frame, detections, road_mask, proj_pts)

    def _render_camera_only(self, frame, detections):
        overlay = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cls_name = det["class_name"]
            track_id = det["track_id"]
            color = CLASS_COLORS.get(cls_name, (0, 255, 120))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            tag = f"#{track_id} {cls_name.capitalize()}" if track_id else cls_name.capitalize()
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            cv2.rectangle(overlay, (x1, max(0, y1 - 22)), (x1 + tw + 10, y1), color, -1)
            cv2.putText(overlay, tag, (x1 + 5, max(16, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
        
        cv2.putText(overlay, "OPTICAL FEED // CLEAN VISION", (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2)
        return overlay

    def _render_combined_view(self, frame, detections, road_mask, proj_pts):
        overlay = frame.copy()
        h, w = frame.shape[:2]
        origin_x = w // 2
        origin_y = int(h * 0.98)

        # 1. Road Surface Carpet
        if self.show_segmentation and road_mask is not None:
            road_color = np.zeros_like(frame, dtype=np.uint8)
            road_color[:, :] = (0, 200, 100)
            road_overlay = cv2.bitwise_and(road_color, road_color, mask=road_mask)
            cv2.addWeighted(road_overlay, 0.28, overlay, 1.0, 0, overlay)

        # 2. LiDAR Returns
        if self.show_lidar_points:
            for px, z, py in proj_pts:
                cv2.circle(overlay, (px, int(h * 0.78)), 2, (0, 255, 255), -1)

        # 3. Targets & Tracer Lines
        critical_alerts = []
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cls_name = det["class_name"]
            track_id = det["track_id"]
            dist_m = det["distance_m"]
            danger = det["danger_level"]

            target_cx = (x1 + x2) // 2
            target_cy = y2

            if danger == 'CRITICAL':
                box_color = (0, 0, 255)
                line_color = (0, 0, 255)
                critical_alerts.append(f"{cls_name.upper()} AT {dist_m}m")
            elif danger == 'WARNING':
                box_color = (0, 190, 255)
                line_color = (0, 215, 255)
            else:
                box_color = CLASS_COLORS.get(cls_name, (0, 255, 120))
                line_color = (0, 255, 120)

            # Laser Distance Tracer
            if self.show_tracers and dist_m is not None:
                thickness = 2 if danger != 'CRITICAL' else 3
                cv2.line(overlay, (origin_x, origin_y), (target_cx, target_cy), line_color, thickness, cv2.LINE_AA)
                cv2.circle(overlay, (target_cx, target_cy), 4, line_color, -1)
                
                mid_x = (origin_x + target_cx) // 2
                mid_y = (origin_y + target_cy) // 2
                dist_tag = f"{dist_m}m"
                (mw, mh), _ = cv2.getTextSize(dist_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(overlay, (mid_x - 4, mid_y - mh - 4), (mid_x + mw + 4, mid_y + 4), (10, 15, 25), -1)
                cv2.rectangle(overlay, (mid_x - 4, mid_y - mh - 4), (mid_x + mw + 4, mid_y + 4), line_color, 1)
                cv2.putText(overlay, dist_tag, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # Box
            if self.show_boxes:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, 2)
                tag_parts = []
                if track_id is not None:
                    tag_parts.append(f"#{track_id}")
                tag_parts.append(cls_name.capitalize())
                if dist_m is not None:
                    tag_parts.append(f"{dist_m}m")

                tag_text = " | ".join(tag_parts)
                (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
                cv2.rectangle(overlay, (x1, max(0, y1 - 22)), (x1 + tw + 10, y1), box_color, -1)
                cv2.putText(overlay, tag_text, (x1 + 5, max(16, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255) if danger != 'WARNING' else (0, 0, 0), 2)

        if self.show_tracers:
            cv2.circle(overlay, (origin_x, origin_y), 6, (0, 220, 255), -1)
            cv2.circle(overlay, (origin_x, origin_y), 12, (0, 220, 255), 1)

        if critical_alerts:
            cv2.rectangle(overlay, (0, 0), (w, 42), (0, 0, 220), -1)
            msg = f"COLLISION ALERT // EMERGENCY BRAKE: {', '.join(critical_alerts[:2])}"
            cv2.putText(overlay, msg, (20, 29), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

        return overlay

    def _render_lidar_radar_fullscreen(self, w, h, lidar_pts, detections):
        radar = np.zeros((h, w, 3), dtype=np.uint8)
        radar[:, :] = (12, 17, 26)

        cx = w // 2
        cy = int(h * 0.75)
        max_range_m = 12.0
        scale = (min(w, h) * 0.65) / max_range_m

        for r_m in [3.0, 6.0, 9.0, 12.0]:
            pix_r = int(r_m * scale)
            color = (0, 0, 180) if r_m == 3.0 else (45, 60, 80)
            cv2.circle(radar, (cx, cy), pix_r, color, 1)
            cv2.putText(radar, f"{int(r_m)}m", (cx + 5, cy - pix_r + 14), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 140, 170), 1)

        half_fov = math.radians(82.1 / 2.0)
        cone_len = int(12.0 * scale)
        p1 = (int(cx - cone_len * math.sin(half_fov)), int(cy - cone_len * math.cos(half_fov)))
        p2 = (int(cx + cone_len * math.sin(half_fov)), int(cy - cone_len * math.cos(half_fov)))
        cv2.line(radar, (cx, cy), p1, (0, 180, 220), 1, cv2.LINE_AA)
        cv2.line(radar, (cx, cy), p2, (0, 180, 220), 1, cv2.LINE_AA)

        for angle_deg, dist_m in lidar_pts:
            if dist_m <= 0.1 or dist_m > 12.0:
                continue
            ang_rad = math.radians(angle_deg)
            px = int(cx + (dist_m * scale) * math.sin(ang_rad))
            py = int(cy - (dist_m * scale) * math.cos(ang_rad))
            
            pt_color = (0, 0, 255) if dist_m < 3.0 else ((0, 215, 255) if dist_m < 6.5 else (0, 255, 120))
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(radar, (px, py), 2, pt_color, -1)

        for det in detections:
            dist_m = det.get('distance_m')
            if dist_m is not None and dist_m > 0:
                x1, _, x2, _ = det['box']
                target_cx = (x1 + x2) / 2.0
                norm_x = (target_cx - (w / 2.0)) / (w / 2.0)
                angle_rel = math.atan(norm_x * math.tan(half_fov))
                
                ox = int(cx + (dist_m * scale) * math.sin(angle_rel))
                oy = int(cy - (dist_m * scale) * math.cos(angle_rel))
                
                cv2.circle(radar, (ox, oy), 7, (0, 0, 255) if dist_m < 3.0 else (0, 220, 255), 2)
                cls_txt = f"{det['class_name'][:4].upper()} {dist_m}m"
                cv2.putText(radar, cls_txt, (ox + 10, oy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.rectangle(radar, (cx - 14, cy - 20), (cx + 14, cy + 20), (0, 220, 255), 2)
        cv2.arrowedLine(radar, (cx, cy + 10), (cx, cy - 12), (0, 220, 255), 2, tipLength=0.4)
        cv2.putText(radar, "SLAMTEC A1M8 // 360 DEGREE ACTIVE RADAR", (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2)
        return radar

    def _render_split_view(self, frame, detections, road_mask, proj_pts, lidar_pts):
        h, w = frame.shape[:2]
        half_w = w // 2

        cam_view = self._render_combined_view(frame, detections, road_mask, proj_pts)
        left_panel = cv2.resize(cam_view, (half_w, h))
        right_panel = self._render_lidar_radar_fullscreen(half_w, h, lidar_pts, detections)

        split_frame = np.hstack([left_panel, right_panel])
        cv2.line(split_frame, (half_w, 0), (half_w, h), (0, 220, 255), 2)
        return split_frame
