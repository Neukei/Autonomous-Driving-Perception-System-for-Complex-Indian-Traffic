"""
LiDAR to Camera 3D-to-2D Coordinate Extrinsic & Intrinsic Fusion Engine.
Calibrated for:
  - Camera: Logitech C930e (90° dFOV, 82.1° hFOV, 52.2° vFOV)
  - LiDAR: Slamtec RPLiDAR A1M8 (360°, 12m)
  - Extrinsic Translation: [dX, dY, dZ] + Rotational Yaw/Pitch Trims
"""
import math
import numpy as np
from sensor_rig_config import load_rig_config, save_rig_config

class LidarCameraProjector:
    def __init__(self, config=None, camera_fov_h_deg=None):
        if config is None:
            config = load_rig_config()
            
        self.config = config
        cam = config.get("camera", {})
        ext = config.get("extrinsics", {})

        # Intrinsic parameters (Logitech C930e default 82.1°, or overridden)
        fov_h_deg = camera_fov_h_deg if camera_fov_h_deg is not None else cam.get("fov_horizontal_deg", 82.1)
        self.fov_h = math.radians(fov_h_deg)
        self.fov_v = math.radians(cam.get("fov_vertical_deg", 52.2))
        self.w = cam.get("resolution_width", 1280)
        self.h = cam.get("resolution_height", 720)

        # Extrinsic 3D Spatial Offsets between Camera and LiDAR:
        # dX: lateral horizontal displacement (meters, positive = LiDAR to the right of camera)
        # dY: longitudinal forward displacement (meters, positive = LiDAR forward of camera)
        # dZ: vertical height displacement (meters, positive = LiDAR higher than camera)
        self.dx = ext.get("delta_x_m", 0.0)
        self.dy = ext.get("delta_y_m", 0.15)
        self.dz = ext.get("delta_z_m", 0.25)
        self.yaw_trim = math.radians(ext.get("yaw_trim_deg", 0.0))
        self.pitch_trim = math.radians(ext.get("pitch_trim_deg", 0.0))

        # Pinhole Camera Matrix parameters
        self.fx = (self.w / 2.0) / math.tan(self.fov_h / 2.0)
        self.fy = (self.h / 2.0) / math.tan(self.fov_v / 2.0)
        self.cx = self.w / 2.0
        self.cy = self.h / 2.0

    def update_extrinsics(self, delta_x_m=None, delta_y_m=None, delta_z_m=None, yaw_trim_deg=None):
        """Allows live calibration tuning from the dashboard."""
        ext = self.config.get("extrinsics", {})
        if delta_x_m is not None:
            self.dx = float(delta_x_m)
            ext["delta_x_m"] = self.dx
        if delta_y_m is not None:
            self.dy = float(delta_y_m)
            ext["delta_y_m"] = self.dy
        if delta_z_m is not None:
            self.dz = float(delta_z_m)
            ext["delta_z_m"] = self.dz
        if yaw_trim_deg is not None:
            self.yaw_trim = math.radians(float(yaw_trim_deg))
            ext["yaw_trim_deg"] = float(yaw_trim_deg)
        save_rig_config(self.config)

    def lidar_to_camera_pixel(self, angle_deg, distance_m):
        """
        Projects a physical measurement from Slamtec RPLiDAR A1 into
        Logitech C930e pixel coordinates, applying rigorous 3D translation:
          X_cam = X_lidar + dX
          Y_cam = -dZ (elevation differential)
          Z_cam = Z_lidar + dY (forward offset compensation)
        """
        # Convert LiDAR polar coordinates to forward-centered relative radians [-pi, pi]
        norm_deg = angle_deg % 360.0
        if norm_deg > 180.0:
            rel_angle = math.radians(norm_deg - 360.0)
        else:
            rel_angle = math.radians(norm_deg)
            
        rel_angle += self.yaw_trim

        if distance_m <= 0.10 or distance_m > 12.0:
            return None

        # 1. 3D Coordinates in LiDAR reference frame:
        # Forward = Z_l, Lateral Right = X_l, Height Up = Y_l
        Z_lidar = distance_m * math.cos(rel_angle)
        X_lidar = distance_m * math.sin(rel_angle)

        # 2. Rigid Extrinsic 3D Transformation to Camera reference frame:
        # Applying spatial offsets dX, dY, dZ
        Z_cam = Z_lidar + self.dy
        X_cam = X_lidar + self.dx
        Y_cam = -self.dz  # LiDAR is higher, so in camera frame (Y pointing down), offset is negative

        # Ignore targets behind camera optical plane
        if Z_cam <= 0.2:
            return None

        # 3. Pinhole Projection Matrix (Logitech C930e):
        px = int(self.cx + (X_cam / Z_cam) * self.fx)
        py = int(self.cy + (Y_cam / Z_cam) * self.fy)

        # Bounds check within image frame
        if 0 <= px < self.w:
            # True euclidean distance from vehicle camera origin to object
            true_dist_cam = math.sqrt(X_cam**2 + Y_cam**2 + Z_cam**2)
            return px, true_dist_cam, py
            
        return None

    def associate_lidar_with_detections(self, lidar_points, detections, masks=None):
        """Associates LiDAR points with bounding boxes using extrinsic spatial correction."""
        obj_distances = [[] for _ in detections]
        projected_points = []

        for angle, dist in lidar_points:
            proj = self.lidar_to_camera_pixel(angle, dist)
            if proj is None:
                continue
            px, z_dist, py = proj
            projected_points.append((px, z_dist, py))

            # Match with bounding boxes
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = det['box']
                if x1 <= px <= x2:
                    obj_distances[i].append(z_dist)

        # Compute robust median distance for each target
        for i, det in enumerate(detections):
            dists = obj_distances[i]
            if dists:
                dists.sort()
                k = max(1, len(dists) // 2)
                est_dist = float(np.median(dists[:k]))
                det['distance_m'] = round(est_dist, 2)
                det['lidar_points_count'] = len(dists)
                
                if est_dist < 3.0:
                    det['danger_level'] = 'CRITICAL'
                elif est_dist < 6.5:
                    det['danger_level'] = 'WARNING'
                else:
                    det['danger_level'] = 'SAFE'
            else:
                det['distance_m'] = None
                det['danger_level'] = 'UNKNOWN'
                det['lidar_points_count'] = 0

        return detections, projected_points
