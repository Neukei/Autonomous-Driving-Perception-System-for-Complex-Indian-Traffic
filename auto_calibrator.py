"""
Targetless Automatic Extrinsic Calibration Engine for Camera (Logitech C930e) and 2D LiDAR (Slamtec A1).
Uses Ground-Object Correspondence (Bounding Box Centroids <-> LiDAR Ray Clusters)
to solve for optimal:
  - yaw_trim_deg (rotational heading misalignment)
  - delta_y_m (forward longitudinal offset)
  - delta_x_m (lateral mounting offset)
without requiring manual checkerboards.
"""
import math
import numpy as np
from scipy.optimize import minimize
from sensor_rig_config import load_rig_config, save_rig_config

class AutoExtrinsicCalibrator:
    def __init__(self, fov_h_deg=82.1, img_w=1280, img_h=720):
        self.fov_h = math.radians(fov_h_deg)
        self.w = img_w
        self.h = img_h
        self.fx = (self.w / 2.0) / math.tan(self.fov_h / 2.0)
        self.cx = self.w / 2.0
        
        # Calibration observation buffer: list of pairs (detected_box, lidar_cluster)
        self.observations = []
        self.is_calibrating = False

    def add_observation(self, box_center_x, lidar_angle_deg, lidar_dist_m):
        """
        Record a synchronized observation:
          - box_center_x: optical column of vehicle/target center (px)
          - lidar_angle_deg: physical azimuth of closest echo cluster (deg)
          - lidar_dist_m: measured radial distance (meters)
        """
        if lidar_dist_m < 0.5 or lidar_dist_m > 10.0:
            return
        
        self.observations.append({
            "px_target": box_center_x,
            "angle_deg": lidar_angle_deg,
            "dist_m": lidar_dist_m
        })

    def run_auto_calibration(self):
        """
        Runs non-linear least squares optimization (L-BFGS-B / Nelder-Mead)
        minimizing the re-projection error between camera pixel columns and LiDAR rays.
        """
        if len(self.observations) < 8:
            return {
                "success": False,
                "message": f"Need at least 8 valid target observations (currently have {len(self.observations)}). Keep driving or observing targets."
            }

        # Initial parameters: [yaw_trim_deg, delta_y_m, delta_x_m]
        initial_params = [0.0, 0.15, 0.0]
        
        # Loss function: sum of squared pixel projection errors
        def objective(params):
            yaw_trim_deg, dy, dx = params
            yaw_trim = math.radians(yaw_trim_deg)
            total_error = 0.0

            for obs in self.observations:
                ang_norm = obs["angle_deg"] % 360.0
                rel_ang = math.radians(ang_norm - 360.0) if ang_norm > 180.0 else math.radians(ang_norm)
                rel_ang += yaw_trim

                # Forward Z and Lateral X in Camera Frame
                z_cam = obs["dist_m"] * math.cos(rel_ang) + dy
                x_cam = obs["dist_m"] * math.sin(rel_ang) + dx

                if z_cam <= 0.1:
                    total_error += 10000.0
                    continue

                # Projected pixel column
                px_proj = self.cx + (x_cam / z_cam) * self.fx
                diff = px_proj - obs["px_target"]
                total_error += diff ** 2

            return total_error / len(self.observations)

        # Optimization bounds:
        # yaw_trim: -15° to +15°
        # delta_y: 0.0m to 0.40m
        # delta_x: -0.15m to +0.15m
        bounds = [(-15.0, 15.0), (0.0, 0.40), (-0.15, 0.15)]
        
        res = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds)

        if res.success:
            opt_yaw, opt_dy, opt_dx = res.x
            rms_pixel_error = math.sqrt(res.fun)

            # Update configuration file
            config = load_rig_config()
            config["extrinsics"]["yaw_trim_deg"] = round(float(opt_yaw), 2)
            config["extrinsics"]["delta_y_m"] = round(float(opt_dy), 3)
            config["extrinsics"]["delta_x_m"] = round(float(opt_dx), 3)
            save_rig_config(config)

            return {
                "success": True,
                "yaw_trim_deg": round(float(opt_yaw), 2),
                "delta_y_m": round(float(opt_dy), 3),
                "delta_x_m": round(float(opt_dx), 3),
                "rms_pixel_error": round(rms_pixel_error, 1),
                "sample_count": len(self.observations),
                "message": f"Auto-calibration successful! Residual reprojection error: ±{rms_pixel_error:.1f}px"
            }
        else:
            return {
                "success": False,
                "message": "Optimization did not converge cleanly. Try gathering more targets."
            }
