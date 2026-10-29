"""
Calibration Configuration for DREIVE Sensor Rig:
  - Camera: Logitech C930e (90° diagonal, 82.1° horizontal FOV)
  - LiDAR: Slamtec RPLiDAR A1 / A1M8 (360°, 12m range)
  - Extrinsic Calibration Offsets (measured relative to camera optical center)
"""
import json
import math
from pathlib import Path

CONFIG_FILE = Path(__file__).parent / "sensor_rig_config.json"

DEFAULT_RIG_CONFIG = {
    "camera": {
        "model": "Logitech C930e",
        "fov_diagonal_deg": 90.0,
        "fov_horizontal_deg": 82.1,   # Exact C930e 16:9 horizontal FOV
        "fov_vertical_deg": 52.2,     # Exact C930e 16:9 vertical FOV
        "resolution_width": 1280,
        "resolution_height": 720,
        "fps": 30,
        "mount_location": "Inside cabin on rearview mirror",
        "elevation_ground_m": 1.25
    },
    "lidar": {
        "model": "Slamtec RPLiDAR A1M8",
        "range_min_m": 0.15,
        "range_max_m": 12.0,
        "sample_rate_hz": 8000,
        "scan_frequency_hz": 7.0,
        "mount_location": "Vehicle roof above windshield",
        "elevation_ground_m": 1.50
    },
    "extrinsics": {
        "delta_x_m": 0.00,    # Horizontal lateral offset (LiDAR relative to Camera center)
        "delta_y_m": 0.15,    # Forward longitudinal offset (LiDAR forward towards bumper)
        "delta_z_m": 0.25,    # Vertical elevation offset (Roof LiDAR is +0.25m higher than Camera)
        "yaw_trim_deg": 0.0,  # Rotational alignment trim (0.0 = aligned straight forward)
        "pitch_trim_deg": 0.0 # Vertical tilt trim
    }
}

def load_rig_config():
    """Loads configuration from JSON file or writes default if missing."""
    if not CONFIG_FILE.exists():
        save_rig_config(DEFAULT_RIG_CONFIG)
        return DEFAULT_RIG_CONFIG
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return DEFAULT_RIG_CONFIG

def save_rig_config(config_dict):
    """Saves updated calibration offsets to JSON file."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config_dict, f, indent=4)
