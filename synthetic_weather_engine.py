import cv2
import os
import random
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm

def add_rain(image):
    """Simulates rain by drawing slanted lines and adding slight blur"""
    rain_img = image.copy()
    h, w = rain_img.shape[:2]
    
    # Generate random rain drops
    drop_length = 20
    drop_width = 1
    num_drops = random.randint(300, 1000)
    
    for _ in range(num_drops):
        x = random.randint(0, w - drop_length)
        y = random.randint(0, h - drop_length)
        cv2.line(rain_img, (x, y), (x + drop_length, y + int(drop_length*1.5)), (200, 200, 210), drop_width)
    
    # Add slight blur for visibility drop
    rain_img = cv2.blur(rain_img, (3, 3))
    return rain_img

def add_fog(image):
    """Simulates fog/smog by blending with a gray overlay and reducing contrast"""
    h, w = image.shape[:2]
    fog_overlay = np.full((h, w, 3), (180, 180, 180), dtype=np.uint8)
    
    # Random fog density between 0.3 and 0.6
    density = random.uniform(0.3, 0.6)
    foggy_img = cv2.addWeighted(image, 1 - density, fog_overlay, density, 0)
    return foggy_img

def add_night(image):
    """Simulates night by darkening the image and shifting to blue/cool tones"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Darken significantly (reduce Value)
    v = cv2.multiply(v, random.uniform(0.3, 0.5))
    
    # Desaturate slightly
    s = cv2.multiply(s, random.uniform(0.5, 0.8))
    
    # Shift Hue slightly towards blue/cool (Night time ambient)
    h = np.mod(h + 10, 180).astype(np.uint8)
    
    hsv = cv2.merge((h, s, v))
    night_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return night_img

def augment_dataset(dataset_dir, sample_size=2000):
    """
    Takes a subset of the dataset, applies synthetic weather/conditions,
    and saves them back with duplicated label files.
    """
    img_dir = Path(dataset_dir) / "images" / "train"
    lbl_dir = Path(dataset_dir) / "labels" / "train"
    
    all_images = list(img_dir.glob("*.jpg"))
    if not all_images:
        print("No images found in train directory.")
        return
        
    # Pick a random subset of images to augment
    subset = random.sample(all_images, min(sample_size, len(all_images)))
    
    print(f"Applying synthetic weather conditions to {len(subset)} frames...")
    
    conditions = [('rain', add_rain), ('fog', add_fog), ('night', add_night)]
    
    for img_path in tqdm(subset):
        label_path = lbl_dir / (img_path.stem + ".txt")
        
        # Only process if it has a label
        if not label_path.exists():
            continue
            
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        # Pick a random condition
        cond_name, cond_func = random.choice(conditions)
        
        # Apply synthetic weather
        aug_img = cond_func(img)
        
        # Save new image and duplicate label
        new_img_name = f"synth_{cond_name}_{img_path.name}"
        new_lbl_name = f"synth_{cond_name}_{label_path.name}"
        
        cv2.imwrite(str(img_dir / new_img_name), aug_img)
        shutil.copy2(label_path, lbl_dir / new_lbl_name)

if __name__ == "__main__":
    PROJECT_ROOT = Path(os.path.abspath(__file__)).parent
    unified_dir = PROJECT_ROOT / "unified_dataset"
    
    print("Starting Autonomous Weather Simulator...")
    augment_dataset(unified_dir, sample_size=3000)
    print("Done! The dataset now contains synthetic rain, fog, and night driving scenarios.")
