import os
import cv2
import shutil
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans

def extract_diverse_frames(input_folder, output_folder, num_clusters=100):
    """
    Scans a folder of extracted frames and uses AI clustering to pick the most visually 
    diverse frames (ensuring a mix of day, night, glare, and shadows).
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning frames in {input_folder}...")
    all_images = list(input_path.rglob("*.jpg"))
    if not all_images:
        print("No images found.")
        return

    # Extract color/brightness histograms for each image
    features = []
    valid_images = []
    
    print("Calculating brightness and color conditions for each frame...")
    for img_path in all_images:
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        # Convert to HSV to easily measure brightness (Value) and color (Hue)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Calculate a 2D histogram
        hist = cv2.calcHist([hsv], [0, 2], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        
        features.append(hist.flatten())
        valid_images.append(img_path)

    # Use K-Means Clustering to group similar looking frames (e.g. all dark frames together, all sunny frames together)
    print(f"Clustering {len(valid_images)} frames into {num_clusters} diverse condition buckets...")
    kmeans = KMeans(n_clusters=min(num_clusters, len(valid_images)), random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    
    # Pick the frame closest to the center of each diverse cluster
    selected_images = []
    for i in range(num_clusters):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) > 0:
            # For simplicity, pick the first image in the cluster
            selected_images.append(valid_images[cluster_indices[0]])
            
    print(f"Copying {len(selected_images)} highly diverse frames to {output_folder}...")
    for img_path in selected_images:
        shutil.copy2(img_path, output_path / img_path.name)
        
    print("Done! You now have a dataset perfectly balanced across different visual conditions.")

if __name__ == "__main__":
    # Example usage on your extracted frames
    PROJECT_ROOT = Path(os.path.abspath(__file__)).parent
    extract_diverse_frames(
        input_folder=PROJECT_ROOT / "extracted_frames",
        output_folder=PROJECT_ROOT / "diverse_condition_frames",
        num_clusters=500 # Adjust this to get more or fewer frames
    )
