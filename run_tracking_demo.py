"""
ByteTrack Object Tracking Demo
Runs YOLO11s + ByteTrack on a dashcam video and saves a 30-second demo clip.
"""
import cv2
from ultralytics import YOLO
from pathlib import Path
import os

def run_tracking_demo(model_path, video_path, output_path, duration_sec=30):
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Get video info to calculate frame limit
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    max_frames = int(fps * duration_sec)
    print(f"Video: {w}x{h} @ {fps:.1f} FPS — extracting {duration_sec}s ({max_frames} frames)")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    # Re-open video for frame-by-frame tracking
    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0
    
    print("Running ByteTrack object tracking...")
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Rotate 180 degrees (dashcam is upside down)
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        
        # Run tracking with ByteTrack
        results = model.track(
            frame,
            persist=True,           # Persist track IDs across frames
            tracker="bytetrack.yaml",  # Use ByteTrack algorithm
            conf=0.3,               # Confidence threshold
            iou=0.5,                # IOU threshold for association
            verbose=False
        )
        
        # Draw the tracked results on the frame
        annotated_frame = results[0].plot()
        
        # Add tracking info overlay
        num_tracked = len(results[0].boxes) if results[0].boxes is not None else 0
        cv2.putText(annotated_frame, f"DREIVE - ByteTrack | Tracked: {num_tracked} objects", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Frame {frame_count}/{max_frames}", 
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        out.write(annotated_frame)
        frame_count += 1
        
        if frame_count % 50 == 0:
            print(f"  Processed {frame_count}/{max_frames} frames...")
    
    cap.release()
    out.release()
    print(f"Demo saved to: {output_path}")
    print(f"Total frames processed: {frame_count}")

if __name__ == "__main__":
    PROJECT = Path(os.path.abspath(__file__)).parent
    
    # Use a Karjat highway video (dense mixed traffic)
    VIDEO = PROJECT / "Towards Karjat-20260815T083400Z-1-001" / "Towards Karjat" / "2026-07-31_13-07-42.mp4"
    MODEL = PROJECT / "runs" / "detect" / "runs" / "detect" / "indian_driving_yolo11s" / "weights" / "best.pt"
    OUTPUT = PROJECT / "metrics_and_results" / "tracking_demo.mp4"
    
    run_tracking_demo(MODEL, VIDEO, OUTPUT, duration_sec=30)
