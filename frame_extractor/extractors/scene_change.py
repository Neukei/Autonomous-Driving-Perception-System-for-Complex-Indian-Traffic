import cv2
import os
import logging
import numpy as np
from skimage.metrics import structural_similarity as ssim
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class SceneChangeDetector:
    """Detects scene changes in a video and extracts relevant frames.
    
    Uses SSIM and histogram correlation to detect significant visual changes
    between consecutive frames at a reduced resolution for efficiency.
    """
    
    def __init__(self, 
                 ssim_threshold: float = 0.7, 
                 hist_threshold: float = 0.4,
                 min_interval_ms: int = 300,
                 process_resolution: tuple = (640, 360)):
        """Initializes the SceneChangeDetector.
        
        Args:
            ssim_threshold (float): Threshold below which SSIM indicates a change.
            hist_threshold (float): Threshold below which histogram correlation indicates a change.
            min_interval_ms (int): Minimum milliseconds between detected scene changes.
            process_resolution (tuple): Resolution to resize frames to for detection logic.
        """
        self.ssim_threshold = ssim_threshold
        self.hist_threshold = hist_threshold
        self.min_interval_ms = min_interval_ms
        self.process_resolution = process_resolution
        logger.info("Initialized SceneChangeDetector (SSIM thresh: %.2f, Hist thresh: %.2f)",
                    self.ssim_threshold, self.hist_threshold)
                    
    def _compute_histogram(self, image: np.ndarray) -> np.ndarray:
        """Computes a normalized color histogram for an image."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist
        
    def detect_changes(self, video_path: str) -> List[int]:
        """Detects scene changes in the video.
        
        Args:
            video_path (str): Path to the input video file.
            
        Returns:
            List[int]: A list of frame indices where scene changes occur.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
            
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        min_frames_between_changes = int((self.min_interval_ms / 1000.0) * video_fps)
        
        changes = []
        prev_frame_gray = None
        prev_hist = None
        last_change_frame = -min_frames_between_changes
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize for performance
            small_frame = cv2.resize(frame, self.process_resolution)
            gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            hist = self._compute_histogram(small_frame)
            
            if prev_frame_gray is not None and prev_hist is not None:
                # Calculate histogram correlation
                hist_corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                
                # Calculate SSIM
                score, _ = ssim(prev_frame_gray, gray_frame, full=True)
                
                # Check for scene change
                is_change = score < self.ssim_threshold or hist_corr < self.hist_threshold
                
                if is_change and (frame_idx - last_change_frame) >= min_frames_between_changes:
                    changes.append(frame_idx)
                    last_change_frame = frame_idx
                    logger.debug(f"Scene change detected at frame {frame_idx} (SSIM: {score:.3f}, Hist Corr: {hist_corr:.3f})")
                    
            prev_frame_gray = gray_frame
            prev_hist = hist
            frame_idx += 1
            
        cap.release()
        logger.info(f"Detected {len(changes)} scene changes in {video_path}")
        return changes
        
    def extract_at_changes(self, video_path: str, output_dir: str, context_frames: int = 2) -> List[Dict[str, Any]]:
        """Extracts frames at scene changes along with context frames.
        
        Args:
            video_path (str): Path to the input video.
            output_dir (str): Directory to save extracted frames.
            context_frames (int): Number of frames to extract before and after the scene change.
            
        Returns:
            List[Dict[str, Any]]: Metadata about extracted frames.
        """
        change_indices = self.detect_changes(video_path)
        if not change_indices:
            return []
            
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        os.makedirs(output_dir, exist_ok=True)
        
        # Compile a list of frames to extract based on changes and context
        frames_to_extract = set()
        for idx in change_indices:
            start = max(0, idx - context_frames)
            end = min(total_frames - 1, idx + context_frames)
            for f in range(start, end + 1):
                frames_to_extract.add(f)
                
        sorted_frames = sorted(list(frames_to_extract))
        results = []
        
        for f_idx in sorted_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            timestamp_sec = f_idx / video_fps
            filename = f"scene_{f_idx:06d}_{timestamp_sec:.2f}s.jpg"
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            results.append({
                "frame_idx": f_idx,
                "timestamp_sec": timestamp_sec,
                "output_path": output_path,
                "is_scene_change": f_idx in change_indices
            })
            
        cap.release()
        logger.info(f"Extracted {len(results)} frames (including context) around scene changes.")
        return results
