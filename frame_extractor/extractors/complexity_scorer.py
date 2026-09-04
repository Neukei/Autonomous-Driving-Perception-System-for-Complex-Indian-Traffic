import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

class ComplexityScorer:
    """Scores visual complexity of video frames or segments.
    
    Complexity is estimated using edge density, optical flow magnitude,
    and an object density proxy (contour counts).
    """
    
    def __init__(self, process_resolution: tuple = (640, 360)):
        """Initializes the ComplexityScorer.
        
        Args:
            process_resolution (tuple): Resolution for processing to improve speed.
        """
        self.process_resolution = process_resolution
        
    def score_frame(self, frame: np.ndarray, prev_frame: Optional[np.ndarray] = None) -> float:
        """Computes a complexity score (0.0 to 1.0) for a single frame.
        
        Args:
            frame (np.ndarray): The current frame.
            prev_frame (Optional[np.ndarray]): The previous frame, used for optical flow.
                If None, motion complexity is scored as 0.
                
        Returns:
            float: Complexity score bounded between 0.0 and 1.0.
        """
        small_frame = cv2.resize(frame, self.process_resolution)
        gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Edge Density
        edges = cv2.Canny(gray_frame, 100, 200)
        edge_density = np.sum(edges > 0) / (self.process_resolution[0] * self.process_resolution[1])
        # Normalize edge density (empirical max around 0.15 for complex scenes)
        norm_edge_density = min(1.0, edge_density / 0.15)
        
        # 2. Object Density Proxy (Contours)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 50
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        # Normalize contour count (empirical max around 100 for dense scenes)
        norm_obj_density = min(1.0, len(valid_contours) / 100.0)
        
        # 3. Optical Flow (Motion Complexity)
        norm_flow = 0.0
        if prev_frame is not None:
            small_prev = cv2.resize(prev_frame, self.process_resolution)
            gray_prev = cv2.cvtColor(small_prev, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(gray_prev, gray_frame, None, 
                                                0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mean_mag = np.mean(mag)
            # Normalize motion (empirical max mean magnitude around 10.0)
            norm_flow = min(1.0, mean_mag / 10.0)
            
        # Weighted combination of metrics
        if prev_frame is not None:
            score = (0.4 * norm_edge_density) + (0.3 * norm_obj_density) + (0.3 * norm_flow)
        else:
            # If no previous frame, base entirely on spatial complexity
            score = (0.6 * norm_edge_density) + (0.4 * norm_obj_density)
            
        return float(np.clip(score, 0.0, 1.0))
        
    def score_video_segments(self, video_path: str, segment_duration_sec: float = 2.0) -> List[Tuple[float, float, float]]:
        """Scores complexity of segments in a video.
        
        Args:
            video_path (str): Path to the video file.
            segment_duration_sec (float): Duration of each segment in seconds.
            
        Returns:
            List[Tuple[float, float, float]]: A list of tuples containing
                (start_time, end_time, avg_complexity_score).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
            
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_per_segment = int(video_fps * segment_duration_sec)
        
        segments = []
        current_segment_scores = []
        segment_start_time = 0.0
        
        prev_frame = None
        frame_idx = 0
        
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Sample 2 frames per second within the segment to compute complexity fast
            sample_rate = max(1, int(video_fps / 2))
            if frame_idx % sample_rate == 0:
                score = self.score_frame(frame, prev_frame)
                current_segment_scores.append(score)
                prev_frame = frame
                
            frame_idx += 1
            
            if frame_idx % frames_per_segment == 0 or frame_idx == total_frames:
                segment_end_time = frame_idx / video_fps
                avg_score = np.mean(current_segment_scores) if current_segment_scores else 0.0
                segments.append((segment_start_time, segment_end_time, float(avg_score)))
                
                logger.debug(f"Segment [{segment_start_time:.2f}s - {segment_end_time:.2f}s]: Complexity = {avg_score:.3f}")
                
                segment_start_time = segment_end_time
                current_segment_scores = []
                
        cap.release()
        logger.info(f"Analyzed {len(segments)} segments for video {video_path}")
        return segments
