import cv2
import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class FixedIntervalExtractor:
    """Extracts frames from a video at a fixed time interval.
    
    Attributes:
        fps (float): Frames per second to extract. For example, 0.5 means
            extracting 1 frame every 2 seconds.
    """
    
    def __init__(self, fps: float = 0.5):
        """Initializes the FixedIntervalExtractor.
        
        Args:
            fps (float): Frames per second to extract. Defaults to 0.5.
        """
        self.fps = fps
        logger.info(f"Initialized FixedIntervalExtractor with {self.fps} FPS")
        
    def extract(self, video_path: str, output_dir: str, quality: int = 95) -> List[Dict[str, Any]]:
        """Extracts frames at fixed intervals.
        
        Args:
            video_path (str): Path to the input video file.
            output_dir (str): Path to the output directory to save frames.
            quality (int): JPEG quality of the saved frames (0-100). Defaults to 95.
            
        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing 'frame_idx',
                'timestamp_sec', and 'output_path' for each extracted frame.
                
        Raises:
            FileNotFoundError: If the video_path does not exist.
            ValueError: If the video cannot be opened.
        """
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            raise ValueError(f"Failed to open video file: {video_path}")
            
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration_sec = total_frames / video_fps if video_fps > 0 else 0
        
        logger.info(f"Processing video: {video_path} (Duration: {video_duration_sec:.2f}s, FPS: {video_fps})")
        
        frame_interval = max(1, int(round(video_fps / self.fps))) if self.fps > 0 else 1
        
        results = []
        extracted_count = 0
        
        # Determine milestones for 10% progress logging
        milestone_frames = {int(total_frames * (i / 10.0)) for i in range(1, 11)}
        milestones_passed = set()
        
        current_frame = 0
        while current_frame < total_frames:
            # Seek to the desired frame efficiently
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            
            if not ret:
                logger.warning(f"Failed to read frame {current_frame} from video.")
                break
                
            timestamp_sec = current_frame / video_fps
            filename = f"frame_{current_frame:06d}_{timestamp_sec:.2f}s.jpg"
            output_path = os.path.join(output_dir, filename)
            
            # Save the frame
            cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            results.append({
                "frame_idx": current_frame,
                "timestamp_sec": timestamp_sec,
                "output_path": output_path
            })
            extracted_count += 1
            
            # Progress logging
            for m in sorted(milestone_frames):
                if current_frame >= m and m not in milestones_passed:
                    progress_pct = (m / total_frames) * 100
                    logger.info(f"Progress: {progress_pct:.0f}% ({current_frame}/{total_frames} frames processed)")
                    milestones_passed.add(m)
            
            current_frame += frame_interval
            
        cap.release()
        logger.info(f"Extraction complete. Saved {extracted_count} frames to {output_dir}")
        return results
