import cv2
import numpy as np
import datetime
import logging
from typing import Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)

def get_video_info(path: str) -> Dict[str, Any]:
    """Get metadata information about a video file.
    
    Args:
        path: Path to the video file.
        
    Returns:
        Dictionary containing fps, frame_count, width, height, duration_sec, and codec.
    """
    cap = open_video(path)
    if cap is None:
        return {}
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec_float = cap.get(cv2.CAP_PROP_FOURCC)
    codec = "".join([chr((int(codec_float) >> 8 * i) & 0xFF) for i in range(4)])
    
    duration_sec = frame_count / fps if fps > 0 else 0.0
    
    cap.release()
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": duration_sec,
        "codec": codec
    }

def open_video(path: str) -> Optional[cv2.VideoCapture]:
    """Open a video file with error handling.
    
    Args:
        path: Path to the video file.
        
    Returns:
        cv2.VideoCapture object if successful, None otherwise.
    """
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            logger.error(f"Failed to open video file at: {path}")
            return None
        return cap
    except Exception as e:
        logger.error(f"Error while opening video {path}: {str(e)}")
        return None

def read_frame_at(cap: cv2.VideoCapture, frame_idx: int) -> Optional[np.ndarray]:
    """Seek to a specific frame index and read the frame.
    
    Args:
        cap: The open cv2.VideoCapture object.
        frame_idx: The index of the frame to read.
        
    Returns:
        The frame as a numpy array, or None if failed.
    """
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            return frame
        logger.warning(f"Failed to read frame at index: {frame_idx}")
        return None
    except Exception as e:
        logger.error(f"Error reading frame at index {frame_idx}: {str(e)}")
        return None

def resize_frame(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize a frame maintaining aspect ratio and padding if needed.
    
    Args:
        frame: The input image as a numpy array.
        target_size: A tuple of (width, height).
        
    Returns:
        The resized and padded image.
    """
    h, w = frame.shape[:2]
    target_w, target_h = target_size
    
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left
    
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded

def frame_to_timestamp(frame_idx: int, fps: float) -> str:
    """Convert a frame index to a HH:MM:SS.mmm formatted timestamp string.
    
    Args:
        frame_idx: The index of the frame.
        fps: The frames per second of the video.
        
    Returns:
        Timestamp string formatted as 'HH:MM:SS.mmm'.
    """
    if fps <= 0:
        return "00:00:00.000"
        
    total_seconds = frame_idx / fps
    td = datetime.timedelta(seconds=total_seconds)
    
    secs = int(td.total_seconds())
    microsecs = td.microseconds
    
    hours = secs // 3600
    minutes = (secs % 3600) // 60
    seconds = secs % 60
    milliseconds = microsecs // 1000
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
