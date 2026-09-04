"""
Quality filtering module to assess and filter frames based on multiple criteria.
"""
import cv2
import numpy as np
import logging
from typing import Tuple, Dict, Any

from .blur_detector import BlurDetector
from .deduplicator import FrameDeduplicator
from ..config import QualityConfig

logger = logging.getLogger(__name__)


class QualityFilter:
    """
    Combines various checks (blur, brightness, contrast, deduplication) to filter frames.
    """

    def __init__(self, config: QualityConfig) -> None:
        """
        Initializes the QualityFilter.

        Args:
            config: A configuration object holding thresholds and settings.
        """
        self.config = config
        
        # Initialize detectors
        self.blur_detector = BlurDetector(threshold=getattr(config, 'blur_threshold', 100.0))
        self.deduplicator = FrameDeduplicator(
            hash_size=getattr(config, 'hash_size', 16),
            distance_threshold=getattr(config, 'distance_threshold', 5)
        )
        self.enable_dedup = getattr(config, 'enable_dedup', True)
        
        # Stats tracking
        self.stats = {
            'total_checked': 0,
            'passed': 0,
            'rejected_blur': 0,
            'rejected_brightness': 0,
            'rejected_contrast': 0,
            'rejected_duplicate': 0
        }

    def _check_brightness(self, frame: np.ndarray) -> Tuple[bool, float]:
        """Checks if frame brightness is within acceptable limits."""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        mean_brightness = gray.mean()
        min_b = getattr(self.config, 'min_brightness', 20.0)
        max_b = getattr(self.config, 'max_brightness', 240.0)
        
        is_ok = min_b <= mean_brightness <= max_b
        return is_ok, float(mean_brightness)

    def _check_contrast(self, frame: np.ndarray) -> Tuple[bool, float]:
        """Checks if frame contrast is above minimum threshold."""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        std_contrast = gray.std()
        min_c = getattr(self.config, 'min_contrast', 10.0)
        
        is_ok = std_contrast >= min_c
        return is_ok, float(std_contrast)

    def filter_frame(self, frame: np.ndarray, frame_id: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Applies all quality filters to a frame.

        Args:
            frame (np.ndarray): The input image frame.
            frame_id (str): A unique identifier for the frame (used for deduplication).

        Returns:
            Tuple[bool, dict]: A boolean indicating if the frame passed all checks, 
                               and a dictionary containing details of the checks.
        """
        self.stats['total_checked'] += 1
        
        details: Dict[str, Any] = {}
        
        # 1. Blur Check
        is_blurry, blur_score = self.blur_detector.is_blurry(frame)
        details['blur_score'] = blur_score
        details['is_blurry'] = is_blurry
        if is_blurry:
            self.stats['rejected_blur'] += 1
            details['reject_reason'] = 'blur'
            return False, details
            
        # 2. Brightness Check
        brightness_ok, brightness_val = self._check_brightness(frame)
        details['brightness'] = brightness_val
        details['brightness_ok'] = brightness_ok
        if not brightness_ok:
            self.stats['rejected_brightness'] += 1
            details['reject_reason'] = 'brightness'
            return False, details
            
        # 3. Contrast Check
        contrast_ok, contrast_val = self._check_contrast(frame)
        details['contrast'] = contrast_val
        details['contrast_ok'] = contrast_ok
        if not contrast_ok:
            self.stats['rejected_contrast'] += 1
            details['reject_reason'] = 'contrast'
            return False, details
            
        # 4. Duplicate Check
        if self.enable_dedup:
            is_dup, min_dist = self.deduplicator.is_duplicate(frame)
            details['min_hash_distance'] = min_dist
            details['is_duplicate'] = is_dup
            
            if is_dup:
                self.stats['rejected_duplicate'] += 1
                details['reject_reason'] = 'duplicate'
                return False, details
                
            # If not duplicate, add to store
            self.deduplicator.add_frame(frame, frame_id)
            
        self.stats['passed'] += 1
        return True, details

    def get_stats(self) -> Dict[str, int]:
        """
        Returns the statistics of the filtering process.

        Returns:
            Dict[str, int]: A dictionary containing total checked, passed, and rejection counts.
        """
        return self.stats.copy()
