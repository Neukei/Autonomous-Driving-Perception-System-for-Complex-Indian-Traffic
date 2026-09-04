"""
Blur detection module for assessing frame sharpness.
"""
import cv2
import numpy as np
from typing import Tuple

class BlurDetector:
    """
    Detects if a frame is blurry using Laplacian variance.
    """

    def __init__(self, threshold: float = 100.0) -> None:
        """
        Initializes the BlurDetector.

        Args:
            threshold (float): Laplacian variance threshold. Below this = blurry.
        """
        self.threshold = threshold

    def compute_blur_score(self, frame: np.ndarray) -> float:
        """
        Computes the blur score (Laplacian variance) for a frame.
        Higher score means the image is sharper.

        Args:
            frame (np.ndarray): The input image frame (BGR or Grayscale).

        Returns:
            float: The Laplacian variance score.
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(score)

    def is_blurry(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Determines if a frame is blurry based on the threshold.

        Args:
            frame (np.ndarray): The input image frame.

        Returns:
            Tuple[bool, float]: A tuple containing a boolean (True if blurry) 
                                and the computed blur score.
        """
        score = self.compute_blur_score(frame)
        return score < self.threshold, score
