"""
Frame deduplication module using perceptual hashing.
"""
import cv2
import numpy as np
from typing import Tuple, Dict

class FrameDeduplicator:
    """
    Deduplicates frames using average hash (aHash) and difference hash (dHash).
    """

    def __init__(self, hash_size: int = 16, distance_threshold: int = 5) -> None:
        """
        Initializes the FrameDeduplicator.

        Args:
            hash_size (int): The size to resize the image for hashing.
            distance_threshold (int): Maximum Hamming distance to be considered a duplicate.
        """
        self.hash_size = hash_size
        self.distance_threshold = distance_threshold
        self.stored_hashes: Dict[str, int] = {}

    def _compute_ahash(self, frame: np.ndarray) -> int:
        """
        Computes the average hash (aHash) of a frame.

        Args:
            frame (np.ndarray): The input image frame.

        Returns:
            int: The computed average hash as an integer.
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        resized = cv2.resize(gray, (self.hash_size, self.hash_size), interpolation=cv2.INTER_AREA)
        mean_val = resized.mean()
        
        hash_val = 0
        for i, pixel in enumerate(resized.flatten()):
            if pixel > mean_val:
                hash_val |= (1 << i)
                
        return hash_val

    def _compute_dhash(self, frame: np.ndarray) -> int:
        """
        Computes the difference hash (dHash) of a frame.

        Args:
            frame (np.ndarray): The input image frame.

        Returns:
            int: The computed difference hash as an integer.
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        resized = cv2.resize(gray, (self.hash_size + 1, self.hash_size), interpolation=cv2.INTER_AREA)
        
        hash_val = 0
        idx = 0
        for row in range(self.hash_size):
            for col in range(self.hash_size):
                if resized[row, col] > resized[row, col + 1]:
                    hash_val |= (1 << idx)
                idx += 1
                
        return hash_val

    def _hamming_distance(self, hash1: int, hash2: int) -> int:
        """
        Computes the Hamming distance between two integer hashes.

        Args:
            hash1 (int): First hash.
            hash2 (int): Second hash.

        Returns:
            int: The number of differing bits.
        """
        x = hash1 ^ hash2
        # Count set bits (Brian Kernighan's algorithm)
        count = 0
        while x > 0:
            x &= (x - 1)
            count += 1
        return count

    def is_duplicate(self, frame: np.ndarray) -> Tuple[bool, int]:
        """
        Checks if a frame is a duplicate of any previously stored frame.

        Args:
            frame (np.ndarray): The input image frame.

        Returns:
            Tuple[bool, int]: A tuple containing a boolean (True if duplicate) 
                              and the minimum Hamming distance found (or -1 if no hashes stored).
        """
        if not self.stored_hashes:
            return False, -1

        current_hash = self._compute_dhash(frame) # Using dhash for better resistance to brightness/contrast changes
        
        min_distance = float('inf')
        
        for stored_hash in self.stored_hashes.values():
            distance = self._hamming_distance(current_hash, stored_hash)
            if distance < min_distance:
                min_distance = distance
                if min_distance == 0: # Exact match found, early exit
                    break
                    
        return min_distance <= self.distance_threshold, int(min_distance)

    def add_frame(self, frame: np.ndarray, frame_id: str) -> None:
        """
        Adds a frame's hash to the store.

        Args:
            frame (np.ndarray): The image frame.
            frame_id (str): A unique identifier for the frame.
        """
        current_hash = self._compute_dhash(frame)
        self.stored_hashes[frame_id] = current_hash

    def reset(self) -> None:
        """
        Clears all stored hashes.
        """
        self.stored_hashes.clear()
