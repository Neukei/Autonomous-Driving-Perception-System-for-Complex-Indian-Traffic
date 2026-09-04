import os
import cv2
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional, Tuple

from ..config import PipelineConfig
from .scene_change import SceneChangeDetector
from .complexity_scorer import ComplexityScorer

logger = logging.getLogger(__name__)

@dataclass
class ExtractionResult:
    """Stores the result metrics and metadata of an extraction run."""
    total_frames_processed: int = 0
    frames_extracted: int = 0
    frames_rejected: int = 0
    extraction_rate_by_segment: Dict[Tuple[float, float], float] = field(default_factory=dict)
    duration_sec: float = 0.0
    output_paths: List[str] = field(default_factory=list)

class HybridAdaptiveExtractor:
    """Adaptive frame extractor that intelligently extracts based on scene complexity.
    
    Combines complexity scoring (dynamic framerate extraction) with 
    scene change detection to ensure all important content is captured efficiently.
    """
    
    def __init__(self, config: PipelineConfig):
        """Initializes the HybridAdaptiveExtractor.
        
        Args:
            config (PipelineConfig): Configuration for the extraction pipeline.
        """
        self.config = config
        self.complexity_scorer = ComplexityScorer()
        self.scene_detector = SceneChangeDetector()
        # Optional quality filters to be set externally or passed during extract
        self.quality_filters = []
        logger.info("Initialized HybridAdaptiveExtractor")
        
    def extract(self, 
                video_path: str, 
                output_dir: str, 
                progress_callback: Optional[Callable[[float], None]] = None) -> ExtractionResult:
        """Runs the hybrid adaptive extraction pipeline on a video.
        
        Args:
            video_path (str): Path to the input video.
            output_dir (str): Directory to save extracted frames.
            progress_callback (Optional[Callable]): Callback receiving progress percentage (0.0-100.0).
            
        Returns:
            ExtractionResult: Summary and results of the extraction process.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        os.makedirs(output_dir, exist_ok=True)
        result = ExtractionResult()
        
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        result.duration_sec = total_frames / video_fps if video_fps > 0 else 0
        
        # Step 1: Detect Scene Changes
        logger.info("Starting Scene Change Detection pass...")
        scene_changes = self.scene_detector.detect_changes(video_path)
        scene_change_set = set(scene_changes)
        
        if progress_callback:
            progress_callback(20.0)
            
        # Step 2: Complexity Scoring (First Pass)
        logger.info("Starting Complexity Scoring pass...")
        segments = self.complexity_scorer.score_video_segments(video_path, segment_duration_sec=2.0)
        
        # Define extraction rates per segment based on complexity
        segment_extraction_rates = {}
        base_fps = self.config.extraction.base_fps
        boost_fps = self.config.extraction.boost_fps
        mid_fps = (base_fps + boost_fps) / 2.0
        for start_t, end_t, score in segments:
            if score < 0.3:
                fps = base_fps   # LOW complexity
            elif score <= 0.6:
                fps = mid_fps    # MEDIUM complexity
            else:
                fps = boost_fps  # HIGH complexity
            segment_extraction_rates[(start_t, end_t)] = fps
            result.extraction_rate_by_segment[(start_t, end_t)] = fps
            
        if progress_callback:
            progress_callback(40.0)
            
        # Step 3: Second Pass Extraction
        logger.info("Starting extraction pass...")
        current_segment_idx = 0
        frames_to_extract = set(scene_change_set) # Start with scene change frames
        
        # Calculate periodic frames to extract based on segment FPS
        for (start_t, end_t), fps in segment_extraction_rates.items():
            start_frame = int(start_t * video_fps)
            end_frame = int(end_t * video_fps)
            interval = max(1, int(video_fps / fps))
            
            for f_idx in range(start_frame, end_frame, interval):
                frames_to_extract.add(f_idx)
                
        sorted_extraction_frames = sorted(list(frames_to_extract))
        result.total_frames_processed = len(sorted_extraction_frames)
        
        for i, frame_idx in enumerate(sorted_extraction_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Apply quality filters if any
            rejected = False
            for q_filter in self.quality_filters:
                if not q_filter(frame):
                    rejected = True
                    break
                    
            if rejected:
                result.frames_rejected += 1
                continue
                
            timestamp_sec = frame_idx / video_fps
            prefix = "scene" if frame_idx in scene_change_set else "frame"
            filename = f"{prefix}_{frame_idx:06d}_{timestamp_sec:.2f}s.jpg"
            output_path = os.path.join(output_dir, filename)
            
            # Auto-rotate if dashcam is mounted inverted
            if self.config.extraction.auto_rotate_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            
            quality = self.config.extraction.output_quality
            cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            result.output_paths.append(output_path)
            result.frames_extracted += 1
            
            if progress_callback and i % 10 == 0:
                progress = 40.0 + (60.0 * (i / len(sorted_extraction_frames)))
                progress_callback(progress)
                
        cap.release()
        
        if progress_callback:
            progress_callback(100.0)
            
        logger.info(f"Hybrid extraction complete: Extracted {result.frames_extracted}, Rejected {result.frames_rejected}")
        return result
