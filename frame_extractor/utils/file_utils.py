import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def ensure_dir(path: str) -> Path:
    """Create directory if it does not exist.
    
    Args:
        path: String path to the directory.
        
    Returns:
        Path object of the directory.
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def get_video_files(directory: str, extensions: List[str]) -> List[Path]:
    """Recursively find all video files in a directory matching the extensions.
    
    Args:
        directory: Root directory to search.
        extensions: List of extensions to look for (e.g., ['.mp4', '.avi']).
        
    Returns:
        List of Path objects for found video files.
    """
    video_files = []
    base_dir = Path(directory)
    
    if not base_dir.exists():
        logger.error(f"Directory not found: {directory}")
        return video_files
        
    for ext in extensions:
        pattern = f"**/*{ext}"
        video_files.extend(base_dir.glob(pattern))
        
    return video_files

def generate_frame_filename(video_name: str, frame_idx: int, timestamp: str, suffix: str = '.jpg') -> str:
    """Generate a descriptive filename for an extracted frame.
    
    Args:
        video_name: Base name of the source video.
        frame_idx: Index of the frame.
        timestamp: Timestamp string from frame_to_timestamp (HH:MM:SS.mmm).
        suffix: File extension for the image.
        
    Returns:
        Formatted filename string.
    """
    parts = timestamp.split(":")
    if len(parts) == 3:
        h, m, s_ms = parts
        s, ms = s_ms.split(".")
        if int(h) > 0:
            formatted_time = f"{h}h{m}m{s}s{ms}ms"
        else:
            formatted_time = f"{m}m{s}s{ms}ms"
    else:
        # Fallback if timestamp format is unexpected
        formatted_time = timestamp.replace(":", "h", 1).replace(":", "m", 1).replace(".", "s") + "ms"
        
    return f"{video_name}_f{frame_idx:05d}_{formatted_time}{suffix}"

def get_output_subdir(output_base: str, video_path: str, input_base: str) -> Path:
    """Determine the output directory by mirroring the input structure.
    
    Args:
        output_base: Root output directory.
        video_path: Absolute path to the source video.
        input_base: Root input directory.
        
    Returns:
        Path object for the specific output subdirectory.
    """
    video_p = Path(video_path)
    input_p = Path(input_base)
    output_p = Path(output_base)
    
    try:
        rel_path = video_p.parent.relative_to(input_p)
    except ValueError:
        logger.warning(f"Video path {video_path} is not relative to {input_base}. Storing at root.")
        rel_path = Path("")
        
    return output_p / rel_path

def save_extraction_manifest(output_dir: str, manifest: List[Dict[str, Any]]) -> None:
    """Save a JSON manifest of all extracted frames.
    
    Args:
        output_dir: Directory to save the manifest.json file.
        manifest: List of dictionaries containing frame metadata.
    """
    ensure_dir(output_dir)
    manifest_path = Path(output_dir) / "manifest.json"
    
    try:
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=4)
        logger.info(f"Successfully saved extraction manifest to {manifest_path}")
    except Exception as e:
        logger.error(f"Failed to save manifest to {manifest_path}: {str(e)}")
