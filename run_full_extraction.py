"""
Batch extraction script for all unique driving videos.
Processes Towards Home and Towards Karjat folders, skipping duplicates.
"""
import sys
import os
import time
import json
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from frame_extractor.config import PipelineConfig
from frame_extractor.extractors.hybrid import HybridAdaptiveExtractor, ExtractionResult
from frame_extractor.processors.quality_filter import QualityFilter
from frame_extractor.utils.video_utils import get_video_info
from frame_extractor.utils.file_utils import ensure_dir, get_video_files

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Define all video source folders (avoiding duplicates)
# Main "Towards Home" folder is duplicate of "Towards Home-003" — skip it
VIDEO_FOLDERS = [
    # Towards Home (parts 1-3 from zip archives, main folder is dup of 003)
    ("Towards Home-20260815T083359Z-1-001/Towards Home", "towards_home"),
    ("Towards Home-20260815T083359Z-1-002/Towards Home", "towards_home"),
    # Main "Towards Home" has same files as 003, so use main and skip 003
    ("Towards Home", "towards_home"),
    # Towards Karjat (parts 1-4)
    ("Towards Karjat-20260815T083400Z-1-001/Towards Karjat", "towards_karjat"),
    ("Towards Karjat-20260815T083400Z-1-002/Towards Karjat", "towards_karjat"),
    ("Towards Karjat-20260815T083400Z-1-003/Towards Karjat", "towards_karjat"),
    ("Towards Karjat-20260815T083400Z-1-004/Towards Karjat", "towards_karjat"),
]

OUTPUT_BASE = "extracted_frames"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    start_time = time.time()
    config = PipelineConfig()
    config.extraction.base_fps = 0.5
    config.extraction.boost_fps = 3.0
    config.extraction.auto_rotate_180 = True
    config.quality.blur_threshold = 100.0

    all_summaries = []
    processed_filenames = set()  # Track by filename to skip dups across zip parts
    total_extracted = 0
    total_rejected = 0
    total_skipped = 0
    total_errored = 0
    video_num = 0

    for folder_rel, route_name in VIDEO_FOLDERS:
        folder_path = os.path.join(PROJECT_ROOT, folder_rel)
        if not os.path.isdir(folder_path):
            logger.warning(f"Folder not found, skipping: {folder_rel}")
            continue

        videos = sorted([
            f for f in os.listdir(folder_path) 
            if f.lower().endswith(".mp4")
        ])

        for video_file in videos:
            # Skip duplicates (same filename already processed)
            if video_file in processed_filenames:
                total_skipped += 1
                logger.info(f"  ⊘ Skipping duplicate: {video_file}")
                continue

            video_path = os.path.join(folder_path, video_file)
            
            # Verify video is readable
            info = get_video_info(video_path)
            if not info:
                logger.error(f"  ✗ Corrupt/unreadable: {video_file}")
                total_errored += 1
                continue

            video_num += 1
            processed_filenames.add(video_file)

            # Output directory: extracted_frames/<route>/<video_stem>
            video_stem = os.path.splitext(video_file)[0]
            output_dir = os.path.join(PROJECT_ROOT, OUTPUT_BASE, route_name, video_stem)
            ensure_dir(output_dir)

            logger.info(f"[{video_num}] ▶ {route_name}/{video_file} "
                       f"({info['duration_sec']:.0f}s, {info['frame_count']} frames)")

            try:
                # Build extractor and quality filter
                extractor = HybridAdaptiveExtractor(config)
                quality_filter = QualityFilter(config.quality)

                def _quality_gate(frame, _qf=quality_filter, _vid=video_stem):
                    frame_id = f"{_vid}_{time.monotonic_ns()}"
                    passed, _ = _qf.filter_frame(frame, frame_id)
                    return passed

                extractor.quality_filters = [_quality_gate]

                result: ExtractionResult = extractor.extract(video_path, output_dir)
                q_stats = quality_filter.get_stats()

                total_extracted += result.frames_extracted
                total_rejected += result.frames_rejected

                summary = {
                    "video": video_file,
                    "route": route_name,
                    "source_folder": folder_rel,
                    "duration_sec": round(result.duration_sec, 2),
                    "frames_extracted": result.frames_extracted,
                    "frames_rejected": result.frames_rejected,
                    "quality_stats": q_stats,
                    "output_dir": output_dir,
                }
                all_summaries.append(summary)

                logger.info(f"    ✓ {result.frames_extracted} extracted, "
                           f"{result.frames_rejected} rejected")

            except Exception as e:
                logger.error(f"    ✗ FAILED: {e}", exc_info=False)
                total_errored += 1
                all_summaries.append({
                    "video": video_file,
                    "route": route_name,
                    "error": str(e),
                })

    elapsed = time.time() - start_time

    # Print final summary
    logger.info("=" * 70)
    logger.info("FULL DATASET EXTRACTION COMPLETE")
    logger.info(f"  Unique videos processed : {video_num}")
    logger.info(f"  Duplicates skipped      : {total_skipped}")
    logger.info(f"  Corrupt/errored         : {total_errored}")
    logger.info(f"  Total frames extracted   : {total_extracted}")
    logger.info(f"  Total frames rejected    : {total_rejected}")
    logger.info(f"  Time elapsed            : {elapsed/60:.1f} minutes")
    logger.info(f"  Output directory        : {OUTPUT_BASE}/")
    logger.info("=" * 70)

    # Save manifest
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_videos": video_num,
        "duplicates_skipped": total_skipped,
        "corrupt_errored": total_errored,
        "total_frames_extracted": total_extracted,
        "total_frames_rejected": total_rejected,
        "elapsed_minutes": round(elapsed / 60, 2),
        "config": {
            "base_fps": config.extraction.base_fps,
            "boost_fps": config.extraction.boost_fps,
            "blur_threshold": config.quality.blur_threshold,
            "auto_rotate_180": config.extraction.auto_rotate_180,
        },
        "videos": all_summaries,
    }
    
    manifest_path = os.path.join(PROJECT_ROOT, OUTPUT_BASE, "full_extraction_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info(f"Manifest saved → {manifest_path}")


if __name__ == "__main__":
    main()
