"""
Command Line Interface for the Frame Extractor project.

Usage:
    python -m frame_extractor.main extract --input-dir <path> --output-dir <path>
    python -m frame_extractor.main info --path <path>
    python -m frame_extractor.main preview --input-dir <path> --output-dir <path>
"""
import argparse
import sys
import os
import json
import time
import signal
import logging
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

from .config import PipelineConfig, ExtractionConfig, QualityConfig
from .extractors.hybrid import HybridAdaptiveExtractor, ExtractionResult
from .processors.quality_filter import QualityFilter
from .utils.video_utils import get_video_info
from .utils.file_utils import (
    ensure_dir,
    get_video_files,
    get_output_subdir,
    save_extraction_manifest,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
class ColorFormatter(logging.Formatter):
    """Custom formatter with ANSI colors for console output."""

    GREY = "\x1b[38;20m"
    GREEN = "\x1b[32;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    CYAN = "\x1b[36;20m"
    RESET = "\x1b[0m"

    FMT = "%(asctime)s | %(levelname)-8s | %(message)s"

    FORMATS = {
        logging.DEBUG: GREY + FMT + RESET,
        logging.INFO: GREEN + FMT + RESET,
        logging.WARNING: YELLOW + FMT + RESET,
        logging.ERROR: RED + FMT + RESET,
        logging.CRITICAL: BOLD_RED + FMT + RESET,
    }

    def format(self, record: logging.LogRecord) -> str:
        fmt = self.FORMATS.get(record.levelno, self.FMT)
        formatter = logging.Formatter(fmt, datefmt="%H:%M:%S")
        return formatter.format(record)


def _setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure root logger with coloured console handler."""
    logger = logging.getLogger("frame_extractor")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(ColorFormatter())
        logger.addHandler(ch)
    return logger


logger = _setup_logging()

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_shutdown_requested = False


def _signal_handler(sig, frame):
    global _shutdown_requested
    _shutdown_requested = True
    logger.warning("\nShutdown requested — finishing current video and exiting…")


signal.signal(signal.SIGINT, _signal_handler)

# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="frame_extractor",
        description="Intelligent frame extraction from driving videos",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ---- extract ----
    ext_p = subparsers.add_parser(
        "extract", help="Run the full hybrid extraction pipeline"
    )
    _add_common_args(ext_p)

    # ---- info ----
    info_p = subparsers.add_parser(
        "info", help="Show video metadata for a file or directory"
    )
    info_p.add_argument("--path", required=True, help="Path to video or directory")

    # ---- preview ----
    prev_p = subparsers.add_parser(
        "preview", help="Extract a small sample to preview settings"
    )
    _add_common_args(prev_p)
    prev_p.add_argument(
        "--duration", type=int, default=30,
        help="Duration in seconds to preview (default: 30)",
    )

    return parser.parse_args()


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by extract and preview commands."""
    parser.add_argument("--input-dir", required=True, help="Directory with videos")
    parser.add_argument("--output-dir", required=True, help="Where to save frames")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--base-fps", type=float, default=0.5)
    parser.add_argument("--boost-fps", type=float, default=3.0)
    parser.add_argument("--blur-threshold", type=float, default=100.0)
    parser.add_argument("--no-dedup", action="store_true", help="Disable dedup")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--video-ext", type=str, default=".mp4",
        help="Comma-separated extensions (e.g. .mp4,.avi)",
    )

# ---------------------------------------------------------------------------
# Build PipelineConfig from CLI args
# ---------------------------------------------------------------------------
def _build_config(args: argparse.Namespace) -> PipelineConfig:
    """Merge CLI flags with optional YAML config."""
    if args.config and Path(args.config).exists():
        config = PipelineConfig.from_yaml(args.config)
    else:
        config = PipelineConfig()

    config.input_dir = args.input_dir
    config.output_dir = args.output_dir
    config.num_workers = args.workers
    config.extraction.base_fps = args.base_fps
    config.extraction.boost_fps = args.boost_fps
    config.quality.blur_threshold = args.blur_threshold

    # Parse video extensions
    exts = [e.strip() for e in args.video_ext.split(",")]
    config.video_extensions = [e if e.startswith(".") else f".{e}" for e in exts]

    return config

# ---------------------------------------------------------------------------
# Single-video extraction (runs in the main process for now)
# ---------------------------------------------------------------------------
def _extract_single_video(
    video_path: Path,
    output_dir: Path,
    config: PipelineConfig,
    enable_dedup: bool = True,
) -> Dict[str, Any]:
    """Run hybrid extraction + quality filtering on one video.

    Returns:
        Summary dict with stats for this video.
    """
    video_name = video_path.stem
    vid_output = output_dir / video_name
    ensure_dir(str(vid_output))

    logger.info(f"▶ Processing: {video_path.name}")

    # Build the hybrid extractor
    extractor = HybridAdaptiveExtractor(config)

    # Build quality filter and attach it
    quality_filter = QualityFilter(config.quality)

    def _quality_gate(frame):
        """Closure used by the extractor to reject bad frames."""
        frame_id = f"{video_name}_{time.monotonic_ns()}"
        passed, _ = quality_filter.filter_frame(frame, frame_id)
        return passed

    if enable_dedup:
        extractor.quality_filters = [_quality_gate]

    # Run extraction
    result: ExtractionResult = extractor.extract(
        str(video_path), str(vid_output)
    )

    q_stats = quality_filter.get_stats()

    summary = {
        "video": str(video_path),
        "duration_sec": round(result.duration_sec, 2),
        "frames_extracted": result.frames_extracted,
        "frames_rejected": result.frames_rejected,
        "quality_stats": q_stats,
        "output_dir": str(vid_output),
        "output_count": len(result.output_paths),
    }
    logger.info(
        f"  ✓ {video_path.name}: {result.frames_extracted} extracted, "
        f"{result.frames_rejected} rejected "
        f"({result.duration_sec:.1f}s video)"
    )
    return summary

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def run_extract(args: argparse.Namespace, preview_duration: int = 0) -> None:
    """Execute the full extraction pipeline on all discovered videos."""
    start_time = time.time()
    config = _build_config(args)
    enable_dedup = not args.no_dedup

    input_dir = Path(config.input_dir)
    output_dir = Path(config.output_dir)

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    ensure_dir(str(output_dir))

    # Discover videos
    videos = sorted(get_video_files(str(input_dir), config.video_extensions))
    if not videos:
        logger.warning(f"No video files found in {input_dir} with extensions {config.video_extensions}")
        sys.exit(0)

    logger.info(f"Found {len(videos)} video(s) in {input_dir}")
    logger.info(f"Config: base_fps={config.extraction.base_fps}, "
                f"boost_fps={config.extraction.boost_fps}, "
                f"blur_thr={config.quality.blur_threshold}, "
                f"dedup={'ON' if enable_dedup else 'OFF'}")

    # Process each video sequentially (avoids OpenCV multi-process issues)
    all_summaries: List[Dict[str, Any]] = []
    total_extracted = 0
    total_rejected = 0

    for i, video_path in enumerate(videos, 1):
        if _shutdown_requested:
            logger.warning("Shutdown requested — stopping early.")
            break

        logger.info(f"[{i}/{len(videos)}] {video_path.name}")
        try:
            summary = _extract_single_video(
                video_path,
                output_dir,
                config,
                enable_dedup=enable_dedup,
            )
            all_summaries.append(summary)
            total_extracted += summary["frames_extracted"]
            total_rejected += summary["frames_rejected"]
        except Exception as e:
            logger.error(f"  ✗ Failed on {video_path.name}: {e}", exc_info=True)
            all_summaries.append({
                "video": str(video_path),
                "error": str(e),
            })

    elapsed = time.time() - start_time

    # ---- Summary ----
    logger.info("=" * 60)
    logger.info("EXTRACTION COMPLETE")
    logger.info(f"  Videos processed : {len(all_summaries)}")
    logger.info(f"  Frames extracted : {total_extracted}")
    logger.info(f"  Frames rejected  : {total_rejected}")
    logger.info(f"  Time elapsed     : {elapsed:.1f}s")
    logger.info(f"  Output directory : {output_dir}")
    logger.info("=" * 60)

    # Save manifest
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "base_fps": config.extraction.base_fps,
            "boost_fps": config.extraction.boost_fps,
            "blur_threshold": config.quality.blur_threshold,
            "dedup_enabled": enable_dedup,
            "video_extensions": config.video_extensions,
        },
        "total_videos": len(all_summaries),
        "total_frames_extracted": total_extracted,
        "total_frames_rejected": total_rejected,
        "elapsed_seconds": round(elapsed, 2),
        "videos": all_summaries,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info(f"Manifest saved → {manifest_path}")


def run_info(args: argparse.Namespace) -> None:
    """Display metadata for a video file or all videos in a directory."""
    target = Path(args.path)

    if target.is_file():
        _print_video_info(target)
    elif target.is_dir():
        videos = sorted(get_video_files(str(target), [".mp4", ".avi", ".mov", ".mkv"]))
        if not videos:
            logger.warning(f"No video files found in {target}")
            return
        logger.info(f"Found {len(videos)} video(s):\n")
        total_duration = 0.0
        for v in videos:
            info = _print_video_info(v)
            total_duration += info.get("duration_sec", 0)
        logger.info(f"\nTotal duration: {total_duration / 60:.1f} minutes across {len(videos)} videos")
    else:
        logger.error(f"Path not found: {target}")
        sys.exit(1)


def _print_video_info(video_path: Path) -> Dict[str, Any]:
    """Print and return video metadata."""
    info = get_video_info(str(video_path))
    if not info:
        logger.error(f"  Could not read: {video_path}")
        return {}

    mins, secs = divmod(info["duration_sec"], 60)
    logger.info(
        f"  {video_path.name:40s} | "
        f"{info['width']}×{info['height']} | "
        f"{info['fps']:.1f}fps | "
        f"{int(mins)}m{int(secs)}s | "
        f"{info['frame_count']} frames | "
        f"{info['codec']}"
    )
    return info

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Main CLI entry point."""
    try:
        args = parse_args()

        if args.command == "extract":
            run_extract(args)
        elif args.command == "preview":
            run_extract(args, preview_duration=getattr(args, "duration", 30))
        elif args.command == "info":
            run_info(args)
        else:
            logger.error("No command specified. Use --help for usage info.")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user. Exiting.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
