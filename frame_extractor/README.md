# Frame Extractor

A robust, intelligent video frame extraction pipeline designed for autonomous driving perception systems. It uses hybrid adaptive extraction with quality filtering to ensure only the highest quality, most relevant frames are kept.

## Features

- **Hybrid Adaptive Extraction**: Extracts frames based on both time intervals and scene activity.
- **Quality Filtering**: Automatically discards blurry, over/under-exposed, and low-contrast frames.
- **Perceptual Deduplication**: Uses average and difference hashing to detect and remove identical or nearly identical frames.
- **Multi-processing**: Utilizes multiple worker processes for fast, parallel extraction.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

The `main.py` script provides a command-line interface with three primary commands.

### 1. Extract
Run the full hybrid extraction pipeline on a directory of videos.

```bash
python main.py extract --input-dir ./videos --output-dir ./dataset --workers 8 --blur-threshold 120.0
```

### 2. Preview
Extract a small sample (first 30 seconds) to preview settings before running a large job.

```bash
python main.py preview --input-dir ./videos --output-dir ./preview --duration 30
```

### 3. Info
Show video information for a file or directory.

```bash
python main.py info --path ./videos/sample.mp4
```

## Configuration

You can provide a YAML configuration file via the `--config` flag to customize extraction parameters, thresholds, and logging behaviors.

## Output Structure

Extracted frames are saved in the specified output directory, along with a `manifest.json` file detailing the extraction run metrics (videos processed, frames kept, frames rejected, etc.).
