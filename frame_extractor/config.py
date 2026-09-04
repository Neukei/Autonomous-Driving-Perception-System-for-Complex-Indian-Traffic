import yaml
from dataclasses import dataclass, field, asdict
from typing import List, Tuple
from pathlib import Path

@dataclass
class ExtractionConfig:
    """Configuration for frame extraction rates and output formatting."""
    base_fps: float = 0.5
    boost_fps: float = 3.0
    output_format: str = "jpg"
    output_quality: int = 95
    target_resolution: Tuple[int, int] = (1920, 1080)
    auto_rotate_180: bool = True  # Dashcam mounted inverted


@dataclass
class SceneChangeConfig:
    """Configuration for scene change detection."""
    ssim_threshold: float = 0.7
    histogram_threshold: float = 0.4
    min_interval_ms: int = 300


@dataclass
class QualityConfig:
    """Configuration for filtering low-quality frames."""
    blur_threshold: float = 100.0
    phash_distance: int = 5
    min_brightness: int = 30
    max_brightness: int = 240
    min_contrast: float = 20.0
    hash_size: int = 16
    distance_threshold: int = 5
    enable_dedup: bool = True


@dataclass
class ComplexityConfig:
    """Configuration for detecting complex scenes that need higher extraction rate."""
    edge_density_threshold: float = 0.15
    flow_magnitude_threshold: float = 5.0
    complexity_window: int = 10


@dataclass
class PipelineConfig:
    """Main pipeline configuration combining all module settings."""
    input_dir: str = ""
    output_dir: str = ""
    video_extensions: List[str] = field(default_factory=lambda: [".mp4", ".avi", ".mov", ".mkv"])
    num_workers: int = 4
    log_level: str = "INFO"
    
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    scene_change: SceneChangeConfig = field(default_factory=SceneChangeConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    complexity: ComplexityConfig = field(default_factory=ComplexityConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load configuration from a YAML file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            
        if not data:
            return cls()

        def _dict_to_dataclass(dataclass_type, data_dict):
            if not data_dict:
                return dataclass_type()
            # Filter valid keys
            valid_keys = dataclass_type.__dataclass_fields__.keys()
            filtered = {k: v for k, v in data_dict.items() if k in valid_keys}
            return dataclass_type(**filtered)

        config = cls()
        
        # Base attributes
        for key in ["input_dir", "output_dir", "video_extensions", "num_workers", "log_level"]:
            if key in data:
                setattr(config, key, data[key])
                
        # Nested dataclasses
        if "extraction" in data:
            config.extraction = _dict_to_dataclass(ExtractionConfig, data["extraction"])
        if "scene_change" in data:
            config.scene_change = _dict_to_dataclass(SceneChangeConfig, data["scene_change"])
        if "quality" in data:
            config.quality = _dict_to_dataclass(QualityConfig, data["quality"])
        if "complexity" in data:
            config.complexity = _dict_to_dataclass(ComplexityConfig, data["complexity"])
            
        return config

    def to_yaml(self, path: str) -> None:
        """Save configuration to a YAML file."""
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
