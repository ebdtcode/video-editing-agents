"""
Configuration management for video processing pipeline
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
import yaml
import psutil

from src.exceptions import ConfigurationError


@dataclass
class FillerConfig:
    """Configuration for filler word detection"""
    mode: str = "rule-based"  # 'rule-based' or 'ml'
    words: List[str] = field(default_factory=lambda: ["uh", "um", "ah", "like", "you know"])
    confidence_threshold: float = 0.8


@dataclass
class TranscriptionConfig:
    """Configuration for transcription"""
    backend: str = "whisperx"
    model: str = "large-v2"
    language: str = "auto"
    device: str = "cuda"
    compute_type: str = "float16"


@dataclass
class TTSConfig:
    """Configuration for TTS generation"""
    backend: str = "chatterbox"  # 'chatterbox', 'coqui', 'elevenlabs'
    voice_mode: str = "auto"  # 'default', 'custom', 'auto'
    voice_reference: Optional[str] = None  # Path to custom voice reference audio file
    sample_rate: int = 44100
    normalize: bool = True
    device: str = "cuda"

    # Deprecated fields (kept for backward compatibility)
    voice_cloning: Optional[bool] = None

    def __post_init__(self):
        """Handle backward compatibility and validate voice_mode"""
        # Backward compatibility: convert old voice_cloning to voice_mode
        if self.voice_cloning is not None:
            if self.voice_cloning:
                # voice_cloning: true -> auto or custom depending on voice_reference
                if self.voice_reference:
                    self.voice_mode = "custom"
                else:
                    self.voice_mode = "auto"
            else:
                # voice_cloning: false -> default
                self.voice_mode = "default"

        # Validate voice_mode
        if self.voice_mode not in ["default", "custom", "auto"]:
            raise ConfigurationError(
                f"Invalid voice_mode: {self.voice_mode}. "
                "Must be 'default', 'custom', or 'auto'"
            )

        # Validate that custom mode has a reference path
        if self.voice_mode == "custom" and not self.voice_reference:
            raise ConfigurationError(
                "voice_mode 'custom' requires voice_reference path to be specified"
            )


@dataclass
class TransitionConfig:
    """Configuration for video transitions"""
    enabled: bool = True
    type: str = "crossfade"  # 'crossfade', 'dissolve', 'none'
    duration: float = 0.3


@dataclass
class VideoConfig:
    """Configuration for video processing"""
    sync_mode: str = "retime"  # 'retime' or 'stretch'
    quality: str = "high"
    codec: str = "h264"
    bitrate: str = "5M"
    transitions: TransitionConfig = field(default_factory=TransitionConfig)


@dataclass
class OutputConfig:
    """Configuration for output"""
    format: str = "mp4"
    temp_dir: str = "./temp_segments"
    keep_intermediates: bool = False
    checkpoint: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging"""
    level: str = "INFO"
    file: str = "video_processing.log"


@dataclass
class ProcessingConfig:
    """Main configuration class"""
    min_segment_duration: float = 0.5
    max_segment_duration: float = 30.0
    max_workers: int = 0  # 0 = auto

    fillers: FillerConfig = field(default_factory=FillerConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self):
        """Validate and adjust configuration after initialization"""
        # Auto-calculate optimal workers if set to 0
        if self.max_workers == 0:
            self.max_workers = self._calculate_optimal_workers()

        # Validate configuration
        self._validate()

    def _calculate_optimal_workers(self) -> int:
        """Calculate optimal number of workers based on system resources"""
        cpu_count = psutil.cpu_count(logical=False) or 4
        available_memory = psutil.virtual_memory().available

        # Reserve 2GB per worker for TTS model
        memory_workers = max(1, available_memory // (2 * 1024**3))

        return min(cpu_count, memory_workers, 8)

    def _validate(self):
        """Validate configuration values"""
        if self.min_segment_duration <= 0:
            raise ConfigurationError("min_segment_duration must be positive")

        if self.max_segment_duration <= self.min_segment_duration:
            raise ConfigurationError("max_segment_duration must be greater than min_segment_duration")

        if self.fillers.mode not in ["rule-based", "ml"]:
            raise ConfigurationError(f"Invalid filler mode: {self.fillers.mode}")

        if self.video.sync_mode not in ["retime", "stretch"]:
            raise ConfigurationError(f"Invalid sync mode: {self.video.sync_mode}")

    @classmethod
    def from_yaml(cls, path: str) -> 'ProcessingConfig':
        """Load configuration from YAML file"""
        config_path = Path(path)
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {path}")

        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)

            # Parse nested configs
            config_dict = {
                'min_segment_duration': data.get('processing', {}).get('min_segment_duration', 0.5),
                'max_segment_duration': data.get('processing', {}).get('max_segment_duration', 30.0),
                'max_workers': data.get('processing', {}).get('max_workers', 0),
                'fillers': FillerConfig(**data.get('fillers', {})),
                'transcription': TranscriptionConfig(**data.get('transcription', {})),
                'tts': TTSConfig(**data.get('tts', {})),
                'video': VideoConfig(
                    **{k: v for k, v in data.get('video', {}).items() if k != 'transitions'},
                    transitions=TransitionConfig(**data.get('video', {}).get('transitions', {}))
                ),
                'output': OutputConfig(**data.get('output', {})),
                'logging': LoggingConfig(**data.get('logging', {}))
            }

            return cls(**config_dict)
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")

    def to_yaml(self, path: str):
        """Save configuration to YAML file"""
        from dataclasses import asdict

        config_dict = asdict(self)

        # Reorganize for better YAML structure
        yaml_dict = {
            'processing': {
                'min_segment_duration': config_dict['min_segment_duration'],
                'max_segment_duration': config_dict['max_segment_duration'],
                'max_workers': config_dict['max_workers']
            },
            'fillers': config_dict['fillers'],
            'transcription': config_dict['transcription'],
            'tts': config_dict['tts'],
            'video': config_dict['video'],
            'output': config_dict['output'],
            'logging': config_dict['logging']
        }

        with open(path, 'w') as f:
            yaml.dump(yaml_dict, f, default_flow_style=False, sort_keys=False)
