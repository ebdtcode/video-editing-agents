"""
TTS Generation Agent - Handles text-to-speech generation
"""

from pathlib import Path
from typing import Optional, Dict, List
from abc import ABC, abstractmethod
import subprocess
import warnings

# Suppress external library warnings globally for this module
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="torchaudio")

from src.config import TTSConfig
from src.agents.content_analysis_agent import CleanSegment
from src.exceptions import TTSGenerationError
from src.utils.logger import get_logger
from src.utils.ffmpeg_utils import get_audio_duration


logger = get_logger(__name__)


class TTSBackend(ABC):
    """Abstract base class for TTS backends"""

    @abstractmethod
    def generate(
        self,
        text: str,
        output_path: Path,
        reference_audio: Optional[Path] = None
    ) -> Path:
        """Generate speech from text"""
        pass

    @abstractmethod
    def get_sample_rate(self) -> int:
        """Get sample rate of generated audio"""
        pass


class ChatterboxBackend(TTSBackend):
    """Chatterbox TTS backend"""

    def __init__(self, config: TTSConfig):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)

        try:
            # Import chatterbox dynamically
            from chatterbox.tts import ChatterboxTTS
            import warnings
            import torch

            # Suppress transformers attention warnings from Chatterbox internals
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

                # Detect device
                device = self._detect_device()

                # For CUDA, verify it's actually available before initializing model
                if device == "cuda":
                    if not torch.cuda.is_available():
                        self.logger.warning("CUDA requested but not available, falling back to CPU")
                        device = "cpu"
                    else:
                        self.logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                        self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

                # Use from_pretrained factory method (actual Chatterbox API)
                self.logger.info(f"Loading Chatterbox TTS model on {device}...")
                self.tts = ChatterboxTTS.from_pretrained(device=device)
                self.logger.info(f"Initialized Chatterbox TTS on {device}")

        except ImportError:
            raise TTSGenerationError(
                "Chatterbox TTS not installed. Install with: pip install chatterbox-tts"
            )
        except Exception as e:
            raise TTSGenerationError(f"Failed to initialize Chatterbox TTS: {e}")

    def _detect_device(self) -> str:
        """Detect CUDA availability and handle multi-process GPU access"""
        import os

        if self.config.device == "cpu":
            return "cpu"

        # Check for CUDA
        result = subprocess.run(
            "nvidia-smi",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        if result.returncode == 0:
            # Enable CUDA device management for multi-process scenarios
            # This helps prevent GPU initialization hangs
            os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')
            os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

            self.logger.info("CUDA detected, using GPU")
            return "cuda"
        else:
            self.logger.info("CUDA not available, using CPU")
            return "cpu"

    def generate(
        self,
        text: str,
        output_path: Path,
        reference_audio: Optional[Path] = None
    ) -> Path:
        """Generate speech using Chatterbox"""
        try:
            import torchaudio
            import warnings

            # Suppress transformers warnings during generation
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
                warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

                # Generate audio with optional voice cloning
                # If reference_audio is provided, use it (orchestrator handles voice_mode logic)
                if reference_audio:
                    self.logger.debug(f"Generating with voice cloning: {reference_audio}")
                    wav = self.tts.generate(text, audio_prompt_path=str(reference_audio))
                else:
                    self.logger.debug("Generating with default voice (no reference)")
                    wav = self.tts.generate(text)

            # CRITICAL: Check for None return (can happen with thread-safety issues)
            if wav is None:
                self.logger.error(
                    f"TTS returned None for text (len={len(text)}): '{text[:100]}...'"
                )
                raise TTSGenerationError(
                    f"Chatterbox TTS returned None. This may indicate:\n"
                    f"  1. Thread-safety issue (parallel execution)\n"
                    f"  2. Model inference failure\n"
                    f"  3. Text normalization issue\n"
                    f"Text preview: '{text[:100]}'"
                )

            # Validate tensor shape
            if not hasattr(wav, 'shape'):
                raise TTSGenerationError(
                    f"TTS returned invalid type: {type(wav)}. Expected torch.Tensor"
                )

            # Save audio (wav is already a tensor from Chatterbox)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(
                str(output_path),
                wav,
                self.tts.sr  # Use .sr property (actual Chatterbox API)
            )

            self.logger.debug(f"Generated TTS audio: {output_path}")
            return output_path

        except Exception as e:
            raise TTSGenerationError(f"Failed to generate speech: {e}")

    def get_sample_rate(self) -> int:
        """Get sample rate"""
        return getattr(self.tts, 'sr', self.config.sample_rate)


class CoquiBackend(TTSBackend):
    """Coqui TTS backend (placeholder for future implementation)"""

    def __init__(self, config: TTSConfig):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        raise TTSGenerationError("Coqui TTS backend not yet implemented")

    def generate(
        self,
        text: str,
        output_path: Path,
        reference_audio: Optional[Path] = None
    ) -> Path:
        raise NotImplementedError

    def get_sample_rate(self) -> int:
        return self.config.sample_rate


class TTSGenerationAgent:
    """Handles TTS generation with multiple backend support"""

    def __init__(self, config: TTSConfig):
        """
        Initialize TTS generation agent

        Args:
            config: TTS configuration
        """
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.backend = self._initialize_backend()

    def _initialize_backend(self) -> TTSBackend:
        """Initialize TTS backend based on configuration"""
        backend_map = {
            'chatterbox': ChatterboxBackend,
            'coqui': CoquiBackend,
        }

        backend_class = backend_map.get(self.config.backend.lower())

        if backend_class is None:
            raise TTSGenerationError(
                f"Unknown TTS backend: {self.config.backend}. "
                f"Available: {list(backend_map.keys())}"
            )

        try:
            return backend_class(self.config)
        except Exception as e:
            raise TTSGenerationError(f"Failed to initialize TTS backend: {e}")

    def generate_for_segment(
        self,
        segment: CleanSegment,
        output_dir: Path,
        reference_audio: Optional[Path] = None
    ) -> Path:
        """
        Generate TTS audio for a single segment

        Args:
            segment: Clean segment to generate audio for
            output_dir: Output directory for audio files
            reference_audio: Optional reference audio for voice cloning

        Returns:
            Path to generated audio file

        Raises:
            TTSGenerationError: If generation fails
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate output filename
            audio_file = output_dir / f"tts_{segment.segment_id}.wav"

            # Generate speech (orchestrator handles voice_mode logic)
            self.backend.generate(
                text=segment.text,
                output_path=audio_file,
                reference_audio=reference_audio
            )

            # Normalize audio if configured
            if self.config.normalize:
                audio_file = self._normalize_audio(audio_file)

            return audio_file

        except Exception as e:
            raise TTSGenerationError(
                f"Failed to generate TTS for segment {segment.segment_id}: {e}"
            )

    def generate_batch(
        self,
        segments: List[CleanSegment],
        output_dir: Path,
        reference_audio: Optional[Path] = None
    ) -> Dict[str, Path]:
        """
        Generate TTS audio for multiple segments

        Args:
            segments: List of clean segments
            output_dir: Output directory
            reference_audio: Optional reference audio for voice cloning

        Returns:
            Dict mapping segment IDs to audio file paths

        Raises:
            TTSGenerationError: If generation fails
        """
        self.logger.info(f"Generating TTS for {len(segments)} segments")

        results = {}
        for i, segment in enumerate(segments):
            try:
                audio_path = self.generate_for_segment(
                    segment,
                    output_dir,
                    reference_audio
                )
                results[segment.segment_id] = audio_path

                if (i + 1) % 10 == 0:
                    self.logger.info(f"Generated {i + 1}/{len(segments)} TTS files")

            except Exception as e:
                self.logger.error(f"Failed to generate TTS for {segment.segment_id}: {e}")
                raise

        self.logger.info(f"Completed TTS generation for {len(results)} segments")
        return results

    def _normalize_audio(self, audio_path: Path) -> Path:
        """
        Normalize audio volume

        Args:
            audio_path: Path to audio file

        Returns:
            Path to normalized audio file
        """
        try:
            normalized_path = audio_path.parent / f"{audio_path.stem}_normalized.wav"

            # Use FFmpeg to normalize audio
            cmd = [
                'ffmpeg', '-y',
                '-i', str(audio_path),
                '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11',
                str(normalized_path)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode != 0:
                self.logger.warning(
                    f"Audio normalization failed, using original: {result.stderr}"
                )
                return audio_path

            # Replace original with normalized
            normalized_path.replace(audio_path)
            return audio_path

        except Exception as e:
            self.logger.warning(f"Audio normalization failed: {e}")
            return audio_path

    def concatenate_audio_files(
        self,
        audio_files: List[Path],
        output_path: Path
    ) -> Path:
        """
        Concatenate multiple audio files

        Args:
            audio_files: List of audio file paths in order
            output_path: Path to output concatenated file

        Returns:
            Path to concatenated audio

        Raises:
            TTSGenerationError: If concatenation fails
        """
        try:
            self.logger.info(f"Concatenating {len(audio_files)} audio files")

            # Create concat list for FFmpeg
            concat_file = output_path.parent / "audio_concat_list.txt"
            with open(concat_file, 'w') as f:
                for audio_file in audio_files:
                    f.write(f"file '{audio_file.resolve()}'\n")

            # Use FFmpeg to concatenate
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file),
                '-c', 'copy',
                str(output_path)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode != 0:
                raise TTSGenerationError(
                    f"Audio concatenation failed: {result.stderr}"
                )

            # Clean up concat list
            concat_file.unlink(missing_ok=True)

            self.logger.info(f"Concatenated audio saved to {output_path}")
            return output_path

        except Exception as e:
            raise TTSGenerationError(f"Failed to concatenate audio: {e}")

    def extract_voice_profile(
        self,
        audio_path: Path,
        start: float,
        end: float,
        output_path: Path
    ) -> Path:
        """
        Extract a segment from audio to use as voice reference

        Args:
            audio_path: Path to source audio
            start: Start time in seconds
            end: End time in seconds
            output_path: Path to output reference audio

        Returns:
            Path to extracted reference audio

        Raises:
            TTSGenerationError: If extraction fails
        """
        try:
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start),
                '-to', str(end),
                '-i', str(audio_path),
                '-ar', str(self.backend.get_sample_rate()),
                '-ac', '1',
                str(output_path)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode != 0:
                raise TTSGenerationError(
                    f"Voice profile extraction failed: {result.stderr}"
                )

            self.logger.info(f"Extracted voice profile: {output_path}")
            return output_path

        except Exception as e:
            raise TTSGenerationError(f"Failed to extract voice profile: {e}")

    def validate_audio_quality(self, audio_path: Path) -> Dict[str, any]:
        """
        Validate generated audio quality

        Args:
            audio_path: Path to audio file

        Returns:
            Dict with quality metrics
        """
        try:
            duration = get_audio_duration(audio_path)

            # Basic validation
            if duration <= 0:
                return {
                    'valid': False,
                    'error': 'Invalid duration',
                    'metrics': {}
                }

            metrics = {
                'duration': duration,
                'sample_rate': self.backend.get_sample_rate(),
                'file_size': audio_path.stat().st_size
            }

            return {
                'valid': True,
                'error': None,
                'metrics': metrics
            }

        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'metrics': {}
            }
