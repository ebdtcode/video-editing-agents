"""
Transcription Agent - Handles speech-to-text operations
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from src.config import TranscriptionConfig
from src.exceptions import TranscriptionError
from src.utils.logger import get_logger
from src.utils.validators import validate_audio_file


logger = get_logger(__name__)


@dataclass
class Word:
    """Represents a word with timing information"""
    word: str
    start: float
    end: float
    confidence: float = 1.0


@dataclass
class Segment:
    """Represents a transcript segment"""
    text: str
    start: float
    end: float
    words: List[Word]


class TranscriptionAgent:
    """Handles audio transcription using WhisperX"""

    def __init__(self, config: TranscriptionConfig):
        """
        Initialize transcription agent

        Args:
            config: Transcription configuration
        """
        self.config = config
        self.logger = get_logger(self.__class__.__name__)

    def transcribe_audio(
        self,
        audio_path: Path,
        output_dir: Optional[Path] = None
    ) -> Tuple[List[Segment], Path]:
        """
        Transcribe audio file using WhisperX

        Args:
            audio_path: Path to audio file
            output_dir: Optional output directory for transcription files

        Returns:
            Tuple of (segments list, path to JSON output)

        Raises:
            TranscriptionError: If transcription fails
        """
        try:
            # Validate input
            validate_audio_file(audio_path)

            # Set output directory
            if output_dir is None:
                output_dir = audio_path.parent / "transcriptions"

            output_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"Starting transcription of {audio_path}")

            # Build WhisperX command
            cmd = self._build_whisperx_command(audio_path, output_dir)

            # Run transcription
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode != 0:
                raise TranscriptionError(
                    f"WhisperX failed: {result.stderr}\n"
                    f"Command: {' '.join(cmd)}"
                )

            # Find output JSON file
            json_files = list(output_dir.glob(f"{audio_path.stem}*.json"))
            if not json_files:
                raise TranscriptionError(
                    f"No JSON output found in {output_dir}"
                )

            json_path = json_files[0]
            self.logger.info(f"Transcription completed: {json_path}")

            # Load and parse transcription
            segments = self._parse_whisperx_output(json_path)

            return segments, json_path

        except subprocess.CalledProcessError as e:
            raise TranscriptionError(f"Transcription process failed: {e}")
        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {e}")

    def load_transcription(self, json_path: Path) -> List[Segment]:
        """
        Load transcription from existing JSON file

        Args:
            json_path: Path to WhisperX JSON output

        Returns:
            List of segments

        Raises:
            TranscriptionError: If loading fails
        """
        try:
            self.logger.info(f"Loading transcription from {json_path}")
            return self._parse_whisperx_output(json_path)
        except Exception as e:
            raise TranscriptionError(f"Failed to load transcription: {e}")

    def _build_whisperx_command(
        self,
        audio_path: Path,
        output_dir: Path
    ) -> List[str]:
        """Build WhisperX command with configuration"""
        cmd = [
            'whisperx',
            str(audio_path),
            '--model', self.config.model,
            '--output_dir', str(output_dir),
            '--output_format', 'json'
        ]

        # Add language if specified
        if self.config.language and self.config.language != 'auto':
            cmd.extend(['--language', self.config.language])

        # Add device
        if self.config.device:
            cmd.extend(['--device', self.config.device])

        # Add compute type (required for both CPU and GPU)
        if self.config.compute_type:
            cmd.extend(['--compute_type', self.config.compute_type])

        return cmd

    def _parse_whisperx_output(self, json_path: Path) -> List[Segment]:
        """
        Parse WhisperX JSON output

        Args:
            json_path: Path to JSON file

        Returns:
            List of Segment objects

        Raises:
            TranscriptionError: If parsing fails
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            segments = []

            for seg_data in data.get('segments', []):
                # Parse words
                words = []
                for word_data in seg_data.get('words', []):
                    if 'start' in word_data and 'end' in word_data:
                        words.append(Word(
                            word=word_data.get('word', '').strip(),
                            start=float(word_data['start']),
                            end=float(word_data['end']),
                            confidence=float(word_data.get('score', 1.0))
                        ))

                # Create segment
                if words:
                    segment = Segment(
                        text=seg_data.get('text', '').strip(),
                        start=float(seg_data.get('start', words[0].start)),
                        end=float(seg_data.get('end', words[-1].end)),
                        words=words
                    )
                    segments.append(segment)

            self.logger.info(f"Parsed {len(segments)} segments with {sum(len(s.words) for s in segments)} words")
            return segments

        except json.JSONDecodeError as e:
            raise TranscriptionError(f"Invalid JSON in {json_path}: {e}")
        except Exception as e:
            raise TranscriptionError(f"Failed to parse transcription: {e}")

    def export_to_srt(
        self,
        segments: List[Segment],
        output_path: Path
    ) -> Path:
        """
        Export segments to SRT subtitle format

        Args:
            segments: List of segments
            output_path: Path to output SRT file

        Returns:
            Path to SRT file
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, segment in enumerate(segments, 1):
                    # SRT format:
                    # 1
                    # 00:00:00,000 --> 00:00:02,000
                    # Text here
                    start_time = self._format_srt_timestamp(segment.start)
                    end_time = self._format_srt_timestamp(segment.end)

                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{segment.text}\n\n")

            self.logger.info(f"Exported SRT to {output_path}")
            return output_path

        except Exception as e:
            raise TranscriptionError(f"Failed to export SRT: {e}")

    def _format_srt_timestamp(self, seconds: float) -> str:
        """Format timestamp for SRT (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def validate_transcription(self, segments: List[Segment]) -> Dict[str, any]:
        """
        Validate transcription quality

        Args:
            segments: List of segments to validate

        Returns:
            Dict with validation metrics
        """
        if not segments:
            return {
                'valid': False,
                'error': 'No segments found',
                'metrics': {}
            }

        total_words = sum(len(s.words) for s in segments)
        avg_confidence = sum(
            w.confidence for s in segments for w in s.words
        ) / total_words if total_words > 0 else 0

        total_duration = sum(s.end - s.start for s in segments)

        metrics = {
            'total_segments': len(segments),
            'total_words': total_words,
            'average_confidence': avg_confidence,
            'total_duration': total_duration,
            'words_per_minute': (total_words / total_duration * 60) if total_duration > 0 else 0
        }

        # Basic validation rules
        valid = (
            len(segments) > 0 and
            total_words > 0 and
            avg_confidence > 0.5  # Minimum confidence threshold
        )

        return {
            'valid': valid,
            'error': None if valid else 'Low confidence or no words',
            'metrics': metrics
        }
