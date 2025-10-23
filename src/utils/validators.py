"""
Validation utilities for input/output files
"""

from pathlib import Path
from typing import Optional

from src.exceptions import ValidationError
from src.utils.logger import get_logger
from src.utils.ffmpeg_utils import probe_video


logger = get_logger(__name__)


def validate_file_exists(file_path: Path, file_type: str = "File") -> Path:
    """
    Validate that a file exists

    Args:
        file_path: Path to file
        file_type: Type of file for error message

    Returns:
        Path to file

    Raises:
        ValidationError: If file doesn't exist
    """
    if not file_path.exists():
        raise ValidationError(f"{file_type} not found: {file_path}")

    if not file_path.is_file():
        raise ValidationError(f"{file_type} is not a file: {file_path}")

    logger.debug(f"Validated {file_type}: {file_path}")
    return file_path


def validate_video_file(video_path: Path) -> Path:
    """
    Validate that a video file exists and is readable

    Args:
        video_path: Path to video file

    Returns:
        Path to video file

    Raises:
        ValidationError: If validation fails
    """
    validate_file_exists(video_path, "Video file")

    try:
        probe_data = probe_video(video_path)

        # Check for video stream
        has_video = any(
            stream.get('codec_type') == 'video'
            for stream in probe_data.get('streams', [])
        )

        if not has_video:
            raise ValidationError(f"No video stream found in {video_path}")

        logger.debug(f"Validated video file: {video_path}")
        return video_path

    except Exception as e:
        raise ValidationError(f"Failed to validate video file {video_path}: {e}")


def validate_audio_file(audio_path: Path) -> Path:
    """
    Validate that an audio file exists and is readable

    Args:
        audio_path: Path to audio file

    Returns:
        Path to audio file

    Raises:
        ValidationError: If validation fails
    """
    validate_file_exists(audio_path, "Audio file")

    try:
        probe_data = probe_video(audio_path)  # ffprobe works for audio too

        # Check for audio stream
        has_audio = any(
            stream.get('codec_type') == 'audio'
            for stream in probe_data.get('streams', [])
        )

        if not has_audio:
            raise ValidationError(f"No audio stream found in {audio_path}")

        logger.debug(f"Validated audio file: {audio_path}")
        return audio_path

    except Exception as e:
        raise ValidationError(f"Failed to validate audio file {audio_path}: {e}")


def validate_json_file(json_path: Path) -> Path:
    """
    Validate that a JSON file exists and is valid

    Args:
        json_path: Path to JSON file

    Returns:
        Path to JSON file

    Raises:
        ValidationError: If validation fails
    """
    import json

    validate_file_exists(json_path, "JSON file")

    try:
        with open(json_path, 'r') as f:
            json.load(f)

        logger.debug(f"Validated JSON file: {json_path}")
        return json_path

    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON in {json_path}: {e}")
    except Exception as e:
        raise ValidationError(f"Failed to validate JSON file {json_path}: {e}")


def validate_output_path(output_path: Path, overwrite: bool = False) -> Path:
    """
    Validate output path and create parent directories

    Args:
        output_path: Path to output file
        overwrite: Whether to allow overwriting existing files

    Returns:
        Path to output file

    Raises:
        ValidationError: If validation fails
    """
    if output_path.exists() and not overwrite:
        raise ValidationError(
            f"Output file already exists: {output_path}. "
            "Use --overwrite to replace it."
        )

    # Create parent directories
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Validated output path: {output_path}")
    return output_path
