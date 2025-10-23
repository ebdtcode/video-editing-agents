"""
FFmpeg utility functions with error handling
"""

import subprocess
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.exceptions import FFmpegError
from src.utils.logger import get_logger


logger = get_logger(__name__)


def run_ffmpeg_safe(
    cmd: List[str],
    error_msg: str,
    capture_output: bool = True,
    check: bool = True
) -> subprocess.CompletedProcess:
    """
    Run FFmpeg command with error handling

    Args:
        cmd: FFmpeg command as list of strings
        error_msg: Error message to display on failure
        capture_output: Whether to capture stdout/stderr
        check: Whether to raise exception on non-zero exit

    Returns:
        CompletedProcess instance

    Raises:
        FFmpegError: If command fails and check=True
    """
    try:
        logger.debug(f"Running FFmpeg: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            check=False
        )

        if check and result.returncode != 0:
            error_detail = f"{error_msg}\nCommand: {' '.join(cmd)}\nStderr: {result.stderr}"
            logger.error(error_detail)
            raise FFmpegError(error_detail)

        return result

    except FileNotFoundError:
        raise FFmpegError("FFmpeg not found. Please install FFmpeg and ensure it's in PATH")
    except Exception as e:
        raise FFmpegError(f"{error_msg}: {e}")


def probe_video(video_path: Path) -> Dict[str, Any]:
    """
    Probe video file to get metadata

    Args:
        video_path: Path to video file

    Returns:
        Dict containing video metadata

    Raises:
        FFmpegError: If probe fails
    """
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        str(video_path)
    ]

    result = run_ffmpeg_safe(
        cmd,
        f"Failed to probe video: {video_path}",
        capture_output=True,
        check=True
    )

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise FFmpegError(f"Failed to parse ffprobe output: {e}")


def probe_audio(audio_path: Path) -> Dict[str, Any]:
    """
    Probe audio file to get metadata

    Args:
        audio_path: Path to audio file

    Returns:
        Dict containing audio metadata

    Raises:
        FFmpegError: If probe fails
    """
    return probe_video(audio_path)  # Same function works for audio


def get_video_duration(video_path: Path) -> float:
    """
    Get video duration in seconds

    Args:
        video_path: Path to video file

    Returns:
        Duration in seconds

    Raises:
        FFmpegError: If probe fails
    """
    probe_data = probe_video(video_path)

    try:
        return float(probe_data['format']['duration'])
    except (KeyError, ValueError) as e:
        raise FFmpegError(f"Failed to get video duration: {e}")


def get_audio_duration(audio_path: Path) -> float:
    """
    Get audio duration in seconds

    Args:
        audio_path: Path to audio file

    Returns:
        Duration in seconds

    Raises:
        FFmpegError: If probe fails
    """
    return get_video_duration(audio_path)  # Same function works for audio


def get_video_fps(video_path: Path) -> float:
    """
    Get video frame rate (fps)

    Args:
        video_path: Path to video file

    Returns:
        Frame rate as float (e.g., 30.0, 60.0, 29.97)

    Raises:
        FFmpegError: If probe fails or fps cannot be determined
    """
    probe_data = probe_video(video_path)

    try:
        # Get video stream
        for stream in probe_data['streams']:
            if stream['codec_type'] == 'video':
                # Parse r_frame_rate (e.g., "30/1" or "60/1")
                fps_str = stream['r_frame_rate']
                num, den = fps_str.split('/')
                fps = float(num) / float(den)
                logger.debug(f"Detected video fps: {fps} from {fps_str}")
                return fps

        raise FFmpegError("No video stream found")
    except (KeyError, ValueError, ZeroDivisionError) as e:
        raise FFmpegError(f"Failed to get video fps: {e}")


def extract_audio_from_video(
    video_path: Path,
    output_path: Path,
    sample_rate: int = 16000,
    channels: int = 1
) -> Path:
    """
    Extract audio from video file

    Args:
        video_path: Path to input video
        output_path: Path to output audio file
        sample_rate: Audio sample rate
        channels: Number of audio channels

    Returns:
        Path to extracted audio file

    Raises:
        FFmpegError: If extraction fails
    """
    cmd = [
        'ffmpeg', '-y',
        '-i', str(video_path),
        '-vn',  # No video
        '-acodec', 'pcm_s16le',
        '-ar', str(sample_rate),
        '-ac', str(channels),
        str(output_path)
    ]

    run_ffmpeg_safe(
        cmd,
        f"Failed to extract audio from {video_path}",
        capture_output=True,
        check=True
    )

    logger.info(f"Extracted audio to {output_path}")
    return output_path


def cut_video_segment(
    video_path: Path,
    output_path: Path,
    start_time: float,
    end_time: float,
    include_audio: bool = False
) -> Path:
    """
    Cut a video segment while preserving original frame rate

    Args:
        video_path: Path to input video
        output_path: Path to output segment
        start_time: Start time in seconds
        end_time: End time in seconds
        include_audio: Whether to include audio

    Returns:
        Path to output segment

    Raises:
        FFmpegError: If cutting fails
    """
    # Detect and preserve original frame rate
    original_fps = get_video_fps(video_path)

    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_time),
        '-to', str(end_time),
        '-i', str(video_path),
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-r', str(original_fps)  # Preserve original fps
    ]

    if not include_audio:
        cmd.append('-an')
    else:
        cmd.extend(['-c:a', 'aac', '-b:a', '128k'])

    cmd.append(str(output_path))

    run_ffmpeg_safe(
        cmd,
        f"Failed to cut video segment {start_time}-{end_time}",
        capture_output=True,
        check=True
    )

    logger.debug(f"Cut video segment {start_time}-{end_time} to {output_path} at {original_fps} fps")
    return output_path


def concatenate_videos(
    video_list: List[Path],
    output_path: Path,
    concat_file: Optional[Path] = None
) -> Path:
    """
    Concatenate multiple video files while preserving frame rate

    Args:
        video_list: List of video file paths in order
        output_path: Path to output video
        concat_file: Optional path to concat list file

    Returns:
        Path to concatenated video

    Raises:
        FFmpegError: If concatenation fails
    """
    if not video_list:
        raise FFmpegError("Cannot concatenate empty video list")

    # Detect fps from first video segment
    original_fps = get_video_fps(video_list[0])

    # Create concat list file
    if concat_file is None:
        concat_file = output_path.parent / f"{output_path.stem}_concat.txt"

    with open(concat_file, 'w') as f:
        for video_path in video_list:
            f.write(f"file '{video_path.resolve()}'\n")

    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', str(concat_file),
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-r', str(original_fps),  # Preserve original fps
        '-c:a', 'aac',
        '-b:a', '128k',
        '-movflags', '+faststart',
        str(output_path)
    ]

    run_ffmpeg_safe(
        cmd,
        "Failed to concatenate videos",
        capture_output=True,
        check=True
    )

    logger.info(f"Concatenated {len(video_list)} videos to {output_path} at {original_fps} fps")
    return output_path


def merge_video_audio(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    video_codec: str = "copy",
    audio_codec: str = "aac",
    audio_bitrate: str = "192k"
) -> Path:
    """
    Merge video and audio files

    Args:
        video_path: Path to input video
        audio_path: Path to input audio
        output_path: Path to output file
        video_codec: Video codec (default: copy)
        audio_codec: Audio codec (default: aac)
        audio_bitrate: Audio bitrate (default: 192k)

    Returns:
        Path to merged output

    Raises:
        FFmpegError: If merge fails
    """
    cmd = [
        'ffmpeg', '-y',
        '-i', str(video_path),
        '-i', str(audio_path),
        '-c:v', video_codec,
        '-c:a', audio_codec,
        '-b:a', audio_bitrate,
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-shortest',
        str(output_path)
    ]

    run_ffmpeg_safe(
        cmd,
        "Failed to merge video and audio",
        capture_output=True,
        check=True
    )

    logger.info(f"Merged video and audio to {output_path}")
    return output_path


def retime_video(
    video_path: Path,
    output_path: Path,
    speed_factor: float
) -> Path:
    """
    Retime video to match a specific speed factor while preserving frame rate

    Args:
        video_path: Path to input video
        output_path: Path to output video
        speed_factor: Speed multiplier (>1 = faster, <1 = slower)

    Returns:
        Path to retimed video

    Raises:
        FFmpegError: If retiming fails
    """
    # Calculate PTS (presentation timestamp) factor
    # For slower video (speed < 1), PTS needs to be larger
    pts_factor = 1.0 / speed_factor

    # Detect and preserve original frame rate
    original_fps = get_video_fps(video_path)

    cmd = [
        'ffmpeg', '-y',
        '-i', str(video_path),
        '-filter:v', f'setpts={pts_factor}*PTS',  # Only retime, don't change fps
        '-r', str(original_fps),  # Preserve original fps
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-an',  # Remove audio
        str(output_path)
    ]

    run_ffmpeg_safe(
        cmd,
        f"Failed to retime video with speed factor {speed_factor}",
        capture_output=True,
        check=True
    )

    logger.debug(f"Retimed video with speed factor {speed_factor} at {original_fps} fps")
    return output_path
