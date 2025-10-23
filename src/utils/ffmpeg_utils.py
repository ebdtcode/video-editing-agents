"""
FFmpeg utility functions with error handling
"""

import subprocess
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.exceptions import FFmpegError
from src.utils.logger import get_logger
from src.utils.gpu_utils import get_gpu_capabilities, get_optimal_encoder


logger = get_logger(__name__)

# Cache GPU capabilities
_gpu_caps = None


def _get_gpu_caps():
    """Get cached GPU capabilities"""
    global _gpu_caps
    if _gpu_caps is None:
        _gpu_caps = get_gpu_capabilities()
    return _gpu_caps


def _get_video_codec(use_gpu: bool = True, codec: str = 'h264') -> str:
    """
    Get optimal video codec based on GPU availability

    Args:
        use_gpu: Whether to prefer GPU encoding
        codec: Desired codec (h264 or hevc)

    Returns:
        Encoder name (h264_nvenc or libx264)
    """
    if use_gpu:
        caps = _get_gpu_caps()
        return get_optimal_encoder(caps, codec=codec, fallback='libx264')
    return 'libx264'


def _add_hwaccel_flags(cmd: List[str], hwaccel: str = 'auto') -> List[str]:
    """
    Add hardware acceleration flags to FFmpeg command

    Args:
        cmd: FFmpeg command list
        hwaccel: Hardware acceleration mode ('auto', 'cuda', 'none')

    Returns:
        Command with hwaccel flags prepended if applicable
    """
    if hwaccel == 'none':
        return cmd

    caps = _get_gpu_caps()

    if hwaccel == 'cuda' or (hwaccel == 'auto' and caps.has_cuda):
        # Add CUDA acceleration flags before input
        # Find position of first -i flag
        try:
            i_index = cmd.index('-i')
            # Insert hwaccel flags before -i
            hwaccel_flags = ['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda']
            return cmd[:i_index] + hwaccel_flags + cmd[i_index:]
        except ValueError:
            # No -i flag found, just return original
            return cmd

    return cmd


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
    include_audio: bool = False,
    use_gpu: bool = True,
    hwaccel: str = 'auto'
) -> Path:
    """
    Cut a video segment while preserving original frame rate

    Args:
        video_path: Path to input video
        output_path: Path to output segment
        start_time: Start time in seconds
        end_time: End time in seconds
        include_audio: Whether to include audio
        use_gpu: Use GPU encoding if available
        hwaccel: Hardware acceleration mode ('auto', 'cuda', 'none')

    Returns:
        Path to output segment

    Raises:
        FFmpegError: If cutting fails
    """
    # Detect and preserve original frame rate
    original_fps = get_video_fps(video_path)

    # Get optimal codec
    video_codec = _get_video_codec(use_gpu)

    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_time),
        '-to', str(end_time),
        '-i', str(video_path),
        '-c:v', video_codec,
        '-preset', 'fast' if video_codec == 'libx264' else 'p4',  # p4 = fast preset for NVENC
        '-crf', '23' if video_codec == 'libx264' else '28',  # NVENC uses different CRF scale
        '-r', str(original_fps)  # Preserve original fps
    ]

    # Add hwaccel if using GPU
    if use_gpu and hwaccel != 'none':
        cmd = _add_hwaccel_flags(cmd, hwaccel)

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

    logger.debug(f"Cut video segment {start_time}-{end_time} to {output_path} at {original_fps} fps using {video_codec}")
    return output_path


def concatenate_videos(
    video_list: List[Path],
    output_path: Path,
    concat_file: Optional[Path] = None,
    use_gpu: bool = True,
    hwaccel: str = 'auto'
) -> Path:
    """
    Concatenate multiple video files while preserving frame rate

    Args:
        video_list: List of video file paths in order
        output_path: Path to output video
        concat_file: Optional path to concat list file
        use_gpu: Use GPU encoding if available
        hwaccel: Hardware acceleration mode ('auto', 'cuda', 'none')

    Returns:
        Path to concatenated video

    Raises:
        FFmpegError: If concatenation fails
    """
    if not video_list:
        raise FFmpegError("Cannot concatenate empty video list")

    # Detect fps from first video segment
    original_fps = get_video_fps(video_list[0])

    # Get optimal codec
    video_codec = _get_video_codec(use_gpu)

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
        '-c:v', video_codec,
        '-preset', 'medium' if video_codec == 'libx264' else 'p5',  # p5 = medium preset for NVENC
        '-crf', '23' if video_codec == 'libx264' else '28',
        '-r', str(original_fps),  # Preserve original fps
        '-c:a', 'aac',
        '-b:a', '128k',
        '-movflags', '+faststart',
        str(output_path)
    ]

    # Add hwaccel if using GPU
    if use_gpu and hwaccel != 'none':
        cmd = _add_hwaccel_flags(cmd, hwaccel)

    run_ffmpeg_safe(
        cmd,
        "Failed to concatenate videos",
        capture_output=True,
        check=True
    )

    logger.info(f"Concatenated {len(video_list)} videos to {output_path} at {original_fps} fps using {video_codec}")
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
    speed_factor: float,
    use_gpu: bool = True,
    hwaccel: str = 'auto'
) -> Path:
    """
    Retime video to match a specific speed factor while preserving frame rate

    Args:
        video_path: Path to input video
        output_path: Path to output video
        speed_factor: Speed multiplier (>1 = faster, <1 = slower)
        use_gpu: Use GPU encoding if available
        hwaccel: Hardware acceleration mode ('auto', 'cuda', 'none')

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

    # Get optimal codec
    video_codec = _get_video_codec(use_gpu)

    cmd = [
        'ffmpeg', '-y',
        '-i', str(video_path),
        '-filter:v', f'setpts={pts_factor}*PTS',  # Only retime, don't change fps
        '-r', str(original_fps),  # Preserve original fps
        '-c:v', video_codec,
        '-preset', 'fast' if video_codec == 'libx264' else 'p4',
        '-crf', '23' if video_codec == 'libx264' else '28',
        '-an',  # Remove audio
        str(output_path)
    ]

    # Add hwaccel if using GPU
    if use_gpu and hwaccel != 'none':
        cmd = _add_hwaccel_flags(cmd, hwaccel)

    run_ffmpeg_safe(
        cmd,
        f"Failed to retime video with speed factor {speed_factor}",
        capture_output=True,
        check=True
    )

    logger.debug(f"Retimed video with speed factor {speed_factor} at {original_fps} fps using {video_codec}")
    return output_path


def add_subtitles_to_video(
    video_path: Path,
    subtitle_path: Path,
    output_path: Path,
    burn_in: bool = False,
    font_size: int = 24,
    font_color: str = "white",
    outline_color: str = "black",
    position: str = "bottom"
) -> Path:
    """
    Add subtitles to video (either burned-in or as soft subtitle track)

    Args:
        video_path: Path to input video
        subtitle_path: Path to SRT subtitle file
        output_path: Path to output video
        burn_in: If True, burn subtitles into video; if False, add as soft subtitle track
        font_size: Font size for burned-in subtitles
        font_color: Font color for burned-in subtitles
        outline_color: Outline color for burned-in subtitles
        position: Position of subtitles ('bottom' or 'top')

    Returns:
        Path to output video with subtitles

    Raises:
        FFmpegError: If adding subtitles fails
    """
    if burn_in:
        # Burn subtitles into video using subtitles filter
        # Calculate vertical position
        if position == "top":
            vertical_pos = "y=10"
        else:  # bottom
            vertical_pos = "y=h-th-10"

        # Escape subtitle path for filter
        subtitle_path_escaped = str(subtitle_path).replace('\\', '\\\\').replace(':', '\\:')

        cmd = [
            'ffmpeg', '-y',
            '-i', str(video_path),
            '-vf', f"subtitles='{subtitle_path_escaped}':force_style='FontSize={font_size},PrimaryColour=&H{_color_to_ass_hex(font_color)},OutlineColour=&H{_color_to_ass_hex(outline_color)},Outline=1,Alignment={(2 if position == 'bottom' else 8)}'",
            '-c:a', 'copy',
            str(output_path)
        ]
    else:
        # Add as soft subtitle track
        cmd = [
            'ffmpeg', '-y',
            '-i', str(video_path),
            '-i', str(subtitle_path),
            '-c:v', 'copy',
            '-c:a', 'copy',
            '-c:s', 'mov_text',  # Subtitle codec for MP4
            '-metadata:s:s:0', 'language=eng',
            '-metadata:s:s:0', 'title=English',
            str(output_path)
        ]

    run_ffmpeg_safe(
        cmd,
        f"Failed to add subtitles to video",
        capture_output=True,
        check=True
    )

    logger.info(f"Added subtitles ({'burned-in' if burn_in else 'soft'}) to {output_path}")
    return output_path


def _color_to_ass_hex(color: str) -> str:
    """
    Convert color name to ASS subtitle format hex (BGR format)

    Args:
        color: Color name ('white', 'black', 'yellow', etc.)

    Returns:
        Hex color in ASS format (BGR)
    """
    color_map = {
        'white': 'FFFFFF',
        'black': '000000',
        'yellow': '00FFFF',  # BGR
        'red': '0000FF',     # BGR
        'green': '00FF00',   # BGR
        'blue': 'FF0000',    # BGR
    }
    return color_map.get(color.lower(), 'FFFFFF')


def add_chapters_to_video(
    video_path: Path,
    chapter_file: Path,
    output_path: Path
) -> Path:
    """
    Add chapter metadata to video using FFmpeg metadata file

    Args:
        video_path: Path to input video
        chapter_file: Path to FFmpeg metadata file with chapters
        output_path: Path to output video

    Returns:
        Path to output video with chapters

    Raises:
        FFmpegError: If adding chapters fails
    """
    cmd = [
        'ffmpeg', '-y',
        '-i', str(video_path),
        '-i', str(chapter_file),
        '-map_metadata', '1',
        '-c', 'copy',
        str(output_path)
    ]

    run_ffmpeg_safe(
        cmd,
        f"Failed to add chapters to video",
        capture_output=True,
        check=True
    )

    logger.info(f"Added chapters to {output_path}")
    return output_path


def generate_chapter_metadata(
    chapters: List[Dict[str, any]],
    output_path: Path
) -> Path:
    """
    Generate FFmpeg metadata file with chapter information

    Args:
        chapters: List of chapter dictionaries with 'title', 'start', 'end' keys
        output_path: Path to output metadata file

    Returns:
        Path to generated metadata file

    Format:
        ;FFMETADATA1
        [CHAPTER]
        TIMEBASE=1/1000
        START=0
        END=30000
        title=Chapter 1
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(";FFMETADATA1\n")

            for chapter in chapters:
                # Convert times to milliseconds
                start_ms = int(chapter['start'] * 1000)
                end_ms = int(chapter['end'] * 1000)

                f.write("\n[CHAPTER]\n")
                f.write("TIMEBASE=1/1000\n")
                f.write(f"START={start_ms}\n")
                f.write(f"END={end_ms}\n")
                f.write(f"title={chapter['title']}\n")

        logger.info(f"Generated chapter metadata: {output_path}")
        return output_path

    except Exception as e:
        raise FFmpegError(f"Failed to generate chapter metadata: {e}")
