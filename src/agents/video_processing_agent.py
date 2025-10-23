"""
Video Processing Agent - Handles video manipulation and synchronization
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from src.config import VideoConfig
from src.agents.content_analysis_agent import CleanSegment
from src.exceptions import FFmpegError, ValidationError
from src.utils.logger import get_logger
from src.utils.ffmpeg_utils import (
    cut_video_segment,
    concatenate_videos,
    merge_video_audio,
    retime_video,
    get_audio_duration,
    run_ffmpeg_safe
)


logger = get_logger(__name__)


@dataclass
class ProcessedSegment:
    """Represents a processed video segment"""
    segment_id: str
    video_path: Path
    audio_path: Path
    original_duration: float
    tts_duration: float
    speed_factor: float


class VideoProcessingAgent:
    """Handles video processing and audio-video synchronization"""

    def __init__(self, config: VideoConfig):
        """
        Initialize video processing agent

        Args:
            config: Video configuration
        """
        self.config = config
        self.logger = get_logger(self.__class__.__name__)

    def process_segment(
        self,
        video_path: Path,
        segment: CleanSegment,
        tts_audio_path: Path,
        output_dir: Path
    ) -> ProcessedSegment:
        """
        Process a single video segment with PERFECT audio sync

        Ensures the video segment duration EXACTLY matches the TTS audio duration.
        Video is retimed (sped up/slowed down) to match audio precisely.

        Args:
            video_path: Path to original video
            segment: Clean segment information
            tts_audio_path: Path to TTS audio for this segment
            output_dir: Output directory for processed segment

        Returns:
            ProcessedSegment with processing information

        Raises:
            FFmpegError: If processing fails
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)

            # Calculate durations with high precision
            original_duration = segment.end - segment.start
            tts_duration = get_audio_duration(tts_audio_path)

            # Ensure TTS duration is valid
            if tts_duration <= 0:
                raise FFmpegError(
                    f"Invalid TTS audio duration: {tts_duration}s for {segment.segment_id}"
                )

            self.logger.debug(
                f"Processing segment {segment.segment_id}: "
                f"original={original_duration:.3f}s, tts={tts_duration:.3f}s"
            )

            # Cut video segment (without audio)
            raw_segment_path = output_dir / f"{segment.segment_id}_raw.mp4"
            cut_video_segment(
                video_path,
                raw_segment_path,
                segment.start,
                segment.end,
                include_audio=False  # Remove original audio
            )

            # Synchronize video with TTS audio (PERFECT SYNC)
            synced_segment_path = output_dir / f"{segment.segment_id}.mp4"

            # Calculate speed factor for perfect sync
            speed_factor = original_duration / tts_duration

            # Always retime for perfect sync (unless difference is negligible)
            if abs(speed_factor - 1.0) < 0.001:  # < 0.1% difference
                self.logger.debug(
                    f"Segment {segment.segment_id}: Perfect match, no retime needed"
                )
                # Direct merge without retiming
                synced_segment_path = self._precise_merge(
                    raw_segment_path,
                    tts_audio_path,
                    synced_segment_path
                )
            else:
                # Retime video to EXACTLY match audio duration
                self.logger.debug(
                    f"Segment {segment.segment_id}: Retiming by factor {speed_factor:.3f}"
                )
                synced_segment_path = self._perfect_sync_merge(
                    raw_segment_path,
                    tts_audio_path,
                    synced_segment_path,
                    speed_factor,
                    tts_duration
                )

            # Verify sync accuracy
            final_duration = get_audio_duration(synced_segment_path)
            duration_diff = abs(final_duration - tts_duration)

            if duration_diff > 0.05:  # More than 50ms drift
                self.logger.warning(
                    f"Segment {segment.segment_id}: Sync drift detected: "
                    f"{duration_diff:.3f}s (expected {tts_duration:.3f}s, got {final_duration:.3f}s)"
                )

            # Clean up temporary files
            raw_segment_path.unlink(missing_ok=True)

            return ProcessedSegment(
                segment_id=segment.segment_id,
                video_path=synced_segment_path,
                audio_path=tts_audio_path,
                original_duration=original_duration,
                tts_duration=tts_duration,
                speed_factor=speed_factor
            )

        except Exception as e:
            raise FFmpegError(f"Failed to process segment {segment.segment_id}: {e}")

    def _retime_and_merge(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
        speed_factor: float
    ) -> Path:
        """
        Retime video to match audio duration and merge

        Args:
            video_path: Path to video segment
            audio_path: Path to audio file
            output_path: Path to output file
            speed_factor: Speed adjustment factor

        Returns:
            Path to synced video
        """
        # Only retime if the difference is significant (>5%)
        if abs(speed_factor - 1.0) < 0.05:
            self.logger.debug(
                f"Speed factor {speed_factor:.2f} close to 1.0, "
                "skipping retime"
            )
            return merge_video_audio(
                video_path,
                audio_path,
                output_path,
                video_codec="copy"
            )

        # Retime video
        retimed_path = output_path.parent / f"{output_path.stem}_retimed.mp4"
        retime_video(video_path, retimed_path, speed_factor)

        # Merge with audio
        merge_video_audio(
            retimed_path,
            audio_path,
            output_path,
            video_codec=self.config.codec
        )

        # Clean up retimed temp file
        retimed_path.unlink(missing_ok=True)

        return output_path

    def _simple_merge(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path
    ) -> Path:
        """
        Simple merge of video and audio

        Args:
            video_path: Path to video segment
            audio_path: Path to audio file
            output_path: Path to output file

        Returns:
            Path to merged video
        """
        return merge_video_audio(
            video_path,
            audio_path,
            output_path,
            video_codec="copy"
        )

    def _precise_merge(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path
    ) -> Path:
        """
        Precise merge when video and audio durations already match perfectly

        Used when speed_factor is very close to 1.0 (< 0.1% difference).
        No retiming needed, just merge with codec copy for maximum quality.

        Args:
            video_path: Path to video segment (without audio)
            audio_path: Path to TTS audio file
            output_path: Path to output file

        Returns:
            Path to merged video with perfect sync
        """
        self.logger.debug(
            f"Precise merge (no retime): {video_path.name} + {audio_path.name}"
        )

        return merge_video_audio(
            video_path,
            audio_path,
            output_path,
            video_codec="copy"  # No re-encoding for maximum quality
        )

    def _perfect_sync_merge(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
        speed_factor: float,
        target_duration: float
    ) -> Path:
        """
        Merge video and audio with perfect synchronization via retiming

        Retimes the video to EXACTLY match the TTS audio duration.
        Ensures perfect lip-sync and timing throughout the segment.

        Args:
            video_path: Path to video segment (without audio)
            audio_path: Path to TTS audio file
            target_duration: Target duration in seconds (TTS audio duration)
            speed_factor: Speed adjustment factor (original/tts)
            output_path: Path to output file

        Returns:
            Path to merged video with perfect sync
        """
        self.logger.debug(
            f"Perfect sync merge: {video_path.name} with speed_factor={speed_factor:.3f}"
        )

        # Step 1: Retime video to match TTS duration exactly
        retimed_path = output_path.parent / f"{output_path.stem}_retimed.mp4"

        try:
            # Use setpts filter for precise speed adjustment
            # PTS (Presentation Timestamp) = speed_factor * original_PTS
            # Speed > 1.0 = speed up (compress time)
            # Speed < 1.0 = slow down (expand time)
            retime_video(video_path, retimed_path, speed_factor)

            self.logger.debug(
                f"Retimed video: {video_path.name} → {retimed_path.name} "
                f"(factor={speed_factor:.3f})"
            )

            # Step 2: Merge retimed video with TTS audio
            merge_video_audio(
                retimed_path,
                audio_path,
                output_path,
                video_codec=self.config.codec,
                audio_codec="aac"
            )

            self.logger.debug(
                f"Merged with audio: {retimed_path.name} + {audio_path.name} → {output_path.name}"
            )

            # Clean up temporary retimed file
            retimed_path.unlink(missing_ok=True)

            return output_path

        except Exception as e:
            # Clean up on failure
            if retimed_path.exists():
                retimed_path.unlink(missing_ok=True)
            raise FFmpegError(f"Failed to create perfect sync merge: {e}")

    def concatenate_segments(
        self,
        segments: List[ProcessedSegment],
        output_path: Path,
        apply_transitions: bool = None
    ) -> Path:
        """
        Concatenate processed segments into final video

        Args:
            segments: List of processed segments
            output_path: Path to output video
            apply_transitions: Whether to apply transitions (uses config if None)

        Returns:
            Path to final video

        Raises:
            FFmpegError: If concatenation fails
        """
        if not segments:
            raise ValidationError("No segments to concatenate")

        self.logger.info(f"Concatenating {len(segments)} video segments")

        # Use config value if not specified
        if apply_transitions is None:
            apply_transitions = self.config.transitions.enabled

        if apply_transitions and len(segments) > 1:
            return self._concatenate_with_transitions(segments, output_path)
        else:
            return self._concatenate_simple(segments, output_path)

    def _concatenate_simple(
        self,
        segments: List[ProcessedSegment],
        output_path: Path
    ) -> Path:
        """Simple concatenation without transitions"""
        video_paths = [seg.video_path for seg in segments]
        return concatenate_videos(video_paths, output_path)

    def _concatenate_with_transitions(
        self,
        segments: List[ProcessedSegment],
        output_path: Path
    ) -> Path:
        """
        Concatenate segments with transition effects

        Args:
            segments: List of processed segments
            output_path: Path to output video

        Returns:
            Path to final video
        """
        try:
            transition_type = self.config.transitions.type
            duration = self.config.transitions.duration

            self.logger.info(
                f"Applying {transition_type} transitions "
                f"({duration}s) between segments"
            )

            # Build complex filter for transitions
            filter_parts = []
            input_labels = []

            # Create input labels
            for i in range(len(segments)):
                input_labels.append(f"[{i}:v]")

            # Build xfade filters between consecutive segments
            current_label = input_labels[0]

            for i in range(len(segments) - 1):
                next_label = input_labels[i + 1]
                output_label = f"[v{i}]" if i < len(segments) - 2 else "[outv]"

                # Calculate offset (cumulative duration minus transition)
                offset = sum(seg.tts_duration for seg in segments[:i+1]) - (duration * i)

                filter_parts.append(
                    f"{current_label}{next_label}xfade="
                    f"transition={transition_type}:"
                    f"duration={duration}:"
                    f"offset={offset:.2f}"
                    f"{output_label}"
                )

                current_label = output_label

            # Build FFmpeg command
            cmd = ['ffmpeg', '-y']

            # Add all inputs
            for segment in segments:
                cmd.extend(['-i', str(segment.video_path)])

            # Add filter complex
            filter_complex = ';'.join(filter_parts)
            cmd.extend([
                '-filter_complex', filter_complex,
                '-map', '[outv]',
                '-c:v', self.config.codec,
                '-b:v', self.config.bitrate,
                str(output_path)
            ])

            run_ffmpeg_safe(
                cmd,
                "Failed to concatenate with transitions",
                capture_output=True,
                check=True
            )

            self.logger.info(f"Created video with transitions: {output_path}")
            return output_path

        except Exception as e:
            self.logger.warning(
                f"Transition concatenation failed: {e}. "
                "Falling back to simple concatenation"
            )
            return self._concatenate_simple(segments, output_path)

    def create_final_video(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path
    ) -> Path:
        """
        Create final video by merging concatenated video and audio

        Args:
            video_path: Path to concatenated video
            audio_path: Path to concatenated audio
            output_path: Path to final output

        Returns:
            Path to final video

        Raises:
            FFmpegError: If creation fails
        """
        self.logger.info("Creating final video")

        return merge_video_audio(
            video_path,
            audio_path,
            output_path,
            video_codec=self.config.codec,
            audio_codec="aac",
            audio_bitrate="192k"
        )

    def validate_segment(self, segment: ProcessedSegment) -> Dict[str, any]:
        """
        Validate a processed segment

        Args:
            segment: Processed segment to validate

        Returns:
            Dict with validation results
        """
        issues = []

        # Check if files exist
        if not segment.video_path.exists():
            issues.append(f"Video file not found: {segment.video_path}")

        if not segment.audio_path.exists():
            issues.append(f"Audio file not found: {segment.audio_path}")

        # Check speed factor
        if segment.speed_factor < 0.5 or segment.speed_factor > 2.0:
            issues.append(
                f"Extreme speed factor {segment.speed_factor:.2f} "
                "(may result in unnatural playback)"
            )

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'segment_id': segment.segment_id,
            'speed_factor': segment.speed_factor,
            'original_duration': segment.original_duration,
            'tts_duration': segment.tts_duration
        }

    def generate_preview(
        self,
        segments: List[ProcessedSegment],
        output_path: Path,
        max_segments: int = 5
    ) -> Path:
        """
        Generate a preview video from first few segments

        Args:
            segments: List of processed segments
            output_path: Path to preview video
            max_segments: Maximum number of segments to include

        Returns:
            Path to preview video
        """
        preview_segments = segments[:max_segments]

        self.logger.info(
            f"Generating preview with {len(preview_segments)} segments"
        )

        return self.concatenate_segments(
            preview_segments,
            output_path,
            apply_transitions=False  # Faster without transitions
        )

    def extract_original_audio(
        self,
        video_path: Path,
        output_path: Path,
        sample_rate: int = 16000
    ) -> Path:
        """
        Extract audio from original video

        Args:
            video_path: Path to video file
            output_path: Path to output audio file
            sample_rate: Audio sample rate

        Returns:
            Path to extracted audio
        """
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', str(video_path),
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', str(sample_rate),
                '-ac', '1',
                str(output_path)
            ]

            run_ffmpeg_safe(
                cmd,
                f"Failed to extract audio from {video_path}",
                capture_output=True,
                check=True
            )

            self.logger.info(f"Extracted audio to {output_path}")
            return output_path

        except Exception as e:
            raise FFmpegError(f"Failed to extract audio: {e}")
