"""
Orchestrator Agent - Coordinates the entire video processing pipeline
"""

from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime
import concurrent.futures
from tqdm import tqdm

from src.config import ProcessingConfig
from src.agents.transcription_agent import TranscriptionAgent, Segment
from src.agents.content_analysis_agent import ContentAnalysisAgent, CleanSegment
from src.agents.tts_generation_agent import TTSGenerationAgent
from src.agents.video_processing_agent import VideoProcessingAgent, ProcessedSegment
from src.exceptions import VideoProcessingError
from src.utils.logger import setup_logger, get_logger
from src.utils.checkpoint import CheckpointManager
from src.utils.validators import (
    validate_video_file,
    validate_audio_file,
    validate_json_file,
    validate_output_path
)
from src.utils.ffmpeg_utils import (
    add_subtitles_to_video,
    add_chapters_to_video,
    generate_chapter_metadata,
    get_video_duration
)


@dataclass
class ProcessingResult:
    """Result of video processing pipeline"""
    success: bool
    output_video: Optional[Path]
    processing_time: float
    segments_processed: int
    segments_failed: int
    filler_stats: Dict
    errors: List[str]
    warnings: List[str]


class OrchestratorAgent:
    """Orchestrates the entire video processing pipeline"""

    def __init__(self, config: ProcessingConfig):
        """
        Initialize orchestrator agent

        Args:
            config: Processing configuration
        """
        self.config = config
        self.logger = setup_logger(
            "OrchestratorAgent",
            log_file=config.logging.file,
            level=config.logging.level
        )

        # Initialize all agents
        self.transcription_agent = TranscriptionAgent(config.transcription)
        self.content_agent = ContentAnalysisAgent(config)
        self.tts_agent = TTSGenerationAgent(config.tts)
        self.video_agent = VideoProcessingAgent(config.video)

        # Initialize checkpoint manager
        checkpoint_dir = Path(config.output.temp_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint = CheckpointManager(checkpoint_dir / "pipeline.json")

        self.logger.info("Orchestrator initialized with all agents")

    def process_video(
        self,
        video_path: Path,
        output_path: Path,
        transcription_json: Optional[Path] = None,
        overwrite: bool = False,
        resume: bool = False
    ) -> ProcessingResult:
        """
        Process video through complete pipeline

        Args:
            video_path: Path to input video
            output_path: Path to output video
            transcription_json: Optional pre-existing transcription JSON
            overwrite: Whether to overwrite existing output
            resume: Whether to resume from checkpoint

        Returns:
            ProcessingResult with pipeline results

        Raises:
            VideoProcessingError: If processing fails
        """
        start_time = datetime.now()
        errors = []
        warnings = []

        try:
            self.logger.info("=" * 60)
            self.logger.info("Starting Video Processing Pipeline")
            self.logger.info("=" * 60)

            # Validate inputs
            self.logger.info("Validating inputs...")
            validate_video_file(video_path)
            validate_output_path(output_path, overwrite=overwrite)

            if transcription_json:
                validate_json_file(transcription_json)

            # Setup working directory
            work_dir = Path(self.config.output.temp_dir)
            work_dir.mkdir(parents=True, exist_ok=True)

            # Clear or load checkpoint
            if not resume:
                self.checkpoint.clear()
                self.logger.info("Starting fresh processing (checkpoint cleared)")
            else:
                progress = self.checkpoint.get_progress()
                self.logger.info(
                    f"Resuming from checkpoint: "
                    f"{progress['completed']}/{progress['total']} segments completed"
                )

            # Stage 1: Transcription
            self.logger.info("\n[Stage 1/5] Transcription")
            segments = self._stage_transcription(
                video_path,
                work_dir,
                transcription_json
            )

            # Stage 2: Content Analysis
            self.logger.info("\n[Stage 2/5] Content Analysis")
            clean_segments, filler_stats = self._stage_content_analysis(segments, work_dir)

            # Stage 3: TTS Generation
            self.logger.info("\n[Stage 3/5] TTS Generation")
            tts_audio_map = self._stage_tts_generation(
                clean_segments,
                work_dir,
                video_path,
                resume
            )

            # Stage 4: Video Processing
            self.logger.info("\n[Stage 4/5] Video Processing")
            processed_segments = self._stage_video_processing(
                video_path,
                clean_segments,
                tts_audio_map,
                work_dir,
                resume
            )

            # Stage 5: Final Assembly
            self.logger.info("\n[Stage 5/5] Final Assembly")
            final_video = self._stage_final_assembly(
                processed_segments,
                output_path,
                work_dir
            )

            # Calculate results
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.info("\n" + "=" * 60)
            self.logger.info("Processing Complete!")
            self.logger.info(f"Output: {final_video}")
            self.logger.info(f"Processing time: {processing_time:.2f}s")
            self.logger.info(f"Segments processed: {len(processed_segments)}")
            self.logger.info("=" * 60)

            # Cleanup
            if not self.config.output.keep_intermediates:
                self._cleanup_intermediates(work_dir, final_video)

            return ProcessingResult(
                success=True,
                output_video=final_video,
                processing_time=processing_time,
                segments_processed=len(processed_segments),
                segments_failed=0,
                filler_stats=filler_stats,
                errors=errors,
                warnings=warnings
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            errors.append(str(e))

            return ProcessingResult(
                success=False,
                output_video=None,
                processing_time=processing_time,
                segments_processed=0,
                segments_failed=0,
                filler_stats={},
                errors=errors,
                warnings=warnings
            )

    def _stage_transcription(
        self,
        video_path: Path,
        work_dir: Path,
        transcription_json: Optional[Path]
    ) -> List[Segment]:
        """Stage 1: Transcription"""
        if transcription_json:
            self.logger.info(f"Loading existing transcription: {transcription_json}")
            return self.transcription_agent.load_transcription(transcription_json)

        # Extract audio from video
        audio_path = work_dir / "original_audio.wav"
        self.logger.info("Extracting audio from video...")
        self.video_agent.extract_original_audio(video_path, audio_path)

        # Transcribe
        self.logger.info("Transcribing audio (this may take a while)...")
        segments, json_path = self.transcription_agent.transcribe_audio(
            audio_path,
            work_dir / "transcriptions"
        )

        # Validate
        validation = self.transcription_agent.validate_transcription(segments)
        if not validation['valid']:
            raise VideoProcessingError(
                f"Transcription validation failed: {validation['error']}"
            )

        self.logger.info(
            f"Transcription complete: {validation['metrics']['total_words']} words, "
            f"confidence: {validation['metrics']['average_confidence']:.2f}"
        )

        return segments

    def _stage_content_analysis(
        self,
        segments: List[Segment],
        work_dir: Path
    ) -> tuple:
        """Stage 2: Content Analysis"""
        # Detect fillers
        clean_segments, filler_stats = self.content_agent.detect_fillers(segments)

        self.logger.info(
            f"Removed {filler_stats['total_fillers']} filler words "
            f"({filler_stats['removal_rate']:.1%} removal rate)"
        )
        self.logger.info(f"Created {len(clean_segments)} clean segments")

        # Optimize segments
        clean_segments = self.content_agent.find_optimal_cuts(clean_segments)

        # Export cleaned transcript
        transcript_path = work_dir / "cleaned_transcript.txt"
        self.content_agent.export_cleaned_transcript(clean_segments, transcript_path)

        # Export subtitles if enabled
        if self.config.output.subtitles.enabled:
            self.logger.info("Generating subtitle file...")
            subtitle_path = work_dir / "subtitles.srt"
            self._generate_subtitles(segments, subtitle_path)

        return clean_segments, filler_stats

    def _stage_tts_generation(
        self,
        segments: List[CleanSegment],
        work_dir: Path,
        video_path: Path,
        resume: bool
    ) -> Dict[str, Path]:
        """Stage 3: TTS Generation"""
        tts_dir = work_dir / "tts_audio"
        tts_dir.mkdir(exist_ok=True)

        # Determine voice reference based on voice_mode
        reference_audio = None
        voice_mode = self.config.tts.voice_mode

        if voice_mode == "default":
            # Use Chatterbox default voice (no voice cloning)
            self.logger.info("Using default Chatterbox voice (no voice cloning)")
            reference_audio = None

        elif voice_mode == "custom":
            # Use custom voice reference file
            if self.config.tts.voice_reference:
                reference_audio = Path(self.config.tts.voice_reference)
                if not reference_audio.exists():
                    self.logger.warning(
                        f"Custom voice reference not found: {reference_audio}. "
                        "Falling back to auto-extraction."
                    )
                    voice_mode = "auto"  # Fallback to auto
                    reference_audio = None
                else:
                    self.logger.info(f"Using custom voice reference: {reference_audio}")
            else:
                self.logger.warning(
                    "voice_mode is 'custom' but no voice_reference provided. "
                    "Falling back to auto-extraction."
                )
                voice_mode = "auto"  # Fallback to auto

        if voice_mode == "auto":
            # Auto-extract voice from video
            self.logger.info("Auto-extracting voice profile from video audio...")
            reference_audio = work_dir / "voice_reference.wav"
            original_audio = work_dir / "original_audio.wav"

            if original_audio.exists():
                self.tts_agent.extract_voice_profile(
                    original_audio,
                    0,
                    min(30, segments[0].end if segments else 30),
                    reference_audio
                )

        # Generate TTS with progress bar
        tts_audio_map = {}
        failed_segments = []

        # Determine executor type and worker count
        if self.config.parallel_tts:
            # Use ProcessPoolExecutor for parallel TTS (each process has isolated memory)
            executor_class = concurrent.futures.ProcessPoolExecutor
            max_workers = self.config.tts_workers
            self.logger.info(f"Using parallel TTS processing with {max_workers} workers")
        else:
            # Use ThreadPoolExecutor with max_workers=1 for sequential processing
            executor_class = concurrent.futures.ThreadPoolExecutor
            max_workers = 1
            self.logger.info("Using sequential TTS processing (parallel_tts=False)")

        with tqdm(total=len(segments), desc="Generating TTS") as pbar:
            with executor_class(max_workers=max_workers) as executor:
                future_to_segment = {}

                for segment in segments:
                    # Skip if already completed
                    if resume and self.checkpoint.is_segment_completed(f"tts_{segment.segment_id}"):
                        pbar.update(1)
                        # Load existing file
                        audio_file = tts_dir / f"tts_{segment.segment_id}.wav"
                        if audio_file.exists():
                            tts_audio_map[segment.segment_id] = audio_file
                        continue

                    # Submit job
                    future = executor.submit(
                        self._generate_tts_for_segment,
                        segment,
                        tts_dir,
                        reference_audio
                    )
                    future_to_segment[future] = segment

                # Collect results
                for future in concurrent.futures.as_completed(future_to_segment):
                    segment = future_to_segment[future]
                    try:
                        audio_path = future.result()
                        tts_audio_map[segment.segment_id] = audio_path
                        self.checkpoint.mark_segment_completed(f"tts_{segment.segment_id}")
                    except Exception as e:
                        self.logger.error(
                            f"Failed to generate TTS for {segment.segment_id}: {e}"
                        )
                        failed_segments.append(segment.segment_id)
                        self.checkpoint.mark_segment_failed(f"tts_{segment.segment_id}", str(e))

                    pbar.update(1)

        if failed_segments:
            raise VideoProcessingError(
                f"TTS generation failed for {len(failed_segments)} segments"
            )

        return tts_audio_map

    def _generate_tts_for_segment(
        self,
        segment: CleanSegment,
        output_dir: Path,
        reference_audio: Optional[Path]
    ) -> Path:
        """Generate TTS for a single segment"""
        return self.tts_agent.generate_for_segment(
            segment,
            output_dir,
            reference_audio
        )

    def _stage_video_processing(
        self,
        video_path: Path,
        segments: List[CleanSegment],
        tts_audio_map: Dict[str, Path],
        work_dir: Path,
        resume: bool
    ) -> List[ProcessedSegment]:
        """Stage 4: Video Processing"""
        video_dir = work_dir / "video_segments"
        video_dir.mkdir(exist_ok=True)

        processed_segments = []
        failed_segments = []

        with tqdm(total=len(segments), desc="Processing video") as pbar:
            # Process in parallel
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config.max_workers
            ) as executor:
                future_to_segment = {}

                for segment in segments:
                    # Skip if already completed
                    if resume and self.checkpoint.is_segment_completed(f"video_{segment.segment_id}"):
                        pbar.update(1)
                        # TODO: Load existing processed segment
                        continue

                    tts_audio = tts_audio_map.get(segment.segment_id)
                    if not tts_audio:
                        self.logger.warning(
                            f"No TTS audio for {segment.segment_id}, skipping"
                        )
                        pbar.update(1)
                        continue

                    future = executor.submit(
                        self.video_agent.process_segment,
                        video_path,
                        segment,
                        tts_audio,
                        video_dir
                    )
                    future_to_segment[future] = segment

                # Collect results
                for future in concurrent.futures.as_completed(future_to_segment):
                    segment = future_to_segment[future]
                    try:
                        processed_seg = future.result()
                        processed_segments.append(processed_seg)
                        self.checkpoint.mark_segment_completed(f"video_{segment.segment_id}")
                    except Exception as e:
                        self.logger.error(
                            f"Failed to process video for {segment.segment_id}: {e}"
                        )
                        failed_segments.append(segment.segment_id)
                        self.checkpoint.mark_segment_failed(f"video_{segment.segment_id}", str(e))

                    pbar.update(1)

        # Sort segments by ID to maintain order
        processed_segments.sort(key=lambda x: x.segment_id)

        if failed_segments:
            raise VideoProcessingError(
                f"Video processing failed for {len(failed_segments)} segments"
            )

        return processed_segments

    def _stage_final_assembly(
        self,
        segments: List[ProcessedSegment],
        output_path: Path,
        work_dir: Path
    ) -> Path:
        """Stage 5: Final Assembly"""
        self.logger.info("Concatenating video segments...")

        # Concatenate all segments
        temp_video = work_dir / f"temp_{output_path.name}"
        final_video = self.video_agent.concatenate_segments(
            segments,
            temp_video
        )

        # Add subtitles if enabled
        if self.config.output.subtitles.enabled:
            subtitle_path = work_dir / "subtitles.srt"
            if subtitle_path.exists():
                self.logger.info("Embedding subtitles into video...")
                video_with_subs = work_dir / f"with_subs_{output_path.name}"
                add_subtitles_to_video(
                    final_video,
                    subtitle_path,
                    video_with_subs,
                    burn_in=self.config.output.subtitles.burn_in,
                    font_size=self.config.output.subtitles.font_size,
                    font_color=self.config.output.subtitles.font_color,
                    outline_color=self.config.output.subtitles.outline_color,
                    position=self.config.output.subtitles.position
                )
                final_video = video_with_subs

        # Add chapters if enabled
        if self.config.output.chapters.enabled:
            self.logger.info("Generating and embedding chapters...")
            chapters = self._generate_chapters(segments, final_video)
            if chapters:
                chapter_file = work_dir / "chapters.txt"
                generate_chapter_metadata(chapters, chapter_file)

                video_with_chapters = output_path
                add_chapters_to_video(final_video, chapter_file, video_with_chapters)
                final_video = video_with_chapters
            else:
                # If no chapters to add but subtitles were added, move to output
                if final_video != output_path:
                    import shutil
                    shutil.move(str(final_video), str(output_path))
                    final_video = output_path
        else:
            # If chapters not enabled but subtitles were added, move to output
            if final_video != output_path:
                import shutil
                shutil.move(str(final_video), str(output_path))
                final_video = output_path

        self.logger.info(f"Final video created: {final_video}")
        return final_video

    def _cleanup_intermediates(self, work_dir: Path, keep_file: Path):
        """Clean up intermediate files"""
        self.logger.info("Cleaning up intermediate files...")

        try:
            import shutil

            for item in work_dir.iterdir():
                if item == keep_file or item == keep_file.parent:
                    continue

                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)

            self.logger.info("Cleanup complete")

        except Exception as e:
            self.logger.warning(f"Cleanup failed: {e}")

    def _generate_subtitles(self, segments: List[Segment], output_path: Path) -> Path:
        """
        Generate subtitle file from segments

        Args:
            segments: List of transcription segments
            output_path: Path to output SRT file

        Returns:
            Path to generated subtitle file
        """
        return self.transcription_agent.export_to_srt(segments, output_path)

    def _generate_chapters(
        self,
        segments: List[ProcessedSegment],
        video_path: Path
    ) -> List[Dict[str, any]]:
        """
        Generate chapter markers from processed segments

        Args:
            segments: List of processed video segments
            video_path: Path to video file (for duration)

        Returns:
            List of chapter dictionaries with 'title', 'start', 'end'
        """
        config = self.config.output.chapters
        chapters = []

        try:
            video_duration = get_video_duration(video_path)
        except Exception as e:
            self.logger.warning(f"Failed to get video duration for chapters: {e}")
            return []

        if config.strategy == "segment":
            # Create one chapter per segment
            current_time = 0.0
            for i, segment in enumerate(segments, 1):
                seg_duration = segment.tts_duration
                chapters.append({
                    'title': f"Segment {i}",
                    'start': current_time,
                    'end': min(current_time + seg_duration, video_duration)
                })
                current_time += seg_duration

        elif config.strategy == "time_interval":
            # Create chapters at fixed time intervals
            current_time = 0.0
            chapter_num = 1
            while current_time < video_duration:
                end_time = min(current_time + config.time_interval, video_duration)
                chapters.append({
                    'title': f"Chapter {chapter_num}",
                    'start': current_time,
                    'end': end_time
                })
                current_time = end_time
                chapter_num += 1

        else:  # auto strategy
            # Intelligently group segments into chapters based on duration
            current_chapter_start = 0.0
            current_chapter_duration = 0.0
            chapter_num = 1
            segment_idx = 0

            while segment_idx < len(segments):
                segment = segments[segment_idx]
                seg_duration = segment.tts_duration

                # Check if adding this segment would exceed max chapter duration
                if (current_chapter_duration > 0 and
                    current_chapter_duration + seg_duration > config.min_chapter_duration * 2):
                    # Create chapter from accumulated segments
                    chapters.append({
                        'title': f"Chapter {chapter_num}",
                        'start': current_chapter_start,
                        'end': current_chapter_start + current_chapter_duration
                    })
                    chapter_num += 1
                    current_chapter_start += current_chapter_duration
                    current_chapter_duration = 0.0
                else:
                    # Add segment to current chapter
                    current_chapter_duration += seg_duration
                    segment_idx += 1

            # Add final chapter if there's remaining content
            if current_chapter_duration > 0:
                chapters.append({
                    'title': f"Chapter {chapter_num}",
                    'start': current_chapter_start,
                    'end': min(current_chapter_start + current_chapter_duration, video_duration)
                })

        # Filter out chapters that are too short
        chapters = [
            ch for ch in chapters
            if (ch['end'] - ch['start']) >= config.min_chapter_duration
        ]

        # Limit number of chapters
        if len(chapters) > config.max_chapters:
            self.logger.warning(
                f"Generated {len(chapters)} chapters, limiting to {config.max_chapters}"
            )
            chapters = chapters[:config.max_chapters]

        self.logger.info(f"Generated {len(chapters)} chapters")
        return chapters

    def generate_report(self, result: ProcessingResult, output_path: Path) -> Path:
        """
        Generate processing report

        Args:
            result: Processing result
            output_path: Path to report file

        Returns:
            Path to report file
        """
        try:
            with open(output_path, 'w') as f:
                f.write("Video Processing Report\n")
                f.write("=" * 60 + "\n\n")

                f.write(f"Status: {'SUCCESS' if result.success else 'FAILED'}\n")
                f.write(f"Output: {result.output_video}\n")
                f.write(f"Processing Time: {result.processing_time:.2f}s\n")
                f.write(f"Segments Processed: {result.segments_processed}\n")
                f.write(f"Segments Failed: {result.segments_failed}\n\n")

                if result.filler_stats:
                    f.write("Filler Word Statistics:\n")
                    f.write(f"  Total Removed: {result.filler_stats.get('total_fillers', 0)}\n")
                    f.write(f"  Removal Rate: {result.filler_stats.get('removal_rate', 0):.1%}\n\n")

                if result.errors:
                    f.write("Errors:\n")
                    for error in result.errors:
                        f.write(f"  - {error}\n")
                    f.write("\n")

                if result.warnings:
                    f.write("Warnings:\n")
                    for warning in result.warnings:
                        f.write(f"  - {warning}\n")

            self.logger.info(f"Report generated: {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Failed to generate report: {e}")
            raise
