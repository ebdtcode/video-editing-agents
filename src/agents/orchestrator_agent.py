"""
Orchestrator Agent - Coordinates the entire video processing pipeline
"""

from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime
import concurrent.futures
from tqdm import tqdm

from src.config import ProcessingConfig, TTSConfig
from src.agents.transcription_agent import TranscriptionAgent, Segment
from src.agents.content_analysis_agent import ContentAnalysisAgent, CleanSegment
from src.agents.content_correction_agent import ContentCorrectionAgent
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


def _generate_tts_for_segment_worker(
    segment: CleanSegment,
    output_dir: Path,
    reference_audio: Optional[Path],
    tts_config: TTSConfig
) -> Path:
    """
    Worker function for parallel TTS generation

    This is a module-level function to avoid pickling issues with ProcessPoolExecutor.
    Each worker process creates its own TTS agent instance.

    Args:
        segment: Clean segment to generate audio for
        output_dir: Output directory for audio files
        reference_audio: Optional reference audio for voice cloning
        tts_config: TTS configuration

    Returns:
        Path to generated audio file

    Raises:
        TTSGenerationError: If generation fails
    """
    # Import here to avoid circular imports and allow per-process initialization
    from src.agents.tts_generation_agent import TTSGenerationAgent

    # Create a fresh TTS agent instance for this worker
    tts_agent = TTSGenerationAgent(tts_config)

    # Generate TTS for the segment
    return tts_agent.generate_for_segment(
        segment,
        output_dir,
        reference_audio
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
        self.correction_agent = ContentCorrectionAgent(config.content_correction)
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
        resume: bool = False,
        from_corrected_transcript: Optional[Path] = None
    ) -> ProcessingResult:
        """
        Process video through complete pipeline

        Args:
            video_path: Path to input video
            output_path: Path to output video
            transcription_json: Optional pre-existing transcription JSON
            overwrite: Whether to overwrite existing output
            resume: Whether to resume from checkpoint
            from_corrected_transcript: Optional edited transcript file to resume from TTS stage

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

            # Check if resuming from edited transcript
            if from_corrected_transcript:
                self.logger.info(
                    f"\n*** EDIT AND RESUME MODE ***\n"
                    f"Loading edited transcript: {from_corrected_transcript}\n"
                    f"Skipping stages 1-3 (Transcription, Filler Removal, Correction)\n"
                    f"Starting from Stage 4 (TTS Generation)\n"
                )

                # Force clear checkpoint in edit-and-resume mode to avoid stale segments
                if not resume:
                    self.logger.info("Cleaning up old TTS and video files for fresh generation...")
                    # Clean TTS audio directory
                    tts_dir = work_dir / "tts_audio"
                    if tts_dir.exists():
                        for old_file in tts_dir.glob("*.wav"):
                            old_file.unlink()

                    # Clean video segments directory
                    video_dir = work_dir / "video_segments"
                    if video_dir.exists():
                        for old_file in video_dir.glob("*.mp4"):
                            old_file.unlink()

                    self.logger.info("Old files cleaned. Regenerating from edited transcript...")

                # Load edited transcript and map to segments
                corrected_segments = self._load_edited_transcript(
                    from_corrected_transcript,
                    work_dir,
                    video_path
                )

                # Set filler stats to N/A since we're skipping that stage
                filler_stats = {'removed': 0, 'removal_rate': 0.0}

            else:
                # Normal pipeline: Run stages 1-3
                # Stage 1: Transcription
                self.logger.info("\n[Stage 1/6] Transcription")
                segments = self._stage_transcription(
                    video_path,
                    work_dir,
                    transcription_json
                )

                # Stage 2: Content Analysis
                self.logger.info("\n[Stage 2/6] Content Analysis (Filler Removal)")
                clean_segments, filler_stats = self._stage_content_analysis(segments, work_dir)

                # Stage 3: Content Correction
                self.logger.info("\n[Stage 3/6] Content Correction (Grammar & Clarity)")
                corrected_segments = self._stage_content_correction(clean_segments, work_dir, video_path)

            # Stage 4: TTS Generation
            self.logger.info("\n[Stage 4/6] TTS Generation")
            tts_audio_map = self._stage_tts_generation(
                corrected_segments,
                work_dir,
                video_path,
                resume
            )

            # Stage 5: Video Processing
            self.logger.info("\n[Stage 5/6] Video Processing")
            processed_segments = self._stage_video_processing(
                video_path,
                corrected_segments,
                tts_audio_map,
                work_dir,
                resume
            )

            # Stage 6: Final Assembly
            self.logger.info("\n[Stage 6/6] Final Assembly")
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

    def _stage_content_correction(
        self,
        segments: List[CleanSegment],
        work_dir: Path,
        video_path: Path
    ) -> List[CleanSegment]:
        """Stage 3: Content Correction with LLM"""
        if not self.config.content_correction.enabled:
            self.logger.info("Content correction disabled, skipping")
            return segments

        # Store original segments for reporting
        original_segments = segments.copy()

        # Correct segments using LLM
        corrected_segments = self.correction_agent.correct_segments(segments)

        # Log correction summary
        changes_count = sum(
            1 for orig, corr in zip(original_segments, corrected_segments)
            if orig.text != corr.text
        )
        self.logger.info(
            f"Corrected {changes_count}/{len(segments)} segments using "
            f"{self.config.content_correction.mode}"
        )

        # Export corrected transcript
        corrected_path = work_dir / "corrected_transcript.txt"
        self.content_agent.export_cleaned_transcript(corrected_segments, corrected_path)

        # Save segment metadata for edit and resume feature
        metadata_path = work_dir / "segment_metadata.json"
        self._save_segment_metadata(corrected_segments, metadata_path, video_path)

        # Generate correction report if corrections were made
        if changes_count > 0:
            report_path = work_dir / "correction_report.txt"
            self.correction_agent.generate_correction_report(
                original_segments,
                corrected_segments,
                report_path
            )
            self.logger.info(f"Correction report saved: {report_path}")

        return corrected_segments

    def _stage_tts_generation(
        self,
        segments: List[CleanSegment],
        work_dir: Path,
        video_path: Path,
        resume: bool
    ) -> Dict[str, Path]:
        """Stage 4: TTS Generation"""
        tts_dir = work_dir / "tts_audio"
        tts_dir.mkdir(exist_ok=True)

        # Determine voice reference based on voice_mode
        reference_audio = None
        voice_mode = self.config.tts.voice_mode

        # Check if using original audio (skip TTS entirely)
        if voice_mode == "original":
            self.logger.info("Using original video audio (skipping TTS generation)")
            return self._extract_original_audio_segments(segments, work_dir, video_path)

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
            self.logger.warning(
                "Parallel TTS with GPU may cause memory contention. "
                "Consider setting parallel_tts: false in config for GPU processing."
            )
        else:
            # Use ThreadPoolExecutor with max_workers=1 for sequential processing
            # This is the recommended mode for GPU to avoid model loading conflicts
            executor_class = concurrent.futures.ThreadPoolExecutor
            max_workers = 1
            self.logger.info("Using sequential TTS processing (recommended for GPU)")

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
                    
                    # Skip if previously failed (with warning)
                    if resume and self.checkpoint.is_segment_failed(f"tts_{segment.segment_id}"):
                        self.logger.warning(
                            f"Skipping TTS segment {segment.segment_id} (previously failed/timeout). "
                            f"Review logs or manually regenerate this segment."
                        )
                        failed_segments.append(segment.segment_id)
                        pbar.update(1)
                        continue

                    # Submit job
                    if self.config.parallel_tts:
                        # Use module-level function for parallel processing
                        future = executor.submit(
                            _generate_tts_for_segment_worker,
                            segment,
                            tts_dir,
                            reference_audio,
                            self.config.tts
                        )
                    else:
                        # Use instance method for sequential processing
                        future = executor.submit(
                            self._generate_tts_for_segment,
                            segment,
                            tts_dir,
                            reference_audio
                        )
                    future_to_segment[future] = segment

                # Collect results with timeout protection
                for future in concurrent.futures.as_completed(future_to_segment, timeout=7200):
                    segment = future_to_segment[future]
                    try:
                        audio_path = future.result(timeout=300)  # 5 minute timeout per segment
                        tts_audio_map[segment.segment_id] = audio_path
                        self.checkpoint.mark_segment_completed(f"tts_{segment.segment_id}")
                    except concurrent.futures.TimeoutError:
                        self.logger.error(
                            f"TTS generation timeout for {segment.segment_id} (5 minutes exceeded)"
                        )
                        failed_segments.append(segment.segment_id)
                        self.checkpoint.mark_segment_failed(f"tts_{segment.segment_id}", "Timeout after 300s")
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

    def _extract_original_audio_segments(
        self,
        segments: List[CleanSegment],
        work_dir: Path,
        video_path: Path
    ) -> Dict[str, Path]:
        """
        Extract original audio segments from the video (skip TTS generation).
        Used when voice_mode is 'original' to preserve the original speaker's voice.

        Args:
            segments: List of cleaned segments
            work_dir: Working directory
            video_path: Path to original video file

        Returns:
            Dictionary mapping segment_id to audio file path
        """
        from ..utils.ffmpeg_utils import run_ffmpeg_safe

        tts_dir = work_dir / "tts_audio"
        tts_dir.mkdir(exist_ok=True)

        audio_map = {}

        self.logger.info(f"Extracting original audio for {len(segments)} segments")

        with tqdm(total=len(segments), desc="Extracting Original Audio") as pbar:
            for segment in segments:
                # Output path for this segment's audio
                # Match the TTS naming convention: tts_{segment_id}.wav
                audio_file = tts_dir / f"tts_{segment.segment_id}.wav"

                # Extract audio for this time range
                try:
                    cmd = [
                        'ffmpeg', '-y',
                        '-ss', str(segment.start),
                        '-to', str(segment.end),
                        '-i', str(video_path),
                        '-vn',  # No video
                        '-acodec', 'pcm_s16le',
                        '-ar', '16000',  # Match TTS sample rate
                        '-ac', '1',  # Mono
                        str(audio_file)
                    ]

                    run_ffmpeg_safe(
                        cmd,
                        error_msg=f"Failed to extract audio for segment {segment.segment_id}",
                        timeout=60
                    )
                    audio_map[segment.segment_id] = audio_file
                    self.logger.debug(f"Extracted audio for {segment.segment_id}: {audio_file}")

                except Exception as e:
                    self.logger.error(f"Failed to extract audio for {segment.segment_id}: {e}")
                    raise VideoProcessingError(
                        f"Failed to extract original audio for segment {segment.segment_id}: {e}"
                    )

                pbar.update(1)

        self.logger.info(f"Successfully extracted original audio for {len(audio_map)} segments")
        return audio_map

    def _stage_video_processing(
        self,
        video_path: Path,
        segments: List[CleanSegment],
        tts_audio_map: Dict[str, Path],
        work_dir: Path,
        resume: bool
    ) -> List[ProcessedSegment]:
        """Stage 5: Video Processing"""
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
                    
                    # Skip if previously failed (with warning)
                    if resume and self.checkpoint.is_segment_failed(f"video_{segment.segment_id}"):
                        self.logger.warning(
                            f"Skipping video segment {segment.segment_id} (previously failed/timeout). "
                            f"Review logs or manually process this segment."
                        )
                        failed_segments.append(segment.segment_id)
                        pbar.update(1)
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

                # Collect results with timeout protection
                for future in concurrent.futures.as_completed(future_to_segment, timeout=7200):
                    segment = future_to_segment[future]
                    try:
                        processed_seg = future.result(timeout=1200)  # 20 minute timeout per segment (allows for 15min FFmpeg + overhead)
                        processed_segments.append(processed_seg)
                        self.checkpoint.mark_segment_completed(f"video_{segment.segment_id}")
                    except concurrent.futures.TimeoutError:
                        self.logger.error(
                            f"Video processing timeout for {segment.segment_id} (10 minutes exceeded)"
                        )
                        failed_segments.append(segment.segment_id)
                        self.checkpoint.mark_segment_failed(f"video_{segment.segment_id}", "Timeout after 600s")
                    except Exception as e:
                        self.logger.error(
                            f"Failed to process video for {segment.segment_id}: {e}"
                        )
                        failed_segments.append(segment.segment_id)
                        self.checkpoint.mark_segment_failed(f"video_{segment.segment_id}", str(e))

                    pbar.update(1)

        # Sort segments by ID to maintain order
        processed_segments.sort(key=lambda x: x.segment_id)

        self.logger.info(f"Processed {len(processed_segments)} video segments")
        if len(processed_segments) > 0:
            self.logger.debug(
                f"First segment: {processed_segments[0].segment_id} "
                f"(original: {processed_segments[0].original_duration:.3f}s, "
                f"TTS: {processed_segments[0].tts_duration:.3f}s, "
                f"speed: {processed_segments[0].speed_factor:.3f}x)"
            )
            if len(processed_segments) > 1:
                self.logger.debug(
                    f"Last segment: {processed_segments[-1].segment_id} "
                    f"(original: {processed_segments[-1].original_duration:.3f}s, "
                    f"TTS: {processed_segments[-1].tts_duration:.3f}s, "
                    f"speed: {processed_segments[-1].speed_factor:.3f}x)"
                )

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
        """Stage 6: Final Assembly"""
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

    def _save_segment_metadata(
        self,
        segments: List[CleanSegment],
        metadata_path: Path,
        video_path: Path
    ):
        """
        Save segment metadata for edit and resume feature

        Args:
            segments: List of segments with timing info
            metadata_path: Path to save metadata JSON
            video_path: Original source video path for validation
        """
        import json
        from datetime import datetime

        metadata = {
            'version': '1.0',
            'created_at': datetime.now().isoformat(),
            'source_video': str(video_path.resolve()),
            'total_segments': len(segments),
            'segments': []
        }

        for seg in segments:
            metadata['segments'].append({
                'segment_id': seg.segment_id,
                'start': seg.start,
                'end': seg.end,
                'duration': round(seg.end - seg.start, 3),
                'text': seg.text  # Save original corrected text for reference
            })

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved segment metadata: {metadata_path}")
        self.logger.info(f"  Source video: {video_path}")
        self.logger.info(f"  Total segments: {len(segments)}")

    def _load_edited_transcript(
        self,
        transcript_file: Path,
        work_dir: Path,
        video_path: Path
    ) -> List[CleanSegment]:
        """
        Load edited transcript and map to segments with original timestamps

        Args:
            transcript_file: Path to edited transcript file
            work_dir: Working directory containing segment_metadata.json
            video_path: Current video path to validate against metadata

        Returns:
            List of CleanSegment objects with edited text and original timestamps

        Raises:
            VideoProcessingError: If metadata file not found or mismatch
        """
        import json

        # Load metadata
        metadata_path = work_dir / "segment_metadata.json"
        if not metadata_path.exists():
            raise VideoProcessingError(
                f"Segment metadata not found: {metadata_path}\n"
                "You need to run the full pipeline at least once before using edit and resume.\n"
                "Run without --from-corrected-transcript first to generate metadata."
            )

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Validate source video path
        if 'source_video' in metadata:
            saved_video_path = Path(metadata['source_video']).resolve()
            current_video_path = video_path.resolve()

            if saved_video_path != current_video_path:
                self.logger.warning(
                    f"\n⚠️  VIDEO PATH MISMATCH ⚠️\n"
                    f"Metadata was created from: {saved_video_path}\n"
                    f"But you're now using:      {current_video_path}\n"
                    f"\n"
                    f"Timestamps in metadata reference the ORIGINAL video.\n"
                    f"If you're using a different video file, timestamps may not align correctly.\n"
                    f"Make sure you're using the same source video that was used in the initial run.\n"
                )
        else:
            self.logger.warning(
                "Metadata doesn't include source video path (old format). "
                "Cannot validate video file match."
            )

        # Load edited transcript
        if not transcript_file.exists():
            raise VideoProcessingError(f"Edited transcript not found: {transcript_file}")

        with open(transcript_file, 'r', encoding='utf-8') as f:
            edited_lines = []
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Strip timestamp prefix if present: [1.04s - 12.17s] text
                # This allows users to edit the file with timestamps for reference
                import re
                match = re.match(r'^\[[\d.]+s\s*-\s*[\d.]+s\]\s*(.+)$', line)
                if match:
                    # Extract text without timestamp
                    text = match.group(1)
                    edited_lines.append(text)
                else:
                    # No timestamp prefix, use line as-is
                    edited_lines.append(line)

        # Check if line count matches
        if len(edited_lines) != len(metadata['segments']):
            raise VideoProcessingError(
                f"Line count mismatch!\n"
                f"Edited transcript has {len(edited_lines)} lines\n"
                f"But metadata has {len(metadata['segments'])} segments\n"
                f"Make sure you keep the same number of lines when editing."
            )

        # Create CleanSegment objects with edited text and original timing
        segments = []
        for edited_text, seg_meta in zip(edited_lines, metadata['segments']):
            segment = CleanSegment(
                segment_id=seg_meta['segment_id'],
                text=edited_text,  # Use edited text
                start=seg_meta['start'],  # Keep original timing
                end=seg_meta['end'],
                words=[],  # Word-level timing not needed for TTS
                original_text=seg_meta['text']  # Original text before user edits
            )
            segments.append(segment)

            # Debug logging for first few segments
            if len(segments) <= 3:
                self.logger.debug(
                    f"Loaded segment {segment.segment_id}: "
                    f"[{segment.start:.3f}s - {segment.end:.3f}s] "
                    f"text={edited_text[:50]}..."
                )

        self.logger.info(
            f"\n✅ Loaded {len(segments)} segments from edited transcript"
        )
        self.logger.info(
            f"   Edited text: {transcript_file}"
        )
        self.logger.info(
            f"   Timestamps: {metadata_path} (from original video)"
        )
        self.logger.info(
            f"   Note: Timestamp prefixes like '[1.04s - 12.17s]' are automatically stripped"
        )
        self.logger.info(
            f"\n📝 Edit-and-resume workflow:"
        )
        self.logger.info(
            f"   1. Your edited text will be used for TTS generation (without timestamps)"
        )
        self.logger.info(
            f"   2. Video segments will be extracted from ORIGINAL video using saved timestamps"
        )
        self.logger.info(
            f"   3. Extracted video will be synchronized with new TTS audio"
        )

        return segments
