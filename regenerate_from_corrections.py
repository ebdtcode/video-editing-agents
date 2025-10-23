#!/usr/bin/env python3
"""
Regenerate TTS and video from corrected transcript.

Takes the edited transcript JSON and regenerates only the TTS stage
and subsequent video processing, skipping transcription and analysis.

Usage:
    python regenerate_from_corrections.py --input editable_transcript.json
    python regenerate_from_corrections.py --input editable_transcript.json --resume
"""

import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List
import sys
import warnings

# Suppress PyTorch deprecation warning from Chatterbox library
warnings.filterwarnings('ignore', message='.*torch.backends.cuda.sdp_kernel.*', category=FutureWarning)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agents.tts_generation_agent import TTSGenerationAgent
from src.agents.video_processing_agent import VideoProcessingAgent, ProcessedSegment
from src.agents.orchestrator_agent import OrchestratorAgent
from src.agents.content_analysis_agent import CleanSegment
from src.config import ProcessingConfig
from src.utils.checkpoint import CheckpointManager


def load_corrected_transcript(json_file: Path) -> List[CleanSegment]:
    """Load and parse corrected transcript JSON."""

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    segments = []
    for seg_data in data['segments']:
        # Skip segments marked for exclusion
        if seg_data.get('skip', False):
            print(f"⏭️  Skipping {seg_data['segment_id']}: marked for exclusion")
            continue

        # Use corrected text if available, otherwise original
        text = seg_data.get('corrected_text', seg_data['original_text']).strip()

        if not text:
            print(f"⚠️  Warning: {seg_data['segment_id']} has no text, skipping")
            continue

        segment = CleanSegment(
            segment_id=seg_data['segment_id'],
            start=seg_data['start_time'],
            end=seg_data['end_time'],
            text=text,
            words=[],  # Words not needed for TTS generation
            original_text=seg_data['original_text']
        )
        segments.append(segment)

    return segments


def regenerate_pipeline(corrected_json: Path, config_file: Path, video_file: Path, output_file: Path, resume: bool = False):
    """Regenerate TTS and video from corrected transcript."""

    print("=" * 80)
    print("Regenerating from Corrected Transcript")
    print("=" * 80)

    # Load configuration
    print(f"\n📄 Loading config: {config_file}")
    config = ProcessingConfig.from_yaml(config_file)

    # Setup paths
    temp_dir = Path("temp_segments")
    checkpoint_file = temp_dir / "checkpoints" / "regenerate.json"

    # Initialize checkpoint manager
    checkpoint = CheckpointManager(checkpoint_file)

    if resume:
        progress = checkpoint.get_progress()
        print(f"\n♻️  Resuming from checkpoint:")
        print(f"   Completed: {progress['completed']}")
        print(f"   Failed: {progress['failed']}")
        print(f"   Pending: {progress['pending']}")
    else:
        # Clear checkpoint for fresh start
        checkpoint.clear()
        print(f"\n🆕 Starting fresh (checkpoint cleared)")

    # Load corrected segments
    print(f"\n📄 Loading corrections: {corrected_json}")
    segments = load_corrected_transcript(corrected_json)
    print(f"✅ Loaded {len(segments)} segments")

    # Initialize agents
    print(f"\n🤖 Initializing agents...")
    tts_agent = TTSGenerationAgent(config.tts)
    video_agent = VideoProcessingAgent(config.video)

    # Setup TTS directory
    tts_dir = temp_dir / "tts_audio"
    tts_dir.mkdir(parents=True, exist_ok=True)

    # Clean old TTS files
    print(f"\n🧹 Cleaning old TTS files...")
    for old_file in tts_dir.glob("tts_seg_*.wav"):
        old_file.unlink()

    # Determine voice reference based on voice_mode (same logic as orchestrator)
    reference_audio = None
    voice_mode = config.tts.voice_mode

    if voice_mode == "default":
        print(f"🎤 Using default Chatterbox voice (no voice cloning)")
        reference_audio = None

    elif voice_mode == "custom":
        if config.tts.voice_reference:
            reference_audio = Path(config.tts.voice_reference)
            if not reference_audio.exists():
                print(f"⚠️  Custom voice reference not found: {reference_audio}")
                print(f"    Falling back to default voice")
                reference_audio = None
            else:
                print(f"🎤 Using custom voice reference: {reference_audio}")
        else:
            print(f"⚠️  voice_mode is 'custom' but no voice_reference provided")
            print(f"    Falling back to default voice")

    elif voice_mode == "auto":
        # Auto-extract voice from first 30 seconds of video
        print(f"🎤 Auto-extracting voice profile from video...")
        reference_audio = temp_dir / "voice_reference.wav"
        original_audio = temp_dir / "original_audio.wav"

        # Extract audio from video if not already done
        if not original_audio.exists():
            import subprocess
            result = subprocess.run([
                'ffmpeg', '-y', '-i', str(video_file),
                '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2',
                str(original_audio)
            ], capture_output=True, text=True, check=False)

            if result.returncode != 0:
                print(f"⚠️  Failed to extract audio from video")
                print(f"    Falling back to default voice")
                reference_audio = None

        if reference_audio and original_audio.exists():
            # Extract voice profile (first 30 seconds)
            tts_agent.extract_voice_profile(
                original_audio,
                0,
                min(30, segments[0].end if segments else 30),
                reference_audio
            )
            print(f"✅ Voice profile extracted: {reference_audio}")

    # Stage 3: Generate TTS with corrected text
    print(f"\n📢 [Stage 3/5] TTS Generation (corrected text)")
    print("-" * 80)

    from tqdm import tqdm
    tts_audio_map = {}

    # Count segments that need processing
    segments_to_process = [s for s in segments if not checkpoint.is_segment_completed(f"tts_{s.segment_id}")]
    skipped_count = len(segments) - len(segments_to_process)

    if skipped_count > 0:
        print(f"⏭️  Skipping {skipped_count} already completed TTS segments")

    with tqdm(total=len(segments), desc="Generating TTS", initial=skipped_count) as pbar:
        for segment in segments:
            tts_checkpoint_id = f"tts_{segment.segment_id}"

            # Skip if already completed
            if checkpoint.is_segment_completed(tts_checkpoint_id):
                # Add to map from existing file
                expected_path = tts_dir / f"tts_{segment.segment_id}.wav"
                if expected_path.exists():
                    tts_audio_map[segment.segment_id] = expected_path
                pbar.update(1)
                continue

            try:
                checkpoint.mark_segment_started(tts_checkpoint_id)

                # Generate TTS for this segment (with voice cloning if configured)
                audio_path = tts_agent.generate_for_segment(
                    segment,
                    tts_dir,
                    reference_audio=reference_audio  # Respects voice_mode from config
                )
                tts_audio_map[segment.segment_id] = audio_path

                checkpoint.mark_segment_completed(tts_checkpoint_id, {'audio_path': str(audio_path)})
                pbar.set_postfix({"current": segment.segment_id})
                pbar.update(1)

            except Exception as e:
                print(f"\n❌ Failed to generate TTS for {segment.segment_id}: {e}")
                checkpoint.mark_segment_failed(tts_checkpoint_id, str(e))
                pbar.update(1)
                continue

    print(f"✅ Generated TTS for {len(tts_audio_map)}/{len(segments)} segments")

    if len(tts_audio_map) == 0:
        print("❌ No TTS audio generated. Cannot proceed.")
        return False

    # Stage 4: Process video segments
    print(f"\n🎬 [Stage 4/5] Video Processing")
    print("-" * 80)

    # Extract video properties
    video_path = Path(video_file)
    print(f"Processing video: {video_path}")

    # Count segments that need processing
    video_segments_to_process = [s for s in segments if not checkpoint.is_segment_completed(f"video_{s.segment_id}")]
    video_skipped_count = len(segments) - len(video_segments_to_process)

    if video_skipped_count > 0:
        print(f"⏭️  Skipping {video_skipped_count} already completed video segments")

    # Process each segment
    processed_segments = []
    with tqdm(total=len(segments), desc="Processing video", initial=video_skipped_count) as pbar:
        for segment in segments:
            video_checkpoint_id = f"video_{segment.segment_id}"

            if segment.segment_id not in tts_audio_map:
                pbar.update(1)
                continue

            # Skip if already completed
            if checkpoint.is_segment_completed(video_checkpoint_id):
                # Reconstruct ProcessedSegment from checkpoint
                segment_data = checkpoint.state['segments'][video_checkpoint_id]
                output_files = segment_data.get('output_files', {})
                if 'video_path' in output_files:
                    video_file_path = Path(output_files['video_path'])
                    audio_file_path = Path(output_files['audio_path'])
                    if video_file_path.exists() and audio_file_path.exists():
                        processed_segment = ProcessedSegment(
                            segment_id=segment.segment_id,
                            video_path=video_file_path,
                            audio_path=audio_file_path,
                            start_time=segment.start,
                            end_time=segment.end
                        )
                        processed_segments.append(processed_segment)
                pbar.update(1)
                continue

            try:
                checkpoint.mark_segment_started(video_checkpoint_id)
                tts_audio = tts_audio_map[segment.segment_id]

                # Process this segment with correct API
                processed_segment = video_agent.process_segment(
                    video_path,
                    segment,
                    tts_audio,
                    temp_dir
                )

                checkpoint.mark_segment_completed(video_checkpoint_id, {
                    'video_path': str(processed_segment.video_path),
                    'audio_path': str(processed_segment.audio_path)
                })
                processed_segments.append(processed_segment)
                pbar.update(1)

            except Exception as e:
                print(f"\n❌ Failed to process segment {segment.segment_id}: {e}")
                checkpoint.mark_segment_failed(video_checkpoint_id, str(e))
                pbar.update(1)
                continue

    print(f"✅ Processed {len(processed_segments)}/{len(segments)} video segments")

    # Stage 5: Final assembly
    print(f"\n🎞️  [Stage 5/5] Final Assembly")
    print("-" * 80)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Concatenate segments into final video
    video_agent.concatenate_segments(
        processed_segments,
        output_path
    )

    print(f"\n✅ Video regenerated successfully!")
    print(f"📹 Output: {output_path}")

    # Generate report
    report_path = output_path.parent / f"{output_path.stem}_report.txt"
    with open(report_path, 'w') as f:
        f.write(f"Regeneration Report\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"Input: {corrected_json}\n")
        f.write(f"Segments: {len(segments)} total\n")
        f.write(f"TTS Generated: {len(tts_audio_map)}\n")
        f.write(f"Video Segments: {len(processed_segments)}\n")
        f.write(f"Output: {output_path}\n")

    print(f"📄 Report: {report_path}")
    print("=" * 80)

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Regenerate TTS and video from corrected transcript'
    )
    parser.add_argument(
        '--input',
        default='editable_transcript.json',
        help='Corrected transcript JSON file'
    )
    parser.add_argument(
        '--video',
        default='videos/speaking_eng_lingup.mp4',
        help='Original video file'
    )
    parser.add_argument(
        '--output',
        default='output/cleaned_video_corrected.mp4',
        help='Output video file'
    )
    parser.add_argument(
        '--config',
        default='config_cpu.yaml',
        help='Configuration file'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint if available'
    )

    args = parser.parse_args()

    # Validate inputs
    json_file = Path(args.input)
    if not json_file.exists():
        print(f"❌ Error: Input file not found: {json_file}")
        print(f"\nRun this first:")
        print(f"  python export_transcript_for_editing.py --output {json_file}")
        sys.exit(1)

    video_file = Path(args.video)
    if not video_file.exists():
        print(f"❌ Error: Video file not found: {video_file}")
        sys.exit(1)

    config_file = Path(args.config)
    if not config_file.exists():
        print(f"❌ Error: Config file not found: {config_file}")
        sys.exit(1)

    # Run regeneration
    success = regenerate_pipeline(
        json_file,
        config_file,
        video_file,
        Path(args.output),
        resume=args.resume
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
