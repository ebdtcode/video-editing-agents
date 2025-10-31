"""
Reprocess failed video segments using existing TTS audio files.
This script processes only the segments that failed during video processing,
using CPU encoding instead of CUDA to avoid FFmpeg hangs.
"""

import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agents.video_processing_agent import VideoProcessingAgent
from src.config import ProcessingConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


def find_failed_segments(temp_dir: Path):
    """Find segments that have TTS audio but failed video processing."""
    tts_audio_dir = temp_dir / "tts_audio"
    video_segments_dir = temp_dir / "video_segments"
    
    failed_segments = []
    
    # List all TTS audio files
    tts_files = list(tts_audio_dir.glob("tts_seg_*.wav"))
    
    for tts_file in tts_files:
        segment_id = tts_file.stem.replace("tts_seg_", "seg_")
        
        # Check if final merged video exists
        final_video = video_segments_dir / f"{segment_id}.mp4"
        
        if not final_video.exists():
            # Check if raw video exists (partial processing)
            raw_video = video_segments_dir / f"{segment_id}_raw.mp4"
            failed_segments.append({
                'segment_id': segment_id,
                'tts_audio': str(tts_file),
                'has_raw_video': raw_video.exists(),
                'raw_video': str(raw_video) if raw_video.exists() else None
            })
    
    return failed_segments


def main():
    """Main function to reprocess failed segments."""
    
    # Load config
    config_path = Path("config_gpu.yaml")
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1
    
    config = ProcessingConfig.from_yaml(str(config_path))
    temp_dir = Path(config.output.temp_dir)
    
    # Find failed segments
    logger.info("Scanning for failed segments...")
    failed_segments = find_failed_segments(temp_dir)
    
    if not failed_segments:
        logger.info("No failed segments found! All segments processed successfully.")
        return 0
    
    logger.info(f"Found {len(failed_segments)} failed segments")
    
    # Load segment metadata to get timing information
    metadata_file = temp_dir / "segment_metadata.json"
    if not metadata_file.exists():
        logger.error("Segment metadata not found. Cannot reprocess without timing info.")
        return 1
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Create video processing agent with CPU-only settings
    logger.info("Initializing video processing agent (CPU mode)...")
    video_agent = VideoProcessingAgent(
        config=config,
        video_path=Path(metadata['source_video']),
        temp_dir=temp_dir
    )
    
    # Process each failed segment
    success_count = 0
    for i, failed in enumerate(failed_segments, 1):
        segment_id = failed['segment_id']
        logger.info(f"[{i}/{len(failed_segments)}] Reprocessing {segment_id}...")
        
        # Find segment info in metadata
        segment_info = None
        for seg in metadata['segments']:
            if seg['segment_id'] == segment_id:
                segment_info = seg
                break
        
        if not segment_info:
            logger.warning(f"Metadata not found for {segment_id}, skipping")
            continue
        
        # Create segment object
        from dataclasses import dataclass
        
        @dataclass
        class Segment:
            segment_id: str
            start_time: float
            end_time: float
            text: str
            audio_path: Path
        
        segment = Segment(
            segment_id=segment_id,
            start_time=segment_info['start_time'],
            end_time=segment_info['end_time'],
            text=segment_info.get('text', ''),
            audio_path=Path(failed['tts_audio'])
        )
        
        try:
            # Process the segment with CPU encoding
            output_path = video_agent.process_segment(segment)
            if output_path and output_path.exists():
                logger.info(f"✓ Successfully processed {segment_id}")
                success_count += 1
            else:
                logger.error(f"✗ Failed to process {segment_id}")
        except Exception as e:
            logger.error(f"✗ Error processing {segment_id}: {e}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Reprocessing complete!")
    logger.info(f"Successfully processed: {success_count}/{len(failed_segments)}")
    logger.info(f"Failed: {len(failed_segments) - success_count}/{len(failed_segments)}")
    logger.info(f"{'='*60}")
    
    return 0 if success_count == len(failed_segments) else 1


if __name__ == "__main__":
    sys.exit(main())
