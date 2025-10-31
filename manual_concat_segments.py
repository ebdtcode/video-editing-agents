"""
Manually concatenate all successfully processed video segments.
Creates final video from available segments, skipping failed ones.
"""

import sys
from pathlib import Path
import subprocess
from typing import List, Tuple

def get_available_segments(video_segments_dir: Path) -> List[Tuple[str, Path]]:
    """Get all successfully processed segment files (final merged .mp4)."""
    segments = []
    
    # Look for all final segment files (not _raw, _retimed, etc.)
    for seg_file in sorted(video_segments_dir.glob("seg_*.mp4")):
        # Skip intermediate files
        if any(x in seg_file.stem for x in ['_raw', '_retimed', '_temp']):
            continue
        
        segment_id = seg_file.stem
        segments.append((segment_id, seg_file))
    
    return segments


def create_concat_list(segments: List[Tuple[str, Path]], output_file: Path):
    """Create FFmpeg concat list file."""
    with open(output_file, 'w') as f:
        for segment_id, seg_path in segments:
            # Use absolute path to avoid path resolution issues
            # FFmpeg concat format requires forward slashes even on Windows
            abs_path = seg_path.absolute().as_posix()
            f.write(f"file '{abs_path}'\n")


def concatenate_videos(concat_list: Path, output_video: Path):
    """Concatenate videos using FFmpeg."""
    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', str(concat_list),
        '-c', 'copy',  # Copy without re-encoding for speed
        str(output_video)
    ]
    
    print(f"Concatenating segments...")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: FFmpeg failed:")
        print(result.stderr)
        return False
    
    return True


def main():
    """Main function."""
    video_segments_dir = Path("temp_segments/video_segments")
    
    if not video_segments_dir.exists():
        print(f"ERROR: Video segments directory not found: {video_segments_dir}")
        return 1
    
    # Find all available segments
    print("Scanning for successfully processed segments...")
    segments = get_available_segments(video_segments_dir)
    
    if not segments:
        print("ERROR: No successfully processed segments found!")
        return 1
    
    print(f"Found {len(segments)} successfully processed segments:")
    for segment_id, _ in segments[:10]:
        print(f"  - {segment_id}")
    if len(segments) > 10:
        print(f"  ... and {len(segments) - 10} more")
    
    # Create concat list
    concat_list = Path("temp_segments/manual_concat_list.txt")
    print(f"\nCreating concat list: {concat_list}")
    create_concat_list(segments, concat_list)
    
    # Concatenate
    output_video = Path("output/input_video_copilot_partial.mp4")
    output_video.parent.mkdir(exist_ok=True)
    
    print(f"\nConcatenating to: {output_video}")
    if concatenate_videos(concat_list, output_video):
        print(f"\n{'='*60}")
        print(f"SUCCESS!")
        print(f"Partial video created: {output_video}")
        print(f"Segments included: {len(segments)}")
        print(f"{'='*60}")
        print(f"\nNote: This video contains only the successfully processed segments.")
        print(f"Failed segments have been skipped.")
        return 0
    else:
        print(f"\nERROR: Failed to concatenate videos")
        return 1


if __name__ == "__main__":
    sys.exit(main())
