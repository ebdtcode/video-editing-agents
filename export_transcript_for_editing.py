#!/usr/bin/env python3
"""
Export transcript for manual editing and correction.

Creates an editable JSON file that can be modified and re-imported
to regenerate TTS with corrected text.

Usage:
    python export_transcript_for_editing.py [--output custom_name.json]
"""

import json
import argparse
from pathlib import Path
from datetime import datetime


def detect_source_video():
    """Try to detect the source video from temp files."""
    temp_dir = Path("temp_segments")
    
    # Look for temp video files that indicate source
    for pattern in ["temp_*.mp4", "with_subs_*.mp4"]:
        temp_files = list(temp_dir.glob(pattern))
        if temp_files:
            # Extract video name from temp file
            temp_name = temp_files[0].stem
            # Remove prefixes
            for prefix in ["temp_", "with_subs_"]:
                if temp_name.startswith(prefix):
                    temp_name = temp_name[len(prefix):]
            
            # Try to find matching video in videos folder
            video_file = Path("videos") / f"{temp_name}.mp4"
            if video_file.exists():
                return str(video_file)
    
    return None


def export_transcript(source_video=None, output_file='editable_transcript.json'):
    """Export cleaned transcript to editable JSON format."""

    # Input file
    cleaned_transcript = Path("temp_segments/cleaned_transcript.txt")

    if not cleaned_transcript.exists():
        print(f"❌ Error: Cleaned transcript not found at {cleaned_transcript}")
        print("   Run the pipeline first to generate the transcript.")
        return False

    # Parse the transcript
    segments = []
    current_segment = None

    with open(cleaned_transcript, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Check if this is a timestamp line
            if line.startswith('[') and 's]' in line:
                # Extract timestamp and text
                parts = line.split('] ', 1)
                if len(parts) == 2:
                    timestamp = parts[0] + ']'
                    text = parts[1]

                    # Extract start/end times
                    time_part = timestamp[1:-1]  # Remove []
                    if ' - ' in time_part:
                        start, end = time_part.split(' - ')
                        start_time = float(start.replace('s', ''))
                        end_time = float(end.replace('s', ''))

                        segments.append({
                            "segment_id": f"seg_{len(segments):04d}",
                            "start_time": start_time,
                            "end_time": end_time,
                            "duration": round(end_time - start_time, 2),
                            "original_text": text,
                            "corrected_text": text,  # User can edit this
                            "notes": "",  # User can add notes
                            "skip": False  # Set to true to skip this segment
                        })

    if not segments:
        print("❌ Error: No segments found in transcript")
        return False

    # Detect source video if not provided
    if source_video is None:
        source_video = detect_source_video()
        if source_video:
            print(f"🔍 Detected source video: {source_video}")
        else:
            print("⚠️  Warning: Could not auto-detect source video")
            print("   You'll need to specify --video when regenerating")

    # Create export data
    export_data = {
        "metadata": {
            "exported_at": datetime.now().isoformat(),
            "source": str(cleaned_transcript),
            "source_video": source_video,
            "total_segments": len(segments),
            "total_duration": round(segments[-1]["end_time"] - segments[0]["start_time"], 2),
            "instructions": {
                "1": "Edit the 'corrected_text' field to fix transcription errors",
                "2": "Set 'skip' to true to exclude a segment from TTS generation",
                "3": "Add notes to document why you made changes",
                "4": "Do NOT modify segment_id, start_time, or end_time",
                "5": "Save and run: python regenerate_from_corrections.py"
            }
        },
        "segments": segments
    }

    # Output file
    output_file = Path(output_file)

    # Write JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("✅ Transcript Exported Successfully!")
    print("=" * 80)
    print(f"\nFile: {output_file}")
    print(f"Segments: {len(segments)}")
    print(f"Duration: {export_data['metadata']['total_duration']:.1f}s")
    print(f"\nEdit the file to:")
    print("  1. Fix transcription errors in 'corrected_text'")
    print("  2. Set 'skip': true to exclude unwanted segments")
    print("  3. Add notes to document changes")
    print(f"Then run:")
    if source_video:
        print(f"  python regenerate_from_corrections.py --input {output_file}")
        print(f"  (Video will be auto-detected from metadata: {source_video})")
    else:
        print(f"  python regenerate_from_corrections.py --input {output_file} --video <video_file>")
    print("=" * 80)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export transcript for editing')
    parser.add_argument('--output', default='editable_transcript.json',
                       help='Output JSON file (default: editable_transcript.json)')
    parser.add_argument('--video', default=None,
                       help='Source video file (if not specified, will try to auto-detect)')
    args = parser.parse_args()
    
    export_transcript(source_video=args.video, output_file=args.output)
