#!/usr/bin/env python3
"""
Diagnose concatenation issue by testing the xfade filter chain
"""
import subprocess
import json
from pathlib import Path

# Get segment info
segments_dir = Path("temp_segments/video_segments")
segments = sorted(segments_dir.glob("seg_*.mp4"))

print(f"Found {len(segments)} segments")
print("\nSegment durations:")

total_duration = 0
for seg in segments:
    result = subprocess.run([
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'json',
        str(seg)
    ], capture_output=True, text=True)
    
    data = json.loads(result.stdout)
    duration = float(data['format']['duration'])
    total_duration += duration
    print(f"  {seg.name}: {duration:.2f}s")

print(f"\nTotal duration (without transitions): {total_duration:.2f}s")

transition_duration = 0.3
num_transitions = len(segments) - 1
expected_duration = total_duration - (transition_duration * num_transitions)
print(f"Expected duration (with {transition_duration}s transitions): {expected_duration:.2f}s")

# Check if the output file exists and its actual duration
output_file = Path("output/enhance_image_quality_corrected_new.mp4")
if output_file.exists():
    result = subprocess.run([
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-show_entries', 'stream=codec_type,duration,nb_frames',
        '-of', 'json',
        str(output_file)
    ], capture_output=True, text=True)
    
    data = json.loads(result.stdout)
    print(f"\nActual output file:")
    print(f"  Video duration: {data['streams'][0].get('duration', 'N/A')}s")
    print(f"  Video frames: {data['streams'][0].get('nb_frames', 'N/A')}")
    print(f"  Audio duration: {data['streams'][1].get('duration', 'N/A')}s")
    print(f"  Total duration: {data['format']['duration']}s")
    
    video_dur = float(data['streams'][0].get('duration', 0))
    audio_dur = float(data['streams'][1].get('duration', 0))
    
    if abs(video_dur - audio_dur) > 1:
        print(f"\n⚠️  MISMATCH: Video ({video_dur:.2f}s) and Audio ({audio_dur:.2f}s) durations differ!")
        print(f"   Video is {audio_dur - video_dur:.2f}s shorter than audio")
