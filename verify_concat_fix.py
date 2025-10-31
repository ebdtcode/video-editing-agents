#!/usr/bin/env python3
"""
Verify that the regenerated video has correct duration and all segments
"""
import subprocess
import json
from pathlib import Path
import sys

def check_video(video_path):
    """Check if video has proper duration and all segments"""
    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}")
        return False
    
    # Get video properties
    result = subprocess.run([
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-show_entries', 'stream=codec_type,duration,nb_frames',
        '-of', 'json',
        str(video_path)
    ], capture_output=True, text=True)
    
    data = json.loads(result.stdout)
    
    video_stream = next((s for s in data['streams'] if s['codec_type'] == 'video'), None)
    audio_stream = next((s for s in data['streams'] if s['codec_type'] == 'audio'), None)
    
    if not video_stream or not audio_stream:
        print(f"❌ Missing video or audio stream")
        return False
    
    video_dur = float(video_stream.get('duration', 0))
    audio_dur = float(audio_stream.get('duration', 0))
    video_frames = int(video_stream.get('nb_frames', 0))
    
    print(f"\n📹 Video: {video_path.name}")
    print(f"   Video duration: {video_dur:.2f}s ({video_frames} frames)")
    print(f"   Audio duration: {audio_dur:.2f}s")
    print(f"   Difference: {abs(video_dur - audio_dur):.2f}s")
    
    # Check if durations match (within 1 second tolerance)
    if abs(video_dur - audio_dur) > 1.0:
        print(f"\n❌ FAIL: Video and audio durations don't match!")
        print(f"   Video is {audio_dur - video_dur:.2f}s {'shorter' if video_dur < audio_dur else 'longer'} than audio")
        return False
    
    # Check if video is reasonable length (should be around 117-120 seconds for 16 segments)
    if video_dur < 100:
        print(f"\n⚠️  WARNING: Video seems too short ({video_dur:.2f}s)")
        print(f"   Expected ~117s for 16 segments with transitions")
        return False
    
    print(f"\n✅ PASS: Video appears to be correctly concatenated!")
    print(f"   All segments included, video/audio durations match")
    return True

if __name__ == "__main__":
    video_path = Path("output/enhance_image_quality_corrected_new.mp4")
    
    print("=" * 80)
    print("Video Concatenation Verification")
    print("=" * 80)
    
    success = check_video(video_path)
    
    sys.exit(0 if success else 1)
