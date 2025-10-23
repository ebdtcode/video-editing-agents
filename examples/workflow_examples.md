# Workflow Examples

Real-world examples of using Video Edit Agents for different use cases.

---

## Example 1: Tutorial Video Cleanup

**Scenario:** You recorded a coding tutorial with many "ums" and "uhs".

**Goal:** Remove filler words while maintaining natural flow.

### Step 1: Create Config

```yaml
# tutorial_config.yaml
processing:
  min_segment_duration: 1.0  # Keep longer pauses for emphasis
  max_segment_duration: 20.0

fillers:
  words:
    - uh
    - um
    - ah
    - like
    - so
    - basically

transcription:
  model: large-v2  # Accuracy important for technical terms
  language: en

tts:
  voice_cloning: true  # Maintain your voice
  normalize: true

video:
  sync_mode: retime
  transitions:
    enabled: true
    duration: 0.2  # Subtle transitions
```

### Step 2: Process

```bash
python video_edit.py \
    --video tutorial_raw.mp4 \
    --config tutorial_config.yaml \
    --output tutorial_final.mp4 \
    --log-level INFO
```

### Step 3: Review

```bash
# Check cleaned transcript
cat temp_segments/cleaned_transcript.txt

# Review report
cat tutorial_final_report.txt
```

---

## Example 2: Podcast to Video

**Scenario:** Convert audio podcast to video with generated voiceover.

**Goal:** Create polished video version with smooth audio.

### Step 1: Extract Audio

```bash
# If starting with video
ffmpeg -i podcast_video.mp4 -vn -ar 16000 podcast_audio.wav

# If starting with audio
cp podcast.mp3 podcast_audio.mp3
```

### Step 2: Transcribe

```bash
whisperx podcast_audio.wav \
    --model large-v2 \
    --output_dir transcriptions \
    --language en
```

### Step 3: Configure

```yaml
# podcast_config.yaml
processing:
  min_segment_duration: 2.0  # Natural conversation pauses
  max_workers: 4

fillers:
  words:
    - uh
    - um
    - like
    - you know
    - kind of

tts:
  voice_cloning: true
  sample_rate: 48000  # High quality audio

video:
  sync_mode: retime
  bitrate: 8M
  transitions:
    enabled: true
    type: crossfade
    duration: 0.5
```

### Step 4: Process

```bash
python video_edit.py \
    --video podcast_video.mp4 \
    --transcription transcriptions/podcast_audio.en.json \
    --config podcast_config.yaml \
    --output podcast_clean.mp4 \
    --keep-intermediates
```

---

## Example 3: Presentation Recording

**Scenario:** Clean up conference presentation recording.

**Goal:** Fast processing, professional output.

### Config

```yaml
# presentation_config.yaml
processing:
  max_segment_duration: 30.0
  max_workers: 6

fillers:
  words:
    - uh
    - um
    - ah
    - like
    - basically
    - actually
    - essentially

transcription:
  model: medium  # Faster than large
  device: cuda

tts:
  voice_cloning: false  # Use default voice for speed
  normalize: true

video:
  sync_mode: retime
  quality: high
  transitions:
    enabled: false  # Clean cuts
```

### Processing

```bash
python video_edit.py \
    --video presentation.mp4 \
    --config presentation_config.yaml \
    --output presentation_final.mp4 \
    --overwrite
```

---

## Example 4: YouTube Content Creation

**Scenario:** Create polished YouTube video from raw footage.

**Goal:** Maximum quality, engaging delivery.

### Multi-Stage Workflow

#### Stage 1: Draft Processing

```bash
# Quick pass to check results
python video_edit.py \
    --video raw_footage.mp4 \
    --config fast_config.yaml \
    --output draft.mp4
```

**fast_config.yaml:**
```yaml
transcription:
  model: small
video:
  transitions:
    enabled: false
tts:
  voice_cloning: false
```

#### Stage 2: Review and Adjust

```bash
# Review draft
vlc draft.mp4

# Adjust filler words based on review
vim final_config.yaml
```

#### Stage 3: Final Production

```bash
# High-quality final render
python video_edit.py \
    --video raw_footage.mp4 \
    --transcription transcriptions/raw_footage.en.json \
    --config final_config.yaml \
    --output youtube_final.mp4
```

**final_config.yaml:**
```yaml
processing:
  min_segment_duration: 0.8
  max_segment_duration: 25.0

fillers:
  words:
    - uh
    - um
    - ah
    - like
    - you know
    - sort of
    - kind of
    - basically

transcription:
  model: large-v2
  device: cuda

tts:
  voice_cloning: true
  sample_rate: 48000
  normalize: true

video:
  sync_mode: retime
  codec: h264
  bitrate: 8M
  quality: high
  transitions:
    enabled: true
    type: crossfade
    duration: 0.3

output:
  keep_intermediates: true
```

---

## Example 5: Long-Form Content (1+ hour)

**Scenario:** Process 90-minute webinar recording.

**Goal:** Reliable processing with resume support.

### Strategy

Use checkpoint/resume feature for reliability:

```bash
# Initial processing
python video_edit.py \
    --video webinar_90min.mp4 \
    --output webinar_final.mp4 \
    --config webinar_config.yaml \
    --resume \
    --log-level INFO

# If interrupted, simply re-run same command
# Progress will resume from last checkpoint
```

### Config Optimization

```yaml
# webinar_config.yaml
processing:
  max_workers: 3  # Conservative to avoid OOM
  min_segment_duration: 1.5
  max_segment_duration: 30.0

transcription:
  model: medium  # Balance speed/accuracy
  device: cuda

tts:
  voice_cloning: true

video:
  sync_mode: retime
  transitions:
    enabled: true

output:
  checkpoint: true  # Critical for long videos
  keep_intermediates: false  # Save disk space
```

### Monitoring Progress

```bash
# In another terminal, monitor progress
tail -f video_processing.log

# Check checkpoint status
cat temp_segments/checkpoints/pipeline.json
```

---

## Example 6: Batch Processing Multiple Files

**Scenario:** Process multiple training videos with same settings.

### Batch Script

```bash
#!/bin/bash
# batch_process.sh

CONFIG="batch_config.yaml"
INPUT_DIR="raw_videos"
OUTPUT_DIR="processed_videos"

mkdir -p "$OUTPUT_DIR"

for video in "$INPUT_DIR"/*.mp4; do
    filename=$(basename "$video")
    output="$OUTPUT_DIR/${filename%.mp4}_clean.mp4"

    echo "Processing: $filename"

    python video_edit.py \
        --video "$video" \
        --config "$CONFIG" \
        --output "$output" \
        --log-level INFO

    if [ $? -eq 0 ]; then
        echo "✓ Success: $filename"
    else
        echo "✗ Failed: $filename"
    fi
done
```

### Usage

```bash
chmod +x batch_process.sh
./batch_process.sh
```

---

## Example 7: Custom Filler Words for Domain

**Scenario:** Technical presentation with domain-specific fillers.

### Custom Filler List

```yaml
# technical_config.yaml
fillers:
  words:
    # Standard fillers
    - uh
    - um
    - ah
    - like

    # Technical filler phrases
    - you know
    - kind of
    - sort of
    - basically
    - actually
    - essentially
    - obviously
    - clearly

    # Specific to tech talks
    - right
    - okay
    - so yeah
    - let's see
```

---

## Example 8: Voice Cloning from Specific Segment

**Scenario:** Clone voice from clearest part of audio.

### Custom Processing

```python
# custom_voice_clone.py
from pathlib import Path
from src.agents.tts_generation_agent import TTSGenerationAgent
from src.config import TTSConfig

# Initialize TTS
config = TTSConfig(voice_cloning=True, device="cuda")
tts_agent = TTSGenerationAgent(config)

# Extract voice from specific clean segment
reference_audio = Path("reference_voice.wav")
tts_agent.extract_voice_profile(
    audio_path=Path("original_audio.wav"),
    start=120.0,  # 2 minutes in
    end=150.0,    # 30 seconds of clean audio
    output_path=reference_audio
)

print(f"Voice profile saved: {reference_audio}")
```

Then use in processing:

```bash
python video_edit.py \
    --video input.mp4 \
    --output final.mp4 \
    --keep-intermediates

# TTS agent will use reference_voice.wav if found
```

---

## Example 9: Preview First

**Scenario:** Preview results before full processing.

### Quick Preview Workflow

```bash
# 1. Extract first 2 minutes
ffmpeg -i full_video.mp4 -t 120 -c copy preview.mp4

# 2. Process preview
python video_edit.py \
    --video preview.mp4 \
    --config test_config.yaml \
    --output preview_processed.mp4

# 3. Review results
vlc preview_processed.mp4

# 4. Adjust config based on preview

# 5. Process full video
python video_edit.py \
    --video full_video.mp4 \
    --config final_config.yaml \
    --output final.mp4
```

---

## Example 10: Integration with Video Editor

**Scenario:** Use as preprocessing step for Final Cut Pro / Premiere.

### Workflow

```bash
# 1. Process video to remove fillers
python video_edit.py \
    --video raw.mp4 \
    --output clean_audio_video.mp4 \
    --keep-intermediates

# 2. Export cleaned transcript
cat temp_segments/cleaned_transcript.txt > script.txt

# 3. Import to video editor:
#    - clean_audio_video.mp4 as main track
#    - script.txt for subtitles/notes
#    - Original raw.mp4 for B-roll

# 4. Final edits in video editor
```

---

## Performance Comparison

### Different Configurations

**Test video: 10 minutes, 720p**

| Config | Transcription | TTS | Processing Time | Quality |
|--------|--------------|-----|-----------------|---------|
| Fast | small model | no cloning | 3m 20s | Good |
| Balanced | medium model | cloning | 6m 45s | Very Good |
| Quality | large-v2 | cloning | 12m 30s | Excellent |

---

## Tips and Tricks

### 1. Testing Configuration

Always test on a short clip first:

```bash
ffmpeg -i long_video.mp4 -t 60 -c copy test_60s.mp4
python video_edit.py --video test_60s.mp4 --output test.mp4
```

### 2. Parallel Transcription

Transcribe separately while working on config:

```bash
# Terminal 1: Transcribe
whisperx original.wav --model large-v2 --output_dir transcriptions

# Terminal 2: Test configs
python video_edit.py --video test.mp4 --config v1.yaml --output out1.mp4
python video_edit.py --video test.mp4 --config v2.yaml --output out2.mp4
```

### 3. Disk Space Management

For large projects:

```bash
# Process with checkpoints
python video_edit.py --video large.mp4 --output final.mp4 --resume

# Clean up after success
rm -rf temp_segments/
rm video_processing.log
```

### 4. Quality Checks

```bash
# Check audio levels
ffmpeg -i final.mp4 -af volumedetect -f null -

# Compare durations
ffprobe -v error -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 original.mp4
ffprobe -v error -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 final.mp4
```

---

## Troubleshooting Specific Scenarios

### Scenario: Multiple Speakers

**Problem:** Voice cloning uses wrong speaker

**Solution:** Disable voice cloning or extract reference from single speaker

```yaml
tts:
  voice_cloning: false
```

### Scenario: Background Music

**Problem:** Music interferes with transcription

**Solution:** Use audio track without music, or extract music separately

```bash
# Extract voice track only
ffmpeg -i video.mp4 -map 0:a:0 voice_only.wav

# Process with voice only
python video_edit.py --video video.mp4 --transcription voice.json --output clean.mp4

# Re-add music in post
```

### Scenario: Heavy Accent

**Problem:** Poor transcription accuracy

**Solution:** Specify language, use larger model

```yaml
transcription:
  model: large-v2
  language: en  # or specific variant like en-GB
```
