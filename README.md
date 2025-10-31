# Video Edit Agents

AI-powered video editing pipeline that automatically removes filler words, generates AI voiceovers, and syncs video with new audio.

## Features

- **Automatic Transcription**: Uses WhisperX for accurate speech-to-text with word-level timestamps
- **Filler Word Removal**: Intelligently detects and removes "uh", "um", "ah", and other filler words
- **AI Voiceover**: Generates natural-sounding voiceovers using Chatterbox TTS
- **Voice Cloning**: Optionally clones your voice from the original audio
- **Audio-Video Sync**: Automatically retimes video segments to match TTS audio duration
- **Parallel Processing**: Efficiently processes multiple segments in parallel
- **Checkpoint Support**: Resume processing from where it left off if interrupted
- **Transition Effects**: Smooth crossfade transitions between segments

## Installation

### Prerequisites

- Python 3.10 or higher
- FFmpeg installed and in PATH
- CUDA-capable GPU (optional, for faster processing)

#### Install FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH

### Install Video Edit Agents

1. Clone the repository:
```bash
git clone <repository-url>
cd video-edit-agents
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Install in development mode:
```bash
pip install -e .
```

## Quick Start

### Basic Usage

Process a video with default settings:

```bash
python video_edit.py --video input.mp4 --output final.mp4
```

### Advanced Usage

Use custom configuration:

```bash
# Create config template
python video_edit.py --create-config my_config.yaml

# Edit my_config.yaml to your preferences

# Process video with custom config
python video_edit.py --video input.mp4 --config my_config.yaml --output final.mp4
```

Resume from checkpoint:

```bash
python video_edit.py --video input.mp4 --output final.mp4 --resume
```

Use existing transcription:

```bash
python video_edit.py --video input.mp4 --transcription transcript.json --output final.mp4
```

## Configuration

### Configuration File

Create a custom configuration file:

```bash
python video_edit.py --create-config config.yaml
```

Edit `config.yaml` to customize:

- **Processing settings**: Min/max segment duration, parallel workers
- **Filler words**: List of words to remove
- **Transcription**: Model size, language, device (CPU/GPU)
- **TTS**: Backend, voice cloning, sample rate
- **Video**: Sync mode, codec, quality, transitions
- **Output**: Temporary directory, checkpoint settings

### Command-Line Options

```
Required:
  --video PATH              Input video file
  --output PATH             Output video file

Optional:
  --config PATH             Configuration YAML file
  --transcription PATH      Existing WhisperX JSON transcription
  --temp-dir PATH           Temporary working directory
  --keep-intermediates      Keep temporary files
  --resume                  Resume from checkpoint
  --overwrite               Overwrite existing output

TTS Options:
  --no-voice-cloning        Use default TTS voice
  --tts-backend {chatterbox,coqui}

Video Options:
  --sync-mode {retime,stretch}
  --no-transitions          Disable transition effects

Logging:
  --log-level {DEBUG,INFO,WARNING,ERROR}
  --log-file PATH
  -v, --verbose             Enable debug output
  -q, --quiet               Suppress non-error output
```

## Processing Pipeline

The system processes videos through 5 stages:

### 1. Transcription
- Extracts audio from video
- Transcribes using WhisperX
- Generates word-level timestamps

### 2. Content Analysis
- Detects filler words
- Splits into clean segments
- Optimizes segment lengths

### 3. TTS Generation
- Extracts voice profile (if voice cloning enabled)
- Generates AI voiceover for each segment
- Normalizes audio volume

### 4. Video Processing
- Cuts video segments
- Retimes video to match TTS duration
- Merges video with TTS audio

### 5. Final Assembly
- Concatenates all segments
- Applies transition effects
- Creates final output video

## Examples

### Example 1: Tutorial Video

Remove filler words from a tutorial recording:

```bash
python video_edit.py \
    --video tutorial.mp4 \
    --output tutorial_clean.mp4 \
    --sync-mode retime \
    --log-level INFO
```

### Example 2: Podcast Video

Process podcast with voice cloning:

```bash
python video_edit.py \
    --video podcast.mp4 \
    --output podcast_clean.mp4 \
    --config podcast_config.yaml \
    --keep-intermediates
```

### Example 3: Presentation

Quick processing without transitions:

```bash
python video_edit.py \
    --video presentation.mp4 \
    --output presentation_clean.mp4 \
    --no-transitions \
    --sync-mode stretch
```

### Example 4: Resume Long Video

Process long video with resume support:

```bash
# First run (may be interrupted)
python video_edit.py --video long_video.mp4 --output final.mp4

# Resume after interruption
python video_edit.py --video long_video.mp4 --output final.mp4 --resume
```

## Architecture

### Agent-Based Design

The system uses a modular agent architecture:

- **TranscriptionAgent**: Handles speech-to-text
- **ContentAnalysisAgent**: Analyzes and cleans transcripts
- **TTSGenerationAgent**: Generates AI voiceovers
- **VideoProcessingAgent**: Processes and syncs video
- **OrchestratorAgent**: Coordinates the pipeline

### Key Technologies

- **WhisperX**: State-of-the-art transcription
- **Chatterbox TTS**: Natural-sounding text-to-speech
- **FFmpeg**: Video/audio processing
- **PyTorch**: ML model execution

## Performance

### Processing Speed

Typical processing times on M1 Mac / RTX 3080:

- **10-minute video**: ~5-10 minutes
- **30-minute video**: ~15-30 minutes
- **60-minute video**: ~30-60 minutes

Speed depends on:
- Hardware (CPU/GPU)
- Video resolution
- Transcription model size
- Number of parallel workers

### Optimization Tips

1. **Use GPU**: Add `--device cuda` in config for 3-5x speedup
2. **Smaller model**: Use `medium` instead of `large-v2` for faster transcription
3. **Disable voice cloning**: Use `--no-voice-cloning` for simpler processing
4. **More workers**: Increase `max_workers` in config (careful with memory)

## Troubleshooting

### Common Issues

**1. FFmpeg not found**
```
Solution: Install FFmpeg and ensure it's in PATH
```

**2. CUDA out of memory**
```
Solution: Reduce max_workers or use CPU mode
```

**3. Transcription quality poor**
```
Solution: Use larger model (large-v2) or specify language
```

**4. Audio-video desync**
```
Solution: Use sync_mode: retime instead of stretch
```

**5. TTS sounds unnatural**
```
Solution: Enable voice cloning or try different TTS backend
```

### Debug Mode

Enable detailed logging:

```bash
python video_edit.py --video input.mp4 --output final.mp4 --verbose
```

Check logs in `video_processing.log`

## Advanced Features

### Custom Filler Words

Edit `config.yaml`:

```yaml
fillers:
  words:
    - uh
    - um
    - ah
    - like
    - you know
    - basically
    - literally
    - actually
```

### Multiple TTS Backends

Support for different TTS engines:

```yaml
tts:
  backend: chatterbox  # or 'coqui'
```

### Voice Mode Options

The pipeline supports 4 different voice modes for TTS generation:

**1. Default Mode** - Use Chatterbox's built-in default voice (no voice cloning):
```yaml
tts:
  voice_mode: default
```

**2. Auto Mode** - Automatically extract and clone voice from the first 30 seconds of your video:
```yaml
tts:
  voice_mode: auto
```

**3. Custom Mode** - Use a custom voice reference file for voice cloning:
```yaml
tts:
  voice_mode: custom
  voice_reference: "voices/my_voice.wav"
```

**4. Original Mode** - Skip TTS generation entirely and use the original video audio:
```yaml
tts:
  voice_mode: original
```

**When to use each mode:**
- **default**: Quick processing, don't need voice cloning
- **auto**: Want to clone the speaker's voice automatically
- **custom**: Have a high-quality voice sample for better cloning
- **original**: Only removing filler words, want to preserve the original speaker's voice and avoid TTS quality issues

**Recommended workflow for text corrections:**
1. Export transcript: `python export_transcript_for_editing.py`
2. Correct transcript with LLM: `python auto_correct_transcript.py --input editable_transcript.json --provider ollama --model mistral`
3. Regenerate with original audio: Set `voice_mode: original` in config
4. Run: `python regenerate_from_corrections.py --input editable_transcript_corrected.json`

This workflow is much faster and preserves audio quality since it skips TTS generation.

### Video Quality Settings

Adjust output quality:

```yaml
video:
  quality: high  # or 'medium', 'low'
  codec: h264
  bitrate: 5M
```

### Transition Customization

```yaml
video:
  transitions:
    enabled: true
    type: crossfade  # or 'dissolve'
    duration: 0.5  # seconds
```

## Development

### Running Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

### Code Quality

```bash
# Format code
black src/

# Lint
flake8 src/

# Type checking
mypy src/
```

## Limitations

- Requires clean audio input for best results
- Processing time scales with video length
- GPU memory limits maximum parallel workers
- Voice cloning quality depends on reference audio
- Best results with clear, single-speaker videos

## Roadmap

- [ ] ML-based filler detection
- [ ] Support for multiple TTS backends
- [ ] Real-time preview
- [ ] Web UI interface
- [ ] Batch processing mode
- [ ] Custom model fine-tuning
- [ ] Multi-language support improvements

## License

See LICENSE file for details.

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## Support

For issues and questions:
- GitHub Issues
- Documentation: See `/docs` folder

## Acknowledgments

- WhisperX for transcription
- Chatterbox TTS for voice synthesis
- FFmpeg for video processing
