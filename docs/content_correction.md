# Content Correction Feature

The video processing pipeline now includes advanced content correction capabilities using LLMs to improve transcript quality before TTS generation.

## Features

### 1. Enhanced Filler Word Detection
The pipeline now detects and removes a comprehensive list of filler words:
- Basic fillers: "uh", "um", "ah"
- Common verbal crutches: "like", "you know", "ok", "okay", "so", "well", "right"
- Hedge phrases: "i mean", "you see", "actually", "basically", "literally", "sort of", "kind of"

### 2. LLM-Based Content Correction
After filler removal, segments are optionally corrected using an LLM to:
- **Fix grammar errors** - Correct grammatical mistakes from transcription
- **Rephrase awkward phrasing** - Make segments flow more naturally when spoken
- **Remove repetitions** - Eliminate unnecessary word or phrase repetition
- **Improve clarity** - Make content more concise and understandable

### 3. Supported LLM Providers
- **OpenAI** (default): gpt-4o-mini, gpt-4o, gpt-3.5-turbo
- **Anthropic**: claude-3-5-sonnet-20241022, claude-3-opus-20240229
- **Local** (planned): Support for local LLMs via Ollama

## Configuration

### Basic Configuration (config.yaml)

```yaml
# Content correction settings
content_correction:
  enabled: true  # Enable/disable LLM correction
  mode: "openai"  # 'openai' or 'anthropic'
  model: "gpt-4o-mini"  # Model to use
  # api_key: "your-api-key-here"  # Optional, can use environment variable

  # Correction features (all enabled by default)
  fix_grammar: true
  rephrase_awkward: true
  remove_repetitions: true
  improve_clarity: true

# Filler word detection
fillers:
  mode: "rule-based"
  words:
    - "uh"
    - "um"
    - "ah"
    - "like"
    - "you know"
    - "ok"
    - "okay"
    - "so"
    - "well"
    - "right"
    - "i mean"
    - "you see"
    - "actually"
    - "basically"
    - "literally"
    - "sort of"
    - "kind of"
```

### Environment Variables

Set your API key via environment variable (recommended):

```bash
# For OpenAI
export OPENAI_API_KEY="sk-..."

# For Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Disable Content Correction

If you want to skip LLM-based correction (faster, no API costs):

```yaml
content_correction:
  enabled: false
```

## Pipeline Stages

The video processing pipeline now has 6 stages:

1. **Transcription** - Audio to text using WhisperX
2. **Content Analysis** - Filler word removal and segmentation
3. **Content Correction** - LLM-based grammar and clarity improvements ← NEW
4. **TTS Generation** - Text-to-speech with voice cloning
5. **Video Processing** - Audio replacement and video editing
6. **Final Assembly** - Concatenation and output generation

## Output Files

When content correction is enabled, additional files are generated:

```
temp_segments/
├── cleaned_transcript.txt      # After filler removal
├── corrected_transcript.txt    # After LLM correction
└── correction_report.txt       # Shows all corrections made
```

### Correction Report Example

```
Content Correction Report
================================================================================

Segment: seg_0001
Time: [2.45s - 5.67s]

Original:
So basically what we're gonna do is we're gonna like start by um creating a new file.

Corrected:
What we're going to do is start by creating a new file.

--------------------------------------------------------------------------------

Segment: seg_0003
Time: [8.12s - 11.34s]

Original:
And then, you know, after that we'll we'll add some code to it.

Corrected:
Then we'll add some code to it.

--------------------------------------------------------------------------------

Total segments corrected: 15/26
```

## Cost Considerations

### OpenAI Pricing (as of 2024)
- **gpt-4o-mini**: ~$0.15 per 1M input tokens, $0.60 per 1M output tokens
  - Estimated cost: ~$0.01 per 10 minutes of video
- **gpt-4o**: ~$2.50 per 1M input tokens, $10 per 1M output tokens
  - Estimated cost: ~$0.15 per 10 minutes of video

### Anthropic Pricing
- **Claude 3.5 Sonnet**: ~$3 per 1M input tokens, $15 per 1M output tokens
  - Estimated cost: ~$0.18 per 10 minutes of video

### Tips to Reduce Costs
1. Use `gpt-4o-mini` (cheapest, still excellent quality)
2. Disable features you don't need:
   ```yaml
   content_correction:
     fix_grammar: true
     rephrase_awkward: false  # Disable this
     remove_repetitions: true
     improve_clarity: false   # Disable this
   ```
3. Pre-screen your videos - only use correction for videos that need it
4. Use aggressive filler removal first to reduce text sent to LLM

## Advanced Usage

### Custom Filler Words

Add domain-specific fillers to your configuration:

```yaml
fillers:
  words:
    # ... default fillers ...
    # Add your own:
    - "essentially"
    - "fundamentally"
    - "at the end of the day"
    - "moving forward"
```

### Selective Correction

You can enable/disable specific correction features:

```yaml
content_correction:
  enabled: true
  fix_grammar: true        # Always fix grammar
  rephrase_awkward: false  # Keep original phrasing
  remove_repetitions: true # Remove repetitions
  improve_clarity: false   # Don't rephrase for clarity
```

### Temperature Control

Adjust creativity vs consistency:

```yaml
content_correction:
  temperature: 0.1  # More deterministic (default: 0.3)
  # 0.0 = very consistent, 1.0 = more creative
```

## Troubleshooting

### Issue: "No API key found"
**Solution**: Set environment variable or add to config:
```bash
export OPENAI_API_KEY="sk-..."
```

### Issue: "Content correction disabled"
**Check**: Ensure package is installed:
```bash
pip install openai  # for OpenAI
pip install anthropic  # for Anthropic
```

### Issue: Corrections are too aggressive
**Solution**: Disable specific features:
```yaml
content_correction:
  rephrase_awkward: false
  improve_clarity: false
```

### Issue: High API costs
**Solutions**:
1. Use `gpt-4o-mini` instead of `gpt-4o`
2. Disable correction for short videos
3. Set `enabled: false` when not needed

## Examples

### Before Content Correction
```
So, um, in this video we're gonna, like, talk about how to, you know,
create a React component and, um, we'll show you basically how to, uh,
how to use it in your app.
```

### After Filler Removal
```
In this video we're gonna talk about how to create a React component and
we'll show you how to use it in your app.
```

### After LLM Correction
```
In this video, we'll discuss how to create a React component and demonstrate
how to use it in your application.
```

## Performance Impact

- **Filler removal**: ~1-2 seconds (no impact)
- **LLM correction**: ~5-15 seconds per video (depends on segment count and API speed)
- **Overall**: Adds ~5% to total processing time for typical videos

## Future Enhancements

Planned features:
- [ ] Local LLM support (Ollama, LLaMA)
- [ ] Batch correction for efficiency
- [ ] Custom correction prompts
- [ ] Style preservation (formal/casual tone)
- [ ] Multi-language support
- [ ] Caching for repeated corrections
