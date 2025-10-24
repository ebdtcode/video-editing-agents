# Using Ollama for FREE Local Content Correction

Run LLM-based content correction completely free, locally, and offline using Ollama!

## Why Ollama?

- ✅ **100% Free** - No API costs ever
- ✅ **Private** - Your data never leaves your machine
- ✅ **Fast** - No network latency
- ✅ **Offline** - Works without internet after model download
- ✅ **Easy** - Simple installation and usage

## Quick Start (5 minutes)

### 1. Install Ollama

**macOS / Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download from: https://ollama.com/download

**Verify installation:**
```bash
ollama --version
```

### 2. Download a Model

Choose one based on your hardware:

**For 8GB+ RAM (Recommended):**
```bash
ollama pull llama3.1:8b
```

**For 4-8GB RAM (Faster, still good quality):**
```bash
ollama pull llama3.2:3b
```

**Alternative options:**
```bash
ollama pull mistral:7b     # Excellent quality
ollama pull phi3:mini      # Microsoft's small model
```

### 3. Test It

```bash
ollama run llama3.2:3b "Fix this grammar: So um I was like going to the store and stuff"
```

Expected output:
```
I was going to the store.
```

### 4. Configure Your Pipeline

Use the included free configuration:

```bash
cp config_ollama_free.yaml config.yaml
```

Or manually set in your `config.yaml`:

```yaml
content_correction:
  enabled: true
  mode: "ollama"
  model: "llama3.2:3b"  # or llama3.1:8b, mistral:7b, etc.
```

### 5. Run Your Video Processing

```bash
python video_edit.py input.mp4 output.mp4
```

That's it! No API keys needed!

## Model Recommendations

| Model | Size | RAM Needed | Speed | Quality | Best For |
|-------|------|------------|-------|---------|----------|
| **llama3.2:3b** | 2GB | 4GB+ | ⚡⚡⚡ Fast | Good | Quick processing, limited RAM |
| **llama3.1:8b** | 4.7GB | 8GB+ | ⚡⚡ Moderate | Excellent | Best balance |
| **mistral:7b** | 4.1GB | 8GB+ | ⚡⚡ Moderate | Excellent | Alternative to LLaMA |
| **phi3:mini** | 2.3GB | 4GB+ | ⚡⚡⚡ Fast | Good | Microsoft model, compact |
| **llama3.1:70b** | 40GB | 64GB+ | ⚡ Slow | Best | Production quality (if you have the RAM) |

**Recommended for most users: `llama3.1:8b`**

## Performance Comparison

### Speed (per 10-minute video)

| Backend | Correction Time | Cost |
|---------|----------------|------|
| OpenAI gpt-4o-mini | ~5 seconds | $0.01 |
| Ollama llama3.2:3b | ~15 seconds | $0.00 |
| Ollama llama3.1:8b | ~30 seconds | $0.00 |

### Quality Comparison

**Input:**
```
So, um, in this video we're gonna, like, talk about how to create a React component.
```

**OpenAI gpt-4o-mini:**
```
In this video, we'll discuss how to create a React component.
```

**Ollama llama3.1:8b:**
```
In this video, we'll talk about creating a React component.
```

**Ollama llama3.2:3b:**
```
In this video, we will discuss creating a React component.
```

**Quality rating: OpenAI = 10/10, LLaMA 3.1 = 9/10, LLaMA 3.2 = 8/10**

## Advanced Configuration

### Custom Ollama Host

If running Ollama on a different machine:

```yaml
content_correction:
  mode: "ollama"
  model: "llama3.1:8b"
  api_key: "http://192.168.1.100:11434"  # Remote Ollama server
```

Or use environment variable:
```bash
export OLLAMA_HOST="http://192.168.1.100:11434"
```

### Temperature Control

```yaml
content_correction:
  mode: "ollama"
  temperature: 0.1  # More consistent (default: 0.3)
  # 0.0 = very deterministic
  # 0.3 = balanced (recommended)
  # 0.7 = more creative
```

### Enable/Disable Features

Save processing time by only using what you need:

```yaml
content_correction:
  enabled: true
  mode: "ollama"
  fix_grammar: true          # Always fix grammar
  rephrase_awkward: false    # Skip rephrasing (faster)
  remove_repetitions: true   # Remove repetitions
  improve_clarity: false     # Skip clarity improvements (faster)
```

## Troubleshooting

### Issue: "Cannot connect to Ollama"

**Solution 1:** Make sure Ollama is running
```bash
ollama serve
```

**Solution 2:** Check if port 11434 is in use
```bash
lsof -i :11434
```

**Solution 3:** Test connection
```bash
curl http://localhost:11434/api/tags
```

### Issue: "Model not found"

**Solution:** Download the model first
```bash
ollama pull llama3.2:3b
```

**Verify it's downloaded:**
```bash
ollama list
```

### Issue: Ollama is slow

**Solutions:**
1. **Use a smaller model**: Switch from `llama3.1:8b` to `llama3.2:3b`
2. **Check RAM usage**: `top` or Activity Monitor
3. **Use GPU**: Ollama automatically uses GPU if available
4. **Close other apps**: Free up system resources

### Issue: Out of memory

**Solutions:**
1. **Use smaller model**:
   ```bash
   ollama pull llama3.2:3b  # Only 2GB
   ```
2. **Close other applications**
3. **Increase swap space** (Linux)
4. **Consider cloud-based Ollama** (see below)

## Using Ollama in the Cloud (Still Free!)

Run Ollama on a free cloud instance:

### Option 1: Google Colab (Free GPU)
```python
# In Colab notebook
!curl -fsSL https://ollama.com/install.sh | sh
!ollama serve &
!ollama pull llama3.1:8b
```

### Option 2: Vast.ai (Pay per use, ~$0.10/hr)
Rent a GPU instance and install Ollama.

### Option 3: Your Own Server
Run Ollama on a home server and access remotely.

## Comparing Costs

### 100 videos (10 minutes each)

| Backend | Total Cost | Monthly |
|---------|-----------|---------|
| OpenAI gpt-4o-mini | $1.00 | $12/year |
| OpenAI gpt-4o | $15.00 | $180/year |
| **Ollama (local)** | **$0.00** | **$0.00** |

**Savings with Ollama: $12-180/year**

## Best Practices

### 1. Choose the Right Model

**For most users:**
```yaml
model: "llama3.1:8b"  # Best balance
```

**For speed:**
```yaml
model: "llama3.2:3b"  # Fastest
```

**For quality:**
```yaml
model: "llama3.1:70b"  # Best (requires 64GB RAM)
```

### 2. Pre-download Models

Download models before processing:
```bash
ollama pull llama3.1:8b
ollama pull llama3.2:3b
```

### 3. Keep Ollama Updated

```bash
# Check for updates
ollama --version

# Update Ollama
curl -fsSL https://ollama.com/install.sh | sh
```

### 4. Monitor Performance

Watch resource usage during processing:
```bash
# macOS/Linux
top

# Or use htop
htop
```

## Examples

### Minimal Config (Fastest)

```yaml
content_correction:
  enabled: true
  mode: "ollama"
  model: "llama3.2:3b"
  temperature: 0.1
  fix_grammar: true
  rephrase_awkward: false
  remove_repetitions: false
  improve_clarity: false
```

**Result:** Grammar fixes only, ~10 seconds per video

### Balanced Config (Recommended)

```yaml
content_correction:
  enabled: true
  mode: "ollama"
  model: "llama3.1:8b"
  temperature: 0.3
  fix_grammar: true
  rephrase_awkward: true
  remove_repetitions: true
  improve_clarity: true
```

**Result:** Full corrections, ~30 seconds per video

### Quality Config (Slowest)

```yaml
content_correction:
  enabled: true
  mode: "ollama"
  model: "mistral:7b"  # Alternative high-quality model
  temperature: 0.2
  max_tokens: 1000
  fix_grammar: true
  rephrase_awkward: true
  remove_repetitions: true
  improve_clarity: true
```

**Result:** Best quality, ~45 seconds per video

## FAQ

### Can I use multiple models?

Not simultaneously, but you can switch between runs:

```bash
# Try with small model first
python video_edit.py input.mp4 output1.mp4 --config config_fast.yaml

# If quality isn't good enough, try larger model
python video_edit.py input.mp4 output2.mp4 --config config_quality.yaml
```

### Does it work offline?

Yes! After downloading models:
```bash
ollama pull llama3.1:8b
```

You can disconnect from internet and it will still work.

### Can I use commercial models?

Ollama models are open source with permissive licenses:
- **LLaMA 3.x**: Meta License (commercial use allowed)
- **Mistral**: Apache 2.0
- **Phi-3**: MIT License

Check each model's license for specifics.

### How do I uninstall?

**Remove Ollama:**
```bash
# macOS
rm -rf /usr/local/bin/ollama
rm -rf ~/.ollama

# Linux
sudo rm /usr/bin/ollama
rm -rf ~/.ollama
```

**Remove models** (to free disk space):
```bash
ollama rm llama3.1:8b
ollama rm llama3.2:3b
```

## Get Help

- **Ollama Docs**: https://ollama.com/docs
- **Model Library**: https://ollama.com/library
- **GitHub**: https://github.com/ollama/ollama
- **Discord**: https://discord.gg/ollama

## Next Steps

1. ✅ Install Ollama
2. ✅ Download a model
3. ✅ Update your config
4. ✅ Process a test video
5. Compare quality with original
6. Adjust settings as needed

**You're now running LLM content correction completely free!**
