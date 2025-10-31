# GPU Setup Instructions for Video Edit Agents

## Current Status

✅ **Virtual Environment**: Created and activated  
✅ **PyTorch with CUDA 12.1**: Installed (2.5.1+cu121)  
✅ **Core Dependencies**: Installed (PyYAML, psutil, tqdm, ffmpeg-python)  
✅ **WhisperX**: Installed  
⚠️ **Chatterbox TTS**: Version conflict detected (see below)

## Known Issue: WhisperX vs Chatterbox TTS Dependency Conflict

**Problem**: 
- WhisperX requires `numpy>=2.0.2`
- Chatterbox TTS requires `numpy<1.26.0`
- These requirements are incompatible

**Workaround Options**:

### Option 1: Use CPU Config (Recommended for Testing)
The pipeline has already been tested on CPU. Use the existing `config_cpu.yaml`:

```bash
python video_edit.py --video input.mp4 --output output.mp4 --config config_cpu.yaml
```

### Option 2: Sequential Installation (For GPU Testing)
1. Install Chatterbox TTS first (with old numpy)
2. Reinstall WhisperX (will upgrade numpy)
3. Test if Chatterbox still works with new numpy

```bash
# This has been partially done - continue from here
pip install chatterbox-tts --no-deps
pip install librosa==0.11.0 s3tokenizer transformers==4.46.3 diffusers==0.29.0
pip install "numpy>=2.0.2"  # Reinstall newer numpy for WhisperX
```

### Option 3: Alternative TTS Backend
Consider using a different TTS backend that doesn't have numpy conflicts:
- Coqui TTS (requires separate installation)
- XTTS (built on Coqui)
- Microsoft Edge TTS (cloud-based, no local dependencies)

## What's Working Now

### Environment Setup ✅
- Python 3.11.9 in virtual environment
- PyTorch 2.5.1 with CUDA 12.1 support
- All core dependencies installed

### GPU Detection ✅
Run this to verify CUDA is available:

```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU Count: {torch.cuda.device_count()}')"
```

### Configuration Files ✅
- `config_gpu.yaml` - For GPU testing (once TTS is resolved)
- `config_cpu.yaml` - For CPU processing (working)
- `config/default_config.yaml` - Default settings

## Next Steps

### Immediate: Test GPU Detection

```bash
python test_gpu_setup.py
```

This will check:
- All imports
- CUDA availability
- FFmpeg installation  
- WhisperX readiness
- Project structure

### After TTS Resolution: Run Full Pipeline

```bash
# With GPU config
python video_edit.py --video videos/your_video.mp4 --output output.mp4 --config config_gpu.yaml

# With CPU config (fallback)
python video_edit.py --video videos/your_video.mp4 --output output.mp4 --config config_cpu.yaml
```

## Files Created

1. **config_gpu.yaml** - GPU-optimized configuration
   - CUDA device for transcription and TTS
   - larger-v2 model for better accuracy
   - Voice cloning enabled
   - 4 parallel workers

2. **requirements_gpu.txt** - GPU-specific dependencies
   - PyTorch with CUDA 12.1
   - All required packages

3. **install_gpu.bat** - Automated installation script
   - Creates venv
   - Installs all dependencies
   - Verifies CUDA

4. **test_gpu_setup.py** - Comprehensive setup testing
   - Checks all imports
   - Tests CUDA operations
   - Validates project structure

## Important Notes

### From Previous Testing (See FINAL_TEST_REPORT.md)

1. **TTS Failures**: 46% of segments failed in previous test
   - Already added None-check in `tts_generation_agent.py` (line 101-111)
   - Sequential processing already enabled (max_workers=1 for TTS)
   
2. **Known Fixes Applied**:
   - ✅ CUDA configuration error - Fixed with `config_cpu.yaml`
   - ✅ Float16 compute type - Fixed in `transcription_agent.py`
   - ⚠️ TTS segment failures - Improved error handling, but may still occur

3. **GPU Benefits**:
   - Transcription: 10-20x faster than CPU
   - TTS: 5-10x faster than CPU
   - Overall pipeline: Expected 5-15x speedup

## System Requirements

- ✅ Python 3.10 or higher (you have 3.11.9)
- ✅ NVIDIA GPU with CUDA support
- ⚠️ NVIDIA drivers with CUDA 12.1+ support
- ✅ FFmpeg in PATH
- ✅ Virtual environment active

## Troubleshooting

### If CUDA is not detected:
1. Update NVIDIA drivers
2. Verify `nvidia-smi` works in command prompt
3. Reinstall PyTorch: `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121`

### If packages conflict:
1. Clear and recreate venv: `rmdir /s venv` then `python -m venv venv`
2. Run `install_gpu.bat` to reinstall everything

### If FFmpeg is missing:
```bash
# Install with winget
winget install ffmpeg

# Or download from https://ffmpeg.org/download.html
# Add to PATH manually
```

## Contact & Support

See `FINAL_TEST_REPORT.md` for detailed test results from previous run.
See `README.md` for general usage instructions.
