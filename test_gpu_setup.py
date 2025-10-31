#!/usr/bin/env python3
"""
Test GPU Setup for Video Edit Agents
Verifies CUDA, PyTorch, and all required dependencies are correctly installed
"""

import sys
from pathlib import Path

def test_imports():
    """Test all required imports"""
    print("=" * 60)
    print("Testing Imports...")
    print("=" * 60)
    
    required_modules = {
        'yaml': 'PyYAML',
        'psutil': 'psutil',
        'tqdm': 'tqdm',
        'torch': 'PyTorch',
        'torchaudio': 'torchaudio',
        'ffmpeg': 'ffmpeg-python',
    }
    
    optional_modules = {
        'whisperx': 'WhisperX',
        'chatterbox': 'Chatterbox TTS',
    }
    
    all_good = True
    
    # Test required modules
    for module, name in required_modules.items():
        try:
            __import__(module)
            print(f"✓ {name:20s} - OK")
        except ImportError as e:
            print(f"✗ {name:20s} - MISSING ({e})")
            all_good = False
    
    # Test optional modules
    print("\nOptional dependencies:")
    for module, name in optional_modules.items():
        try:
            __import__(module)
            print(f"✓ {name:20s} - OK")
        except ImportError as e:
            print(f"⚠ {name:20s} - Not installed ({e})")
    
    return all_good


def test_cuda():
    """Test CUDA availability"""
    print("\n" + "=" * 60)
    print("Testing CUDA Support...")
    print("=" * 60)
    
    try:
        import torch
        
        print(f"PyTorch Version:     {torch.__version__}")
        print(f"CUDA Available:      {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA Version:        {torch.version.cuda}")
            print(f"cuDNN Version:       {torch.backends.cudnn.version()}")
            print(f"GPU Count:           {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"\nGPU {i}:")
                print(f"  Name:              {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"  Compute Capability: {props.major}.{props.minor}")
                print(f"  Total Memory:      {props.total_memory / 1024**3:.2f} GB")
                print(f"  Multi-processors:  {props.multi_processor_count}")
            
            # Test CUDA tensor operation
            print("\nTesting CUDA operations...")
            try:
                x = torch.rand(1000, 1000).cuda()
                y = torch.rand(1000, 1000).cuda()
                z = torch.matmul(x, y)
                print("✓ CUDA tensor operations work!")
                return True
            except Exception as e:
                print(f"✗ CUDA tensor operations failed: {e}")
                return False
        else:
            print("\n⚠ CUDA is not available!")
            print("Possible reasons:")
            print("  1. No NVIDIA GPU detected")
            print("  2. NVIDIA drivers not installed")
            print("  3. PyTorch not installed with CUDA support")
            print("\nTo install PyTorch with CUDA support:")
            print("  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121")
            return False
            
    except ImportError:
        print("✗ PyTorch not installed!")
        return False


def test_ffmpeg():
    """Test FFmpeg availability"""
    print("\n" + "=" * 60)
    print("Testing FFmpeg...")
    print("=" * 60)
    
    import subprocess
    
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✓ {version_line}")
            return True
        else:
            print("✗ FFmpeg command failed")
            return False
            
    except FileNotFoundError:
        print("✗ FFmpeg not found in PATH!")
        print("\nPlease install FFmpeg:")
        print("  Windows: Download from https://ffmpeg.org/download.html")
        print("  Or use: winget install ffmpeg")
        return False


def test_whisperx():
    """Test WhisperX installation"""
    print("\n" + "=" * 60)
    print("Testing WhisperX...")
    print("=" * 60)
    
    try:
        import whisperx
        print(f"✓ WhisperX installed")
        
        # Try to load a model (this will download if needed)
        print("\nNote: First run will download WhisperX models (~1-3GB)")
        print("Models will be cached for future use")
        return True
        
    except ImportError as e:
        print(f"✗ WhisperX not installed: {e}")
        print("\nTo install WhisperX:")
        print("  pip install git+https://github.com/m-bain/whisperX.git")
        return False


def test_chatterbox():
    """Test Chatterbox TTS installation"""
    print("\n" + "=" * 60)
    print("Testing Chatterbox TTS...")
    print("=" * 60)
    
    try:
        from chatterbox.tts import ChatterboxTTS
        print(f"✓ Chatterbox TTS installed")
        
        print("\nNote: First run will download TTS models (~500MB-2GB)")
        print("Models will be cached for future use")
        return True
        
    except ImportError as e:
        print(f"✗ Chatterbox TTS not installed: {e}")
        print("\nTo install Chatterbox TTS:")
        print("  pip install chatterbox-tts")
        return False


def test_project_structure():
    """Test project structure"""
    print("\n" + "=" * 60)
    print("Testing Project Structure...")
    print("=" * 60)
    
    required_files = [
        'video_edit.py',
        'config/default_config.yaml',
        'src/__init__.py',
        'src/config.py',
        'src/agents/orchestrator_agent.py',
    ]
    
    all_good = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            all_good = False
    
    return all_good


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Video Edit Agents - GPU Setup Test")
    print("=" * 60)
    
    results = {
        'Imports': test_imports(),
        'CUDA': test_cuda(),
        'FFmpeg': test_ffmpeg(),
        'WhisperX': test_whisperx(),
        'Chatterbox': test_chatterbox(),
        'Project Structure': test_project_structure(),
    }
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! System is ready for GPU processing.")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Place your video in the videos/ folder")
        print("  2. Run: python video_edit.py --video videos/input.mp4 --output output.mp4 --config config_gpu.yaml")
        return 0
    else:
        print("⚠ Some tests failed. Please fix the issues above.")
        print("=" * 60)
        print("\nQuick fix:")
        print("  Run: install_gpu.bat")
        return 1


if __name__ == "__main__":
    sys.exit(main())
