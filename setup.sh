#!/bin/bash

# Video Edit Agents - Setup Script
# Automated setup for the video editing pipeline

set -e  # Exit on error

echo "====================================="
echo "Video Edit Agents - Setup"
echo "====================================="
echo

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.10 or higher required (found $python_version)"
    exit 1
fi
echo "✓ Python $python_version detected"

# Check FFmpeg
echo
echo "Checking FFmpeg..."
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: FFmpeg not found"
    echo
    echo "Please install FFmpeg:"
    echo "  macOS:   brew install ffmpeg"
    echo "  Ubuntu:  sudo apt install ffmpeg"
    echo "  Windows: Download from ffmpeg.org"
    exit 1
fi
echo "✓ FFmpeg detected"

# Check for CUDA (optional)
echo
echo "Checking for CUDA (optional)..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ CUDA detected - GPU acceleration available"
    USE_CUDA=true
else
    echo "ℹ CUDA not detected - will use CPU mode"
    USE_CUDA=false
fi

# Create virtual environment
echo
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists"
    read -p "Recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
        echo "✓ Virtual environment created"
    fi
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "✓ pip upgraded"

# Install dependencies
echo
echo "Installing dependencies..."
echo "This may take several minutes..."
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Create default config
echo
echo "Creating default configuration..."
if [ ! -f "config.yaml" ]; then
    python video_edit.py --create-config config.yaml
    echo "✓ Configuration created: config.yaml"
else
    echo "ℹ Configuration already exists: config.yaml"
fi

# Create directories
echo
echo "Creating directories..."
mkdir -p temp_segments
mkdir -p examples
mkdir -p tests
echo "✓ Directories created"

# Test installation
echo
echo "Testing installation..."
python -c "
import sys
sys.path.insert(0, '.')
from src.config import ProcessingConfig
from src.agents.orchestrator_agent import OrchestratorAgent
print('✓ All imports successful')
"

# Summary
echo
echo "====================================="
echo "Setup Complete!"
echo "====================================="
echo
echo "Next steps:"
echo
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo
echo "2. Edit configuration (optional):"
echo "   vim config.yaml"
echo
echo "3. Process a video:"
echo "   python video_edit.py --video input.mp4 --output final.mp4"
echo
echo "4. See usage guide:"
echo "   cat USAGE_GUIDE.md"
echo
if [ "$USE_CUDA" = false ]; then
    echo "Note: Running in CPU mode. For GPU acceleration, install CUDA."
    echo
fi
echo "For help: python video_edit.py --help"
echo

exit 0
