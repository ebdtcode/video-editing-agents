@echo off
REM Installation script for GPU-enabled Video Edit Agents
REM Requires: Python 3.10+, CUDA-capable GPU, NVIDIA drivers

echo ====================================
echo Video Edit Agents - GPU Installation
echo ====================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        exit /b 1
    )
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: Failed to upgrade pip
    exit /b 1
)

echo.
echo Installing PyTorch with CUDA 12.1 support...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch with CUDA
    exit /b 1
)

echo.
echo Installing core dependencies...
pip install pyyaml psutil tqdm ffmpeg-python
if errorlevel 1 (
    echo ERROR: Failed to install core dependencies
    exit /b 1
)

echo.
echo Installing WhisperX...
pip install git+https://github.com/m-bain/whisperX.git
if errorlevel 1 (
    echo ERROR: Failed to install WhisperX
    exit /b 1
)

echo.
echo Installing Chatterbox TTS...
pip install chatterbox-tts
if errorlevel 1 (
    echo ERROR: Failed to install Chatterbox TTS
    exit /b 1
)

echo.
echo ====================================
echo Installation Complete!
echo ====================================
echo.
echo Verifying CUDA availability...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo.
echo To test the installation, run:
echo   python test_gpu_setup.py
echo.
echo To process a video:
echo   python video_edit.py --video input.mp4 --output output.mp4 --config config_gpu.yaml
echo.
pause
