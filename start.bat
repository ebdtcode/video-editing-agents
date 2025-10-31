@echo off
REM Quick start script for Video Edit Agents
REM Activates virtual environment and shows status

echo.
echo ========================================
echo   Video Edit Agents - GPU Edition
echo ========================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

echo Virtual Environment: ACTIVATED
echo.
echo GPU Status:
python -c "import torch; print(f'  CUDA Available: {torch.cuda.is_available()}'); print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>nul
echo.
echo ========================================
echo Ready to process videos!
echo ========================================
echo.
echo Quick commands:
echo   python video_edit.py --video INPUT.mp4 --output OUTPUT.mp4 --config config_gpu.yaml
echo   python test_gpu_setup.py
echo.
echo See READY_FOR_GPU_TESTING.md for full instructions
echo.
