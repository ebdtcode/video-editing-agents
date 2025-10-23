"""
GPU detection and capability utilities
"""

import subprocess
import re
from pathlib import Path
from typing import Dict, Optional
from src.utils.logger import get_logger


logger = get_logger(__name__)


class GPUCapabilities:
    """GPU capabilities and encoding support"""

    def __init__(self):
        self.has_cuda = False
        self.has_nvenc = False
        self.has_nvidia_gpu = False
        self.gpu_name = None
        self.cuda_version = None
        self.ffmpeg_encoders = []

    def __repr__(self):
        return (
            f"GPUCapabilities(cuda={self.has_cuda}, nvenc={self.has_nvenc}, "
            f"gpu={self.gpu_name})"
        )


def detect_nvidia_gpu() -> Optional[str]:
    """
    Detect NVIDIA GPU using nvidia-smi

    Returns:
        GPU name if found, None otherwise
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5,
            check=False
        )

        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip().split('\n')[0]
            logger.info(f"Detected NVIDIA GPU: {gpu_name}")
            return gpu_name

    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.debug("nvidia-smi not found or timed out")

    return None


def detect_cuda_version() -> Optional[str]:
    """
    Detect CUDA version

    Returns:
        CUDA version string if found, None otherwise
    """
    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            timeout=5,
            check=False
        )

        if result.returncode == 0:
            # Look for CUDA Version in output
            match = re.search(r'CUDA Version:\s+(\d+\.\d+)', result.stdout)
            if match:
                cuda_version = match.group(1)
                logger.info(f"Detected CUDA version: {cuda_version}")
                return cuda_version

    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.debug("Could not detect CUDA version")

    return None


def detect_ffmpeg_encoders() -> list:
    """
    Detect available FFmpeg encoders

    Returns:
        List of available encoder names
    """
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            capture_output=True,
            text=True,
            timeout=10,
            check=False
        )

        if result.returncode == 0:
            encoders = []
            for line in result.stdout.split('\n'):
                # Look for h264_nvenc, hevc_nvenc, etc.
                if 'nvenc' in line.lower() or 'cuda' in line.lower():
                    # Extract encoder name (format: " V..... h264_nvenc ...")
                    parts = line.split()
                    if len(parts) >= 2:
                        encoder_name = parts[1]
                        encoders.append(encoder_name)

            if encoders:
                logger.info(f"Detected GPU encoders: {', '.join(encoders)}")
            return encoders

    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.debug("Could not detect FFmpeg encoders")

    return []


def detect_gpu_capabilities() -> GPUCapabilities:
    """
    Detect all GPU capabilities for video encoding

    Returns:
        GPUCapabilities object with detection results
    """
    caps = GPUCapabilities()

    # Detect NVIDIA GPU
    gpu_name = detect_nvidia_gpu()
    if gpu_name:
        caps.has_nvidia_gpu = True
        caps.gpu_name = gpu_name

    # Detect CUDA
    cuda_version = detect_cuda_version()
    if cuda_version:
        caps.has_cuda = True
        caps.cuda_version = cuda_version

    # Detect FFmpeg GPU encoders
    encoders = detect_ffmpeg_encoders()
    caps.ffmpeg_encoders = encoders

    # Check for NVENC support
    if 'h264_nvenc' in encoders or 'hevc_nvenc' in encoders:
        caps.has_nvenc = True
        logger.info("NVENC hardware encoding available")
    else:
        logger.warning("NVENC not available - will use CPU encoding")

    return caps


def get_optimal_encoder(
    caps: GPUCapabilities,
    codec: str = 'h264',
    fallback: str = 'libx264'
) -> str:
    """
    Get optimal encoder based on GPU capabilities

    Args:
        caps: GPU capabilities
        codec: Desired codec ('h264', 'hevc')
        fallback: Fallback CPU encoder

    Returns:
        Encoder name to use
    """
    if codec == 'h264' and 'h264_nvenc' in caps.ffmpeg_encoders:
        return 'h264_nvenc'
    elif codec == 'hevc' and 'hevc_nvenc' in caps.ffmpeg_encoders:
        return 'hevc_nvenc'
    else:
        logger.info(f"GPU encoder not available, using CPU fallback: {fallback}")
        return fallback


# Global GPU capabilities (cached)
_gpu_caps: Optional[GPUCapabilities] = None


def get_gpu_capabilities(force_refresh: bool = False) -> GPUCapabilities:
    """
    Get cached GPU capabilities (or refresh if needed)

    Args:
        force_refresh: Force re-detection

    Returns:
        GPUCapabilities object
    """
    global _gpu_caps

    if _gpu_caps is None or force_refresh:
        _gpu_caps = detect_gpu_capabilities()

    return _gpu_caps
