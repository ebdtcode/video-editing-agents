"""
Setup script for Video Edit Agents
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="video-edit-agents",
    version="1.0.0",
    description="AI-powered video editing pipeline with automatic transcription and voice synthesis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="devos",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pyyaml>=6.0",
        "psutil>=5.9.0",
        "tqdm>=4.65.0",
        "chatterbox-tts>=0.1.0",
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "ffmpeg-python>=0.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
        "coqui": [
            "coqui-tts>=0.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "video-edit=video_edit:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
