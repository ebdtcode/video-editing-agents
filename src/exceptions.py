"""
Custom exceptions for video processing pipeline
"""


class VideoProcessingError(Exception):
    """Base exception for all video processing errors"""
    pass


class TranscriptionError(VideoProcessingError):
    """Raised when transcription fails"""
    pass


class TTSGenerationError(VideoProcessingError):
    """Raised when TTS generation fails"""
    pass


class FFmpegError(VideoProcessingError):
    """Raised when FFmpeg operations fail"""
    pass


class ValidationError(VideoProcessingError):
    """Raised when validation checks fail"""
    pass


class ConfigurationError(VideoProcessingError):
    """Raised when configuration is invalid"""
    pass


class CheckpointError(VideoProcessingError):
    """Raised when checkpoint operations fail"""
    pass


class QualityValidationError(VideoProcessingError):
    """Raised when output quality validation fails"""
    pass
