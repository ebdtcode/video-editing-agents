#!/usr/bin/env python3
"""
Video Edit Agents - Main CLI Entry Point

Open-source video editing pipeline with AI-powered features:
- Automatic transcription using WhisperX
- Filler word removal
- AI voiceover generation using TTS
- Video synchronization and processing
"""

import sys
import argparse
from pathlib import Path
import signal
import warnings

# Suppress PyTorch deprecation warning from Chatterbox library
warnings.filterwarnings('ignore', message='.*torch.backends.cuda.sdp_kernel.*', category=FutureWarning)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import ProcessingConfig
from src.agents.orchestrator_agent import OrchestratorAgent
from src.exceptions import VideoProcessingError
from src.utils.logger import setup_logger


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\nInterrupted by user. Exiting...")
    sys.exit(1)


def create_default_config(config_path: Path):
    """Create a default configuration file"""
    config = ProcessingConfig()
    config.to_yaml(str(config_path))
    print(f"Created default configuration: {config_path}")
    print("Edit this file to customize processing settings")


def main():
    """Main CLI entry point"""
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Video Edit Agents - AI-powered video editing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video with default settings
  python video_edit.py --video input.mp4 --output final.mp4

  # Use existing transcription
  python video_edit.py --video input.mp4 --transcription transcript.json --output final.mp4

  # Use custom configuration
  python video_edit.py --video input.mp4 --config custom.yaml --output final.mp4

  # Resume from checkpoint
  python video_edit.py --video input.mp4 --output final.mp4 --resume

  # Create default config template
  python video_edit.py --create-config config.yaml
        """
    )

    # Main arguments
    parser.add_argument(
        '--video',
        type=Path,
        help='Path to input video file'
    )

    parser.add_argument(
        '--output',
        type=Path,
        help='Path to output video file'
    )

    parser.add_argument(
        '--transcription',
        type=Path,
        help='Path to existing WhisperX JSON transcription (optional)'
    )

    # Configuration
    parser.add_argument(
        '--config',
        type=Path,
        help='Path to configuration YAML file (optional)'
    )

    parser.add_argument(
        '--create-config',
        type=Path,
        metavar='PATH',
        help='Create a default configuration file and exit'
    )

    # Processing options
    parser.add_argument(
        '--temp-dir',
        type=Path,
        help='Temporary working directory (default: ./temp_segments)'
    )

    parser.add_argument(
        '--keep-intermediates',
        action='store_true',
        help='Keep intermediate files after processing'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint if available'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite output file if it exists'
    )

    # TTS options
    parser.add_argument(
        '--no-voice-cloning',
        action='store_true',
        help='Disable voice cloning (use default TTS voice)'
    )

    parser.add_argument(
        '--tts-backend',
        choices=['chatterbox', 'coqui'],
        help='TTS backend to use (default: chatterbox)'
    )

    # Video options
    parser.add_argument(
        '--sync-mode',
        choices=['retime', 'stretch'],
        help='Video synchronization mode (default: retime)'
    )

    parser.add_argument(
        '--no-transitions',
        action='store_true',
        help='Disable transition effects between segments'
    )

    # Logging
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )

    parser.add_argument(
        '--log-file',
        type=Path,
        help='Log file path (default: video_processing.log)'
    )

    # Verbosity
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output (equivalent to --log-level DEBUG)'
    )

    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress non-error output'
    )

    # Version
    parser.add_argument(
        '--version',
        action='version',
        version='Video Edit Agents v1.0.0'
    )

    args = parser.parse_args()

    # Handle config creation
    if args.create_config:
        create_default_config(args.create_config)
        return 0

    # Validate required arguments
    if not args.video:
        parser.error("--video is required")
    if not args.output:
        parser.error("--output is required")

    try:
        # Load configuration
        if args.config:
            print(f"Loading configuration from {args.config}")
            config = ProcessingConfig.from_yaml(str(args.config))
        else:
            print("Using default configuration")
            config = ProcessingConfig()

        # Override config with CLI arguments
        if args.temp_dir:
            config.output.temp_dir = str(args.temp_dir)

        if args.keep_intermediates:
            config.output.keep_intermediates = True

        if args.no_voice_cloning:
            config.tts.voice_cloning = False

        if args.tts_backend:
            config.tts.backend = args.tts_backend

        if args.sync_mode:
            config.video.sync_mode = args.sync_mode

        if args.no_transitions:
            config.video.transitions.enabled = False

        # Set logging level
        if args.verbose:
            config.logging.level = 'DEBUG'
        elif args.quiet:
            config.logging.level = 'ERROR'
        elif args.log_level:
            config.logging.level = args.log_level

        if args.log_file:
            config.logging.file = str(args.log_file)

        # Initialize orchestrator
        print("\nInitializing Video Edit Agents...")
        orchestrator = OrchestratorAgent(config)

        # Process video
        print(f"\nProcessing video: {args.video}")
        print(f"Output will be saved to: {args.output}\n")

        result = orchestrator.process_video(
            video_path=args.video,
            output_path=args.output,
            transcription_json=args.transcription,
            overwrite=args.overwrite,
            resume=args.resume
        )

        # Generate report
        report_path = args.output.parent / f"{args.output.stem}_report.txt"
        orchestrator.generate_report(result, report_path)

        # Display results
        if result.success:
            print("\n" + "=" * 60)
            print("SUCCESS! Video processing completed")
            print("=" * 60)
            print(f"Output video: {result.output_video}")
            print(f"Processing time: {result.processing_time:.2f}s")
            print(f"Segments processed: {result.segments_processed}")

            if result.filler_stats:
                print(f"\nFiller words removed: {result.filler_stats.get('total_fillers', 0)}")
                print(f"Removal rate: {result.filler_stats.get('removal_rate', 0):.1%}")

            print(f"\nDetailed report: {report_path}")
            return 0
        else:
            print("\n" + "=" * 60)
            print("FAILED: Video processing encountered errors")
            print("=" * 60)

            for error in result.errors:
                print(f"ERROR: {error}")

            print(f"\nSee log file for details: {config.logging.file}")
            print(f"Report: {report_path}")
            return 1

    except VideoProcessingError as e:
        print(f"\nERROR: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
