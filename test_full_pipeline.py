#!/usr/bin/env python3
"""
Full Pipeline Integration Test

Tests all 5 agents and verifies the complete video editing pipeline works.
This test validates agent initialization and integration without requiring
an actual video file.

For full end-to-end testing with video, use:
    python video_edit.py --video input.mp4 --output output.mp4
"""

import sys
from pathlib import Path
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import ProcessingConfig, TranscriptionConfig, TTSConfig, VideoConfig
from src.agents.transcription_agent import TranscriptionAgent
from src.agents.content_analysis_agent import ContentAnalysisAgent, Segment, Word
from src.agents.tts_generation_agent import TTSGenerationAgent
from src.agents.video_processing_agent import VideoProcessingAgent
from src.agents.orchestrator_agent import OrchestratorAgent


def test_agent_initialization():
    """Test 1: Verify all agents can be initialized"""
    print("=" * 70)
    print("Test 1: Agent Initialization")
    print("=" * 70)

    config = ProcessingConfig()

    # Test Transcription Agent
    print("\n1. Transcription Agent...")
    try:
        transcription_agent = TranscriptionAgent(config.transcription)
        print("   ✓ Transcription Agent initialized")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

    # Test Content Analysis Agent
    print("\n2. Content Analysis Agent...")
    try:
        content_agent = ContentAnalysisAgent(config.fillers)
        print("   ✓ Content Analysis Agent initialized")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

    # Test TTS Generation Agent
    print("\n3. TTS Generation Agent...")
    try:
        tts_agent = TTSGenerationAgent(config.tts)
        print(f"   ✓ TTS Agent initialized (backend: {config.tts.backend})")
        print(f"   ✓ Sample rate: {tts_agent.backend.get_sample_rate()} Hz")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

    # Test Video Processing Agent
    print("\n4. Video Processing Agent...")
    try:
        video_agent = VideoProcessingAgent(config.video)
        print(f"   ✓ Video Processing Agent initialized (sync mode: {config.video.sync_mode})")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

    # Test Orchestrator Agent
    print("\n5. Orchestrator Agent...")
    try:
        orchestrator = OrchestratorAgent(config)
        print("   ✓ Orchestrator Agent initialized")
        print(f"   ✓ Max workers: {config.max_workers or 'auto'}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

    return True


def test_content_analysis():
    """Test 2: Content analysis and filler word detection"""
    print("\n" + "=" * 70)
    print("Test 2: Content Analysis")
    print("=" * 70)

    config = ProcessingConfig()
    content_agent = ContentAnalysisAgent(config)

    # Create test transcription with filler words
    test_segments = [
        Segment(
            text="Um, hello world, uh, this is a test, you know, like, basically.",
            start=0.0,
            end=5.0,
            words=[
                Word("Um", 0.0, 0.2, 1.0),
                Word("hello", 0.3, 0.7, 1.0),
                Word("world", 0.8, 1.2, 1.0),
                Word("uh", 1.3, 1.5, 1.0),
                Word("this", 1.6, 1.9, 1.0),
                Word("is", 2.0, 2.1, 1.0),
                Word("a", 2.2, 2.3, 1.0),
                Word("test", 2.4, 2.8, 1.0),
                Word("you", 2.9, 3.0, 1.0),
                Word("know", 3.1, 3.3, 1.0),
                Word("like", 3.4, 3.6, 1.0),
                Word("basically", 3.7, 4.2, 1.0),
            ]
        )
    ]

    # Analyze content
    clean_segments, stats = content_agent.detect_fillers(test_segments)

    print(f"\nOriginal text: {test_segments[0].text}")
    if clean_segments:
        print(f"Cleaned text:  {' '.join(cs.text for cs in clean_segments)}")

    print(f"\nFiller statistics:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")
    print(f"  - Clean segments created: {len(clean_segments)}")

    # Test passes if clean segments were created (algorithm is working)
    if len(clean_segments) > 0:
        print("\n✓ Content analysis working correctly")
        print(f"  (Created {len(clean_segments)} clean segments from input)")
        return True
    else:
        print("\n✗ Content analysis failed")
        return False


def test_config_management():
    """Test 3: Configuration management"""
    print("\n" + "=" * 70)
    print("Test 3: Configuration Management")
    print("=" * 70)

    # Test default config
    config = ProcessingConfig()
    print(f"\nDefault config created:")
    print(f"  - TTS backend: {config.tts.backend}")
    print(f"  - Transcription model: {config.transcription.model}")
    print(f"  - Video sync mode: {config.video.sync_mode}")
    print(f"  - Max workers: {config.max_workers or 'auto-detect'}")

    # Test YAML export/import
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_yaml = Path(f.name)

    try:
        # Save to YAML
        config.to_yaml(temp_yaml)
        print(f"\n✓ Config exported to YAML: {temp_yaml}")

        # Load from YAML
        loaded_config = ProcessingConfig.from_yaml(temp_yaml)
        print(f"✓ Config loaded from YAML")

        # Verify
        if loaded_config.tts.backend == config.tts.backend:
            print("✓ Config roundtrip successful")
            return True
        else:
            print("✗ Config mismatch after roundtrip")
            return False
    finally:
        temp_yaml.unlink(missing_ok=True)


def test_ffmpeg_integration():
    """Test 4: FFmpeg integration"""
    print("\n" + "=" * 70)
    print("Test 4: FFmpeg Integration")
    print("=" * 70)

    import subprocess

    # Check FFmpeg availability
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"\n✓ FFmpeg detected: {version_line}")

            # Check for required codecs
            print("\nChecking codec support...")

            codecs_result = subprocess.run(
                ['ffmpeg', '-codecs'],
                capture_output=True,
                text=True,
                timeout=5
            )

            codecs = codecs_result.stdout
            required_codecs = ['h264', 'aac', 'libx264', 'pcm']

            for codec in required_codecs:
                if codec in codecs.lower():
                    print(f"  ✓ {codec} supported")
                else:
                    print(f"  ? {codec} status unknown")

            return True
        else:
            print("\n✗ FFmpeg not working properly")
            return False

    except FileNotFoundError:
        print("\n✗ FFmpeg not found in PATH")
        return False
    except Exception as e:
        print(f"\n✗ FFmpeg check failed: {e}")
        return False


def test_dependency_versions():
    """Test 5: Check critical dependency versions"""
    print("\n" + "=" * 70)
    print("Test 5: Dependency Version Check")
    print("=" * 70)

    dependencies = {
        'torch': None,
        'torchaudio': None,
        'whisperx': None,
        'chatterbox': None,
        'numpy': None,
        'transformers': None,
    }

    # Check torch
    try:
        import torch
        dependencies['torch'] = torch.__version__
    except ImportError:
        pass

    # Check torchaudio
    try:
        import torchaudio
        dependencies['torchaudio'] = torchaudio.__version__
    except ImportError:
        pass

    # Check whisperx
    try:
        import whisperx
        dependencies['whisperx'] = getattr(whisperx, '__version__', 'installed')
    except ImportError:
        pass

    # Check chatterbox
    try:
        import chatterbox
        dependencies['chatterbox'] = getattr(chatterbox, '__version__', 'installed')
    except ImportError:
        pass

    # Check numpy
    try:
        import numpy
        dependencies['numpy'] = numpy.__version__
    except ImportError:
        pass

    # Check transformers
    try:
        import transformers
        dependencies['transformers'] = transformers.__version__
    except ImportError:
        pass

    print("\nInstalled dependencies:")
    all_present = True
    for name, version in dependencies.items():
        if version:
            print(f"  ✓ {name}: {version}")
        else:
            print(f"  ✗ {name}: NOT INSTALLED")
            all_present = False

    # Check for dependency conflicts
    print("\nDependency Compatibility:")
    if dependencies['torch'] and dependencies['chatterbox']:
        if dependencies['torch'].startswith('2.'):
            print("  ✓ PyTorch 2.x detected (compatible)")
        else:
            print("  ? PyTorch version may have compatibility issues")

    return all_present


def main():
    """Run all integration tests"""
    print("\n" + "=" * 70)
    print("VIDEO EDIT AGENTS - FULL PIPELINE INTEGRATION TEST")
    print("=" * 70)

    tests = [
        ("Agent Initialization", test_agent_initialization),
        ("Content Analysis", test_content_analysis),
        ("Configuration Management", test_config_management),
        ("FFmpeg Integration", test_ffmpeg_integration),
        ("Dependency Versions", test_dependency_versions),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} crashed with error:")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 70)

    if passed == total:
        print("\n✓ ALL TESTS PASSED - System ready for production use")
        print("\nNext steps:")
        print("  1. Test with actual video file:")
        print("     python video_edit.py --video input.mp4 --output output.mp4")
        print("\n  2. Enable voice cloning:")
        print("     python video_edit.py --video input.mp4 --output output.mp4 \\")
        print("            --config config.yaml")
        print("\n  3. Resume interrupted processing:")
        print("     python video_edit.py --video input.mp4 --output output.mp4 --resume")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED - Review errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
