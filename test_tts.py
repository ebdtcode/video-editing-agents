#!/usr/bin/env python3
"""
Test script to verify Chatterbox TTS integration works
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import TTSConfig
from src.agents.tts_generation_agent import TTSGenerationAgent
from src.agents.content_analysis_agent import CleanSegment, Word

def test_tts_basic():
    """Test basic TTS generation"""
    print("=" * 60)
    print("Test 1: Basic TTS Generation")
    print("=" * 60)

    # Create config (CPU mode for safety)
    config = TTSConfig(
        backend="chatterbox",
        voice_cloning=False,
        device="cpu"
    )

    print(f"Config: {config}")

    # Initialize TTS agent
    print("\nInitializing TTS Agent...")
    tts_agent = TTSGenerationAgent(config)
    print("✓ TTS Agent initialized successfully")

    # Create a test segment
    words = [
        Word("Hello", 0.0, 0.5, 1.0),
        Word("world", 0.5, 1.0, 1.0),
        Word("this", 1.0, 1.2, 1.0),
        Word("is", 1.2, 1.4, 1.0),
        Word("a", 1.4, 1.5, 1.0),
        Word("test", 1.5, 2.0, 1.0),
    ]

    segment = CleanSegment(
        segment_id="test_001",
        text="Hello world this is a test",
        start=0.0,
        end=2.0,
        words=words,
        original_text="Hello world this is a test"
    )

    # Generate TTS
    print("\nGenerating TTS audio...")
    output_dir = Path("test_output")
    audio_path = tts_agent.generate_for_segment(segment, output_dir)

    print(f"✓ TTS audio generated: {audio_path}")
    print(f"  File size: {audio_path.stat().st_size / 1024:.2f} KB")

    # Validate quality
    print("\nValidating audio quality...")
    quality = tts_agent.validate_audio_quality(audio_path)
    print(f"✓ Quality validation: {quality}")

    return True

def test_sample_rate():
    """Test that sample rate property works"""
    print("\n" + "=" * 60)
    print("Test 2: Sample Rate Property")
    print("=" * 60)

    config = TTSConfig(backend="chatterbox", device="cpu")
    tts_agent = TTSGenerationAgent(config)

    sample_rate = tts_agent.backend.get_sample_rate()
    print(f"✓ Sample rate: {sample_rate} Hz")

    return True

def main():
    """Run all tests"""
    print("\nTesting Chatterbox TTS Integration\n")

    try:
        # Test 1: Basic generation
        if not test_tts_basic():
            print("\n✗ Test 1 FAILED")
            return 1

        # Test 2: Sample rate
        if not test_sample_rate():
            print("\n✗ Test 2 FAILED")
            return 1

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n✗ TEST FAILED WITH ERROR:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
