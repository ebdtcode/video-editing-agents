#!/usr/bin/env python3
"""
Quick test script to verify custom voice cloning is working
"""

import warnings
from pathlib import Path

# Suppress PyTorch deprecation warning from Chatterbox library
warnings.filterwarnings('ignore', message='.*torch.backends.cuda.sdp_kernel.*', category=FutureWarning)

from src.config import ProcessingConfig
from src.agents.tts_generation_agent import TTSGenerationAgent
from src.agents.content_analysis_agent import CleanSegment

def test_custom_voice():
    print("=" * 60)
    print("Testing Custom Voice Cloning")
    print("=" * 60)

    # Load config
    print("\n1. Loading config...")
    config = ProcessingConfig.from_yaml('config_cpu.yaml')
    print(f"   voice_mode: {config.tts.voice_mode}")
    print(f"   voice_reference: {config.tts.voice_reference}")

    # Verify voice file exists
    if config.tts.voice_mode == "custom":
        voice_file = Path(config.tts.voice_reference)
        if not voice_file.exists():
            print(f"\n❌ Voice file not found: {voice_file}")
            return False
        print(f"   ✓ Voice file found: {voice_file}")
        print(f"   Size: {voice_file.stat().st_size / 1024 / 1024:.2f} MB")

    # Initialize TTS agent
    print("\n2. Initializing TTS agent...")
    tts_agent = TTSGenerationAgent(config.tts)
    print("   ✓ TTS agent initialized")

    # Create a test segment
    test_text = "This is a test of the voice cloning feature. The voice should match the reference audio."
    test_segment = CleanSegment(
        segment_id="test_0001",
        text=test_text,
        start=0.0,
        end=5.0,
        words=[],
        original_text=test_text
    )

    # Generate test audio
    print("\n3. Generating test TTS audio...")
    output_dir = Path("temp_segments/test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get reference audio
    reference_audio = None
    if config.tts.voice_mode == "custom":
        reference_audio = Path(config.tts.voice_reference)

    try:
        audio_path = tts_agent.generate_for_segment(
            test_segment,
            output_dir,
            reference_audio
        )
        print(f"   ✓ Generated: {audio_path}")
        print(f"\n4. Test audio created successfully!")
        print(f"\n   Listen to the test audio:")
        print(f"   ffplay {audio_path}")
        print(f"\n   Compare with reference voice:")
        if reference_audio:
            print(f"   ffplay {reference_audio}")
        print("\n" + "=" * 60)
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_custom_voice()
    exit(0 if success else 1)
