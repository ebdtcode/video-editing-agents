"""
Test script to validate voice_mode: original implementation
"""
import sys
from pathlib import Path
from src.config import ProcessingConfig

def test_config_validation():
    """Test that config accepts 'original' as valid voice_mode"""
    print("Testing config validation...")
    
    # Create a minimal config with voice_mode: original
    config_dict = {
        'processing': {
            'min_segment_duration': 0.5,
            'max_segment_duration': 30.0,
            'max_workers': 1
        },
        'transcription': {
            'backend': 'whisperx',
            'model': 'base',
            'language': 'auto',
            'device': 'cpu',
            'compute_type': 'int8'
        },
        'tts': {
            'backend': 'chatterbox',
            'voice_mode': 'original',  # TEST: This should be valid
            'sample_rate': 16000,
            'normalize': True,
            'device': 'cpu'
        },
        'video': {
            'sync_mode': 'retime',
            'quality': 'high',
            'codec': 'h264',
            'bitrate': '5M',
            'gpu_acceleration': False,
            'hwaccel': 'none'
        },
        'output': {
            'format': 'mp4',
            'temp_dir': './temp_segments',
            'keep_intermediates': True,
            'checkpoint': True
        },
        'logging': {
            'level': 'INFO',
            'file': 'test.log'
        }
    }
    
    try:
        # This should NOT raise an error if voice_mode: original is valid
        from src.config import load_config_from_dict
        config = load_config_from_dict(config_dict)
        
        # Verify the value was loaded correctly
        assert config.tts.voice_mode == "original", f"Expected 'original', got '{config.tts.voice_mode}'"
        
        print("✅ Config validation passed: voice_mode 'original' is accepted")
        return True
        
    except ValueError as e:
        print(f"❌ Config validation failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_orchestrator_check():
    """Test that orchestrator has the new _extract_original_audio_segments method"""
    print("\nTesting orchestrator implementation...")
    
    try:
        from src.agents.orchestrator_agent import OrchestratorAgent
        
        # Check if the method exists
        if hasattr(OrchestratorAgent, '_extract_original_audio_segments'):
            print("✅ Method _extract_original_audio_segments exists in OrchestratorAgent")
            
            # Check method signature
            import inspect
            sig = inspect.signature(OrchestratorAgent._extract_original_audio_segments)
            params = list(sig.parameters.keys())
            
            expected_params = ['self', 'segments', 'work_dir', 'video_path']
            if params == expected_params:
                print(f"✅ Method signature correct: {params}")
                return True
            else:
                print(f"❌ Method signature incorrect. Expected {expected_params}, got {params}")
                return False
        else:
            print("❌ Method _extract_original_audio_segments not found in OrchestratorAgent")
            return False
            
    except Exception as e:
        print(f"❌ Error checking orchestrator: {e}")
        return False

def test_voice_mode_options():
    """Test all valid voice_mode options"""
    print("\nTesting all voice_mode options...")
    
    valid_modes = ['default', 'auto', 'custom', 'original']
    
    from src.config import load_config_from_dict
    
    base_config = {
        'processing': {'min_segment_duration': 0.5, 'max_segment_duration': 30.0, 'max_workers': 1},
        'transcription': {'backend': 'whisperx', 'model': 'base', 'language': 'auto', 'device': 'cpu', 'compute_type': 'int8'},
        'tts': {'backend': 'chatterbox', 'voice_mode': 'default', 'sample_rate': 16000, 'normalize': True, 'device': 'cpu'},
        'video': {'sync_mode': 'retime', 'quality': 'high', 'codec': 'h264', 'bitrate': '5M', 'gpu_acceleration': False, 'hwaccel': 'none'},
        'output': {'format': 'mp4', 'temp_dir': './temp_segments', 'keep_intermediates': True, 'checkpoint': True},
        'logging': {'level': 'INFO', 'file': 'test.log'}
    }
    
    results = []
    for mode in valid_modes:
        try:
            base_config['tts']['voice_mode'] = mode
            config = load_config_from_dict(base_config)
            assert config.tts.voice_mode == mode
            print(f"  ✅ voice_mode: {mode} - Valid")
            results.append(True)
        except Exception as e:
            print(f"  ❌ voice_mode: {mode} - Failed: {e}")
            results.append(False)
    
    return all(results)

def main():
    print("=" * 60)
    print("Testing voice_mode: original Implementation")
    print("=" * 60)
    
    results = []
    
    # Test 1: Config validation
    results.append(test_config_validation())
    
    # Test 2: Orchestrator method
    results.append(test_orchestrator_check())
    
    # Test 3: All voice modes
    results.append(test_voice_mode_options())
    
    print("\n" + "=" * 60)
    if all(results):
        print("✅ All tests passed!")
        print("=" * 60)
        return 0
    else:
        print("❌ Some tests failed")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
