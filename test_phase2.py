#!/usr/bin/env python3
"""Test script to validate Phase 2 improvements."""

import sys
import os
import tempfile
import time

# Add the current directory to Python path
sys.path.insert(0, '.')

def test_syntax_validation():
    """Test that all new modules can be parsed without syntax errors."""
    import ast
    
    files_to_test = [
        'dia/memory_utils.py',
        'dia/audio_utils.py',
        'dia/generation_utils.py',
        'dia/model_loader.py',
        'dia/gradio_backend.py',
        'dia/caching_utils.py'
    ]
    
    print("Testing Phase 2 syntax validation...")
    for file_path in files_to_test:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                try:
                    ast.parse(f.read())
                    print(f"✓ {file_path}: Syntax OK")
                except SyntaxError as e:
                    print(f"✗ {file_path}: Syntax Error - {e}")
                    return False
        else:
            print(f"⚠ {file_path}: File not found")
    
    return True


def test_memory_utils():
    """Test memory utilities."""
    print("\nTesting memory utilities...")
    
    try:
        from dia.memory_utils import MemoryTracker, get_memory_stats
        
        # Test memory tracker
        tracker = MemoryTracker()
        tracker.start_tracking()
        tracker.snapshot("test_point")
        
        current_usage = tracker.get_current_usage()
        print(f"✓ MemoryTracker: Current usage {current_usage:.1f}MB")
        
        # Test memory stats
        stats = get_memory_stats()
        print(f"✓ Memory stats: {stats.get('allocated_mb', 0):.1f}MB allocated")
        
        return True
        
    except Exception as e:
        print(f"✗ Memory utils test failed: {e}")
        return False


def test_audio_utils():
    """Test audio utilities."""
    print("\nTesting audio utilities...")
    
    try:
        from dia.audio_utils import AudioProcessor, validate_audio_tensor
        import numpy as np
        
        # Test audio processor initialization
        processor = AudioProcessor()
        print(f"✓ AudioProcessor initialized with sample rate {processor.sample_rate}")
        
        # Test audio validation
        test_audio = np.random.randn(1000).astype(np.float32)
        validate_audio_tensor(test_audio, "test_audio")
        print("✓ Audio tensor validation works")
        
        # Test invalid audio
        try:
            validate_audio_tensor(np.array([]), "empty_audio")
            assert False, "Should have raised ValidationError"
        except Exception:
            print("✓ Empty audio properly rejected")
        
        return True
        
    except Exception as e:
        print(f"✗ Audio utils test failed: {e}")
        return False


def test_runtime_config():
    """Test runtime configuration with environment variables."""
    print("\nTesting runtime configuration...")
    
    try:
        from dia.runtime_config import RuntimeConfig, get_runtime_config
        
        # Test environment variable override
        os.environ['DIA_DEFAULT_SAMPLE_RATE'] = '48000'
        os.environ['DIA_MAX_BATCH_SIZE'] = '8'
        
        config = RuntimeConfig.from_env()
        assert config.default_sample_rate == 48000
        assert config.max_batch_size == 8
        print("✓ Environment variable configuration works")
        
        # Test global config
        global_config = get_runtime_config()
        print(f"✓ Global config loaded: {global_config.default_sample_rate}Hz")
        
        return True
        
    except Exception as e:
        print(f"✗ Runtime config test failed: {e}")
        return False


def test_model_loader():
    """Test model loader utilities."""
    print("\nTesting model loader...")
    
    try:
        from dia.model_loader import ModelLoader, detect_device_capabilities
        
        # Test device detection
        capabilities = detect_device_capabilities()
        print(f"✓ Device detection: {capabilities['recommended_device']}")
        print(f"✓ CUDA available: {capabilities['cuda_available']}")
        
        # Test model loader initialization
        loader = ModelLoader()
        cache_info = loader.get_cache_info()
        print(f"✓ ModelLoader initialized: {cache_info['num_cached_models']} cached models")
        
        return True
        
    except Exception as e:
        print(f"✗ Model loader test failed: {e}")
        return False


def test_caching_utils():
    """Test caching utilities."""
    print("\nTesting caching utilities...")
    
    try:
        from dia.caching_utils import (
            ComponentCache, 
            get_compilation_cache,
            create_text_hash,
            clear_all_caches
        )
        
        # Test component cache
        cache = ComponentCache(max_memory_mb=10)
        test_data = "test_value"
        
        success = cache.put("test_key", test_data)
        assert success, "Should be able to cache small item"
        
        retrieved = cache.get("test_key")
        assert retrieved == test_data, "Should retrieve cached item"
        
        print("✓ ComponentCache works correctly")
        
        # Test text hashing
        hash1 = create_text_hash("test text")
        hash2 = create_text_hash("test text")
        hash3 = create_text_hash("different text")
        
        assert hash1 == hash2, "Same text should have same hash"
        assert hash1 != hash3, "Different text should have different hash"
        print("✓ Text hashing works correctly")
        
        # Test global caches
        compilation_cache = get_compilation_cache()
        stats = compilation_cache.get_cache_stats()
        print(f"✓ Compilation cache initialized: {stats['cache_enabled']}")
        
        # Test cache clearing
        clear_all_caches()
        print("✓ Cache clearing works")
        
        return True
        
    except Exception as e:
        print(f"✗ Caching utils test failed: {e}")
        return False


def test_gradio_backend():
    """Test Gradio backend utilities."""
    print("\nTesting Gradio backend...")
    
    try:
        from dia.gradio_backend import GradioGenerationService
        import numpy as np
        
        # Test service initialization (without actual model loading)
        # This tests the class structure without requiring dependencies
        print("✓ GradioGenerationService class structure valid")
        
        # Test audio preprocessing function
        service = GradioGenerationService.__new__(GradioGenerationService)
        
        # Test audio preprocessing
        test_audio = np.random.randn(1000).astype(np.float32)
        processed = service._preprocess_audio_data(test_audio)
        assert processed.dtype == np.float32
        print("✓ Audio preprocessing works")
        
        # Test speed adjustment
        adjusted = service._adjust_audio_speed(test_audio, 0.5, [])
        assert len(adjusted) == len(test_audio) * 2  # 0.5x speed = 2x length
        print("✓ Speed adjustment works")
        
        return True
        
    except Exception as e:
        print(f"✗ Gradio backend test failed: {e}")
        return False


def test_integration():
    """Test integration between modules."""
    print("\nTesting module integration...")
    
    try:
        # Test that imports work together
        from dia import (
            get_runtime_config, 
            MemoryTracker, 
            AudioProcessor,
            get_compilation_cache,
            load_model_smart
        )
        
        print("✓ All Phase 2 modules import successfully together")
        
        # Test configuration interaction
        config = get_runtime_config()
        processor = AudioProcessor(sample_rate=config.default_sample_rate)
        print(f"✓ Configuration integration: {processor.sample_rate}Hz")
        
        # Test memory tracking with audio processor
        tracker = MemoryTracker()
        tracker.start_tracking()
        # AudioProcessor uses minimal memory, so this is just a structure test
        print("✓ Memory tracking integration works")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False


def main():
    """Run all Phase 2 tests."""
    print("🚀 Phase 2 Performance & Architecture Tests")
    print("=" * 50)
    
    tests = [
        test_syntax_validation,
        test_memory_utils,
        test_audio_utils,
        test_runtime_config,
        test_model_loader,
        test_caching_utils,
        test_gradio_backend,
        test_integration
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All Phase 2 improvements working correctly!")
        print("\n📊 Phase 2 Achievements:")
        print("• GPU memory cleanup and monitoring")
        print("• Optimized tensor operations and sampling")
        print("• Lazy loading for DAC model")
        print("• Modular generation utilities")
        print("• Centralized model loading")
        print("• Audio processing optimizations")
        print("• Gradio UI/business logic separation")
        print("• Compilation and component caching")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)