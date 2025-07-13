#!/usr/bin/env python3
"""Simple test script to validate Phase 1 improvements."""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, '.')

def test_syntax_validation():
    """Test that all modules can be parsed without syntax errors."""
    import ast
    
    files_to_test = [
        'dia/exceptions.py',
        'dia/validation.py', 
        'dia/runtime_config.py',
        'dia/model.py',
        'dia/config.py',
        'cli.py'
    ]
    
    print("Testing syntax validation...")
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

def test_runtime_config():
    """Test runtime configuration loading."""
    print("\nTesting runtime configuration...")
    
    try:
        from dia.runtime_config import RuntimeConfig, get_runtime_config
        
        # Test default config
        config = RuntimeConfig()
        print(f"✓ Default sample rate: {config.default_sample_rate}")
        print(f"✓ Default max batch size: {config.max_batch_size}")
        
        # Test environment loading
        os.environ['DIA_DEFAULT_SAMPLE_RATE'] = '48000'
        os.environ['DIA_MAX_BATCH_SIZE'] = '16'
        
        env_config = RuntimeConfig.from_env()
        assert env_config.default_sample_rate == 48000
        assert env_config.max_batch_size == 16
        print("✓ Environment variable loading works")
        
        # Test global config
        global_config = get_runtime_config()
        print(f"✓ Global config loaded: {global_config.default_sample_rate}Hz")
        
        return True
        
    except Exception as e:
        print(f"✗ Runtime config test failed: {e}")
        return False

def test_exceptions():
    """Test custom exception classes."""
    print("\nTesting exception classes...")
    
    try:
        from dia.exceptions import (
            DiaError, ModelLoadError, AudioProcessingError, 
            GenerationError, ValidationError
        )
        
        # Test exception creation and attributes
        model_error = ModelLoadError("Test error", "/test/path")
        assert model_error.model_path == "/test/path"
        assert model_error.error_code == "MODEL_LOAD_ERROR"
        print("✓ ModelLoadError works correctly")
        
        validation_error = ValidationError("Invalid param", "test_param")
        assert validation_error.parameter_name == "test_param"
        print("✓ ValidationError works correctly")
        
        # Test inheritance
        assert isinstance(model_error, DiaError)
        assert isinstance(validation_error, DiaError)
        print("✓ Exception inheritance works")
        
        return True
        
    except Exception as e:
        print(f"✗ Exception test failed: {e}")
        return False

def test_validation_functions():
    """Test validation functions."""
    print("\nTesting validation functions...")
    
    try:
        from dia.validation import validate_text_input, validate_generation_parameters
        from dia.exceptions import ValidationError
        
        # Test valid text
        validate_text_input("Hello world")
        validate_text_input(["Hello", "world"])
        print("✓ Valid text input passes")
        
        # Test invalid text
        try:
            validate_text_input("")
            assert False, "Should have raised ValidationError"
        except ValidationError:
            print("✓ Empty text properly rejected")
        
        # Test generation parameters
        validate_generation_parameters(1000, 2.0, 1.5, 0.9, 50)
        print("✓ Valid generation parameters pass")
        
        # Test invalid parameters
        try:
            validate_generation_parameters(-1)
            assert False, "Should have raised ValidationError"
        except ValidationError:
            print("✓ Invalid max_tokens properly rejected")
        
        return True
        
    except Exception as e:
        print(f"✗ Validation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Phase 1 Improvement Tests")
    print("=" * 40)
    
    tests = [
        test_syntax_validation,
        test_runtime_config, 
        test_exceptions,
        test_validation_functions
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All Phase 1 improvements working correctly!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)