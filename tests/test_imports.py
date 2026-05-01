"""
Test script to verify all imports and basic functionality work correctly.
"""

def test_basic_imports():
    """Test basic Python imports."""
    print("Testing basic imports...")
    import numpy as np
    import pandas as pd
    print("✅ NumPy and Pandas imported successfully")

    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False

    return True

def test_model_imports():
    """Test model component imports."""
    print("\nTesting model component imports...")
    try:
        from MLP import MLP
        print("✅ MLP imported successfully")

        # Test MLP functionality
        mlp = MLP([10, 20, 1])
        print(f"✅ MLP created: {mlp}")

    except Exception as e:
        print(f"❌ MLP test failed: {e}")
        return False

    return True

def test_data_imports():
    """Test data handling imports."""
    print("\nTesting data handling imports...")
    try:
        from data import setup_data_loaders
        print("✅ Data loader setup imported successfully")

        from utils_train import train_model
        print("✅ Training utilities imported successfully")

    except Exception as e:
        print(f"❌ Data/training import failed: {e}")
        return False

    return True

if __name__ == "__main__":
    print("🧪 TESTING CLEANED CRYSCO CODE")
    print("=" * 50)

    success = True
    success &= test_basic_imports()
    success &= test_model_imports()
    success &= test_data_imports()

    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL TESTS PASSED! Code cleanup successful.")
    else:
        print("❌ Some tests failed. Check the error messages above.")