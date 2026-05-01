#!/usr/bin/env python
"""
Test script to verify the new repository structure works correctly.
"""

def test_imports():
    """Test all critical imports work."""
    print("🧪 Testing reorganized structure imports...")

    try:
        from crysco.models.MLP import MLP
        print("✅ MLP import successful")

        # Test MLP functionality
        mlp = MLP([10, 20, 1])
        print(f"✅ MLP creation successful: {mlp}")

    except Exception as e:
        print(f"❌ MLP error: {e}")
        return False

    try:
        from crysco.models.CrysCo import CrysCo, ResidualNN
        print("✅ CrysCo model imports successful")

    except Exception as e:
        print(f"❌ CrysCo import error: {e}")
        return False

    try:
        from crysco.utils.utils_train import train_model
        print("✅ Training utilities import successful")

    except Exception as e:
        print(f"❌ Training utils error: {e}")
        return False

    try:
        from crysco.data.data import setup_data_loaders
        print("✅ Data utilities import successful")

    except Exception as e:
        print(f"❌ Data utils error: {e}")
        return False

    return True

def test_structure():
    """Test directory structure."""
    import os
    print("\n📁 Testing directory structure...")

    expected_dirs = [
        'crysco',
        'crysco/models',
        'crysco/data',
        'crysco/utils',
        'scripts',
        'scripts/preprocessing',
        'tests',
        'docs'
    ]

    for directory in expected_dirs:
        if os.path.exists(directory):
            print(f"✅ {directory} exists")
        else:
            print(f"❌ {directory} missing")
            return False

    return True

if __name__ == "__main__":
    print("🚀 TESTING REORGANIZED CRYSCO REPOSITORY")
    print("=" * 60)

    imports_ok = test_imports()
    structure_ok = test_structure()

    print("\n" + "=" * 60)
    if imports_ok and structure_ok:
        print("🎉 REPOSITORY STRUCTURE TEST PASSED!")
        print("✅ All imports working correctly")
        print("✅ Professional directory structure verified")
        print("🚀 Ready for production use")
    else:
        print("❌ Some issues detected in repository structure")