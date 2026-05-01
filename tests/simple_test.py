#!/usr/bin/env python
"""
Simple test to verify the core functionality works without full dependencies.
"""

import sys
import torch
import numpy as np

def test_basic_functionality():
    """Test basic PyTorch functionality."""
    print("🔧 Testing basic PyTorch operations...")

    # Test tensor creation
    x = torch.randn(10, 5)
    print(f"✅ Created tensor with shape: {x.shape}")

    # Test GPU availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    else:
        print("⚠️  CUDA not available, using CPU")
        device = 'cpu'

    # Test device usage
    x = x.to(device)
    print(f"✅ Tensor moved to {device}")

    return device

def test_mlp_standalone():
    """Test MLP component standalone."""
    print("\n🧠 Testing MLP component...")

    try:
        from MLP import MLP
        import torch.nn as nn

        # Create MLP
        mlp = MLP([10, 20, 5, 1], act=nn.ReLU())
        print(f"✅ MLP created: {mlp}")

        # Test forward pass
        x = torch.randn(32, 10)  # batch_size=32, input_dim=10
        output = mlp(x)
        print(f"✅ Forward pass successful: input {x.shape} -> output {output.shape}")

        return True
    except Exception as e:
        print(f"❌ MLP test failed: {e}")
        return False

def test_training_loop_components():
    """Test training loop components without full model."""
    print("\n🏋️ Testing training components...")

    try:
        # Create simple model
        from MLP import MLP
        import torch.nn as nn
        import torch.nn.functional as F

        model = MLP([10, 20, 1], act=nn.ReLU())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Simulate training step
        x = torch.randn(16, 10)
        y = torch.randn(16, 1)

        model.train()
        optimizer.zero_grad()
        predictions = model(x)
        loss = F.mse_loss(predictions, y)
        loss.backward()
        optimizer.step()

        print(f"✅ Training step successful: loss = {loss.item():.4f}")
        return True

    except Exception as e:
        print(f"❌ Training component test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 SIMPLE CRYSCO FUNCTIONALITY TEST")
    print("=" * 60)

    device = test_basic_functionality()
    mlp_ok = test_mlp_standalone()
    training_ok = test_training_loop_components()

    print("\n" + "=" * 60)
    if mlp_ok and training_ok:
        print("🎉 CORE FUNCTIONALITY WORKS!")
        print(f"📍 Device: {device}")
        print("✅ Ready for training once torch-geometric is available")
    else:
        print("❌ Some core functionality issues detected")
        sys.exit(1)