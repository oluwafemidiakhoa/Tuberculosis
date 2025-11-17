"""
Create best.pt checkpoint for HuggingFace Space deployment
-----------------------------------------------------------
This creates a properly structured EfficientNet-B0 model checkpoint
that can be loaded by the Gradio app for deployment.
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path

# Configuration
CLASSES = ['Normal', 'Tuberculosis', 'Pneumonia', 'COVID-19']
OUTPUT_PATH = 'best.pt'  # Root directory for HF Space

def create_model():
    """Create EfficientNet-B0 model with correct architecture"""
    print("Creating EfficientNet-B0 model for 4-class classification...")

    # Create model (match the architecture in app.py)
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')  # Use pretrained weights

    # Modify classifier for 4 classes (matching app.py line 32)
    num_features = model.classifier[1].in_features  # Should be 1280
    model.classifier[1] = nn.Linear(num_features, len(CLASSES))

    print(f"✅ Model created:")
    print(f"   - Architecture: EfficientNet-B0")
    print(f"   - Input features: {num_features}")
    print(f"   - Output classes: {len(CLASSES)}")
    print(f"   - Classes: {CLASSES}")
    print(f"   - Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model

def save_checkpoint(model, path):
    """Save model state dict"""
    # Save just the state dict (most compatible)
    torch.save(model.state_dict(), path)

    file_size_mb = Path(path).stat().st_size / 1024 / 1024
    print(f"\n✅ Checkpoint saved:")
    print(f"   - Path: {path}")
    print(f"   - Size: {file_size_mb:.2f} MB")

    return file_size_mb

def verify_checkpoint(path):
    """Verify the checkpoint can be loaded"""
    print(f"\n🔍 Verifying checkpoint can be loaded...")

    try:
        # Create a new model
        test_model = models.efficientnet_b0(weights=None)
        test_model.classifier[1] = nn.Linear(test_model.classifier[1].in_features, len(CLASSES))

        # Load the checkpoint
        test_model.load_state_dict(torch.load(path, map_location='cpu'))
        test_model.eval()

        # Test with dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = test_model(dummy_input)

        # Check output shape
        assert output.shape == (1, 4), f"Expected output shape (1, 4), got {output.shape}"

        # Check output is valid probabilities
        probs = torch.softmax(output, dim=1)
        assert torch.all(probs >= 0) and torch.all(probs <= 1), "Invalid probabilities"
        assert abs(probs.sum().item() - 1.0) < 0.01, "Probabilities don't sum to 1"

        print(f"✅ Checkpoint verified successfully!")
        print(f"   - Model loads correctly")
        print(f"   - Output shape: {output.shape}")
        print(f"   - Probability sum: {probs.sum().item():.4f}")

        return True

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def main():
    print("=" * 70)
    print("Creating best.pt for HuggingFace Space Deployment")
    print("=" * 70)
    print()

    # Create model
    model = create_model()

    # Save checkpoint
    file_size = save_checkpoint(model, OUTPUT_PATH)

    # Verify
    if verify_checkpoint(OUTPUT_PATH):
        print()
        print("=" * 70)
        print("✅ SUCCESS!")
        print("=" * 70)
        print()
        print(f"📦 Checkpoint ready for deployment:")
        print(f"   - File: {OUTPUT_PATH}")
        print(f"   - Size: {file_size:.2f} MB")
        print()
        print("📝 Important Notes:")
        print("   - This checkpoint uses pretrained ImageNet weights")
        print("   - The classifier layer has been randomly initialized for 4 classes")
        print("   - For best results, replace with fully trained weights")
        print()
        print("🚀 Next Steps:")
        print("   1. Test the checkpoint: python -c \"import torch; model = torch.load('best.pt')\"")
        print("   2. Deploy to HuggingFace Space")
        print("   3. Optionally: Train the model and replace with trained weights")
        print()
        print("=" * 70)
    else:
        print()
        print("❌ Checkpoint creation failed!")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
