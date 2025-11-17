"""
Generate demo model weights for deployment
-------------------------------------------
This creates a trained EfficientNet-B0 model structure that can be loaded by the Gradio app.

Since we don't have the actual trained weights from the 87.29% model,
this script creates a properly structured checkpoint that:
1. Has the correct architecture (EfficientNet-B0, 4 classes)
2. Can be loaded by the Gradio app
3. Will show random predictions (for demo structure testing)

For actual deployment, replace this with real trained weights.
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path

# Configuration
CLASSES = ['Normal', 'TB', 'Pneumonia', 'COVID']
OUTPUT_PATH = 'gradio_app/checkpoints/best_multiclass.pt'

def create_demo_model():
    """Create EfficientNet-B0 model with correct architecture"""
    print("Creating EfficientNet-B0 model...")

    # Create model
    model = models.efficientnet_b0(pretrained=True)  # Start with ImageNet weights

    # Modify classifier for 4 classes
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(CLASSES))

    print(f"✅ Model created:")
    print(f"   - Base: EfficientNet-B0")
    print(f"   - Input features: {num_features}")
    print(f"   - Output classes: {len(CLASSES)}")
    print(f"   - Classes: {CLASSES}")

    return model

def save_checkpoint(model, path):
    """Save model checkpoint"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Save just the state dict (most compatible format)
    torch.save(model.state_dict(), path)

    print(f"✅ Checkpoint saved to: {path}")
    print(f"   File size: {Path(path).stat().st_size / 1024 / 1024:.2f} MB")

def main():
    print("="*60)
    print("Generating Demo Model Weights")
    print("="*60)
    print()

    # Create model
    model = create_demo_model()
    print()

    # Save checkpoint
    save_checkpoint(model, OUTPUT_PATH)
    print()

    # Verify it can be loaded
    print("Verifying checkpoint can be loaded...")
    try:
        loaded_model = models.efficientnet_b0(pretrained=False)
        num_features = loaded_model.classifier[1].in_features
        loaded_model.classifier[1] = nn.Linear(num_features, len(CLASSES))
        loaded_model.load_state_dict(torch.load(OUTPUT_PATH))
        print("✅ Checkpoint verified - can be loaded successfully")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return

    print()
    print("="*60)
    print("⚠️  IMPORTANT NOTE")
    print("="*60)
    print()
    print("This is a DEMO checkpoint with ImageNet pretrained weights.")
    print("It has NOT been trained on chest X-rays yet.")
    print()
    print("Predictions will be based on ImageNet features, not medical training.")
    print()
    print("To get the actual 87.29% accuracy model:")
    print("  1. Run the training script (train_multiclass_improved.py)")
    print("  2. Or upload the trained weights from your training run")
    print("  3. Replace this file with the actual trained model")
    print()
    print("For now, this allows testing the Gradio app structure.")
    print("="*60)

if __name__ == "__main__":
    main()
