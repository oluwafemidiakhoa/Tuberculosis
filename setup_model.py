"""
🔧 Setup Model Weights for Gradio App
======================================
Run this script to generate the model checkpoint file needed by the Gradio app.

This creates a properly structured EfficientNet-B0 checkpoint that can be loaded
by gradio_app/app.py.

⚠️ IMPORTANT: This generates UNTRAINED weights!
To get accurate predictions, you must:
1. Download the COVID-QU-Ex dataset from Kaggle
2. Run one of the training scripts (train_multiclass_simple.py or train_optimized_90_95.py)
3. Replace the checkpoint with your trained model

Usage:
    python setup_model.py
"""

import sys
try:
    import torch
    import torch.nn as nn
    from torchvision import models
    from pathlib import Path
except ImportError as e:
    print("❌ Error: Required dependencies not installed")
    print(f"   Missing: {e}")
    print()
    print("Please install requirements:")
    print("   pip install torch torchvision")
    print()
    print("Or install all requirements:")
    print("   pip install -r requirements.txt")
    sys.exit(1)

# Configuration
CLASSES = ['Normal', 'Tuberculosis', 'Pneumonia', 'COVID-19']
# Update path to be relative to where app.py looks for it
OUTPUT_PATH = 'checkpoints/best_multiclass.pt'

def create_model():
    """Create EfficientNet-B0 model with correct architecture"""
    print("📦 Creating EfficientNet-B0 model...")

    # Create model with ImageNet pretrained weights
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')

    # Modify classifier for 4 classes
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(CLASSES))

    print(f"   ✅ Model architecture created")
    print(f"      - Base: EfficientNet-B0")
    print(f"      - Input features: {num_features}")
    print(f"      - Output classes: {len(CLASSES)}")
    print(f"      - Classes: {CLASSES}")

    return model

def save_checkpoint(model, path):
    """Save model checkpoint"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Save state dict (compatible with app.py loading method)
    torch.save(model.state_dict(), path)

    file_size_mb = Path(path).stat().st_size / 1024 / 1024
    print(f"\n💾 Checkpoint saved:")
    print(f"   📁 Path: {path}")
    print(f"   📊 Size: {file_size_mb:.2f} MB")

def verify_checkpoint(path):
    """Verify the checkpoint can be loaded"""
    print(f"\n🔍 Verifying checkpoint...")
    try:
        test_model = models.efficientnet_b0(weights=None)
        test_model.classifier[1] = nn.Linear(test_model.classifier[1].in_features, len(CLASSES))
        test_model.load_state_dict(torch.load(path))
        print(f"   ✅ Checkpoint verified successfully")
        return True
    except Exception as e:
        print(f"   ❌ Error loading checkpoint: {e}")
        return False

def main():
    print("=" * 70)
    print("🫁 Multi-Class Chest X-Ray Detection - Model Setup")
    print("=" * 70)
    print()

    # Create model
    model = create_model()

    # Save checkpoint
    save_checkpoint(model, OUTPUT_PATH)

    # Verify
    if not verify_checkpoint(OUTPUT_PATH):
        print("\n❌ Setup failed - checkpoint could not be verified")
        sys.exit(1)

    # Success message
    print()
    print("=" * 70)
    print("✅ Setup Complete!")
    print("=" * 70)
    print()
    print("📍 Next Steps:")
    print()
    print("   1️⃣  Your Gradio app can now load the model")
    print("      Run: cd gradio_app && python app.py")
    print()
    print("   ⚠️  2️⃣  IMPORTANT: Train the model for accurate predictions!")
    print()
    print("      Current state: ImageNet pretrained weights")
    print("      Predictions: Will be random/inaccurate for medical images")
    print()
    print("      To train the model:")
    print("      a) Download COVID-QU-Ex dataset:")
    print("         https://www.kaggle.com/datasets/anasmohammedtahir/covidqu")
    print()
    print("      b) Run training script:")
    print("         python train_multiclass_simple.py      # Quick (~2-3 hours)")
    print("         python train_optimized_90_95.py        # Best (~8-12 hours)")
    print()
    print("      c) Trained model will be saved to:")
    print(f"         {OUTPUT_PATH}")
    print()
    print("   3️⃣  Deploy to Hugging Face Spaces:")
    print("      - Upload the trained checkpoint to your Space")
    print("      - Or use Git LFS for large model files")
    print()
    print("=" * 70)
    print()
    print("💡 Tip: The untrained model will show ~25% confidence across all")
    print("   classes (random guessing). Train it for accurate medical predictions!")
    print()

if __name__ == "__main__":
    main()
