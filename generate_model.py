#!/usr/bin/env python3
"""
Generate a model file for the Gradio app
This creates a properly initialized model that matches the app's architecture
"""
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path

print("Generating model for Gradio app...")

# Create model matching app.py architecture
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)  # 4 classes

# Initialize with ImageNet pretrained weights for better baseline
print("Loading pretrained ImageNet weights...")
pretrained_model = models.efficientnet_b0(weights='IMAGENET1K_V1')

# Copy pretrained weights for feature extractor
pretrained_dict = pretrained_model.state_dict()
model_dict = model.state_dict()

# Transfer pretrained weights except final classifier
pretrained_dict = {k: v for k, v in pretrained_dict.items()
                   if k in model_dict and 'classifier.1' not in k}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Create checkpoints directory
checkpoint_dir = Path('checkpoints')
checkpoint_dir.mkdir(exist_ok=True)

# Save model
model_path = checkpoint_dir / 'best_multiclass.pt'
torch.save(model.state_dict(), model_path)

print(f"✅ Model saved to: {model_path}")
print(f"   Size: {model_path.stat().st_size / (1024*1024):.2f} MB")
print("\n⚠️  NOTE: This model uses ImageNet pretrained weights.")
print("   For medical accuracy, you need to train on chest X-ray data.")
print("   Run: python train_multiclass_simple.py")
