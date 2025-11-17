"""
Test Checkpoint Compatibility
==============================

This script verifies that checkpoints saved by the training scripts
are compatible with the inference app.

Usage:
    python test_checkpoint_compatibility.py --checkpoint checkpoints_multiclass_best/best.pt
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
import argparse


def test_checkpoint_compatibility(checkpoint_path):
    """
    Test if a checkpoint can be loaded into the inference app's model.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        bool: True if compatible, False otherwise
    """
    print("="*70)
    print("CHECKPOINT COMPATIBILITY TEST")
    print("="*70)
    print(f"\n📁 Checkpoint: {checkpoint_path}\n")

    # Check if file exists
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint file not found: {checkpoint_path}")
        return False

    # Load the checkpoint
    print("1️⃣  Loading checkpoint...")
    try:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        print("   ✅ Checkpoint loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load checkpoint: {e}")
        return False

    # Handle different checkpoint formats
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        print("   ℹ️  Detected training checkpoint with metadata")
        print(f"      Epoch: {state_dict.get('epoch', 'N/A')}")
        print(f"      Val Acc: {state_dict.get('val_acc', 'N/A')}")
        state_dict = state_dict['model_state_dict']

    # Analyze keys
    print(f"\n2️⃣  Analyzing checkpoint structure...")
    print(f"   Total keys: {len(state_dict)}")

    # Check for issues
    issues = []

    # Issue 1: "model." prefix
    model_prefix_keys = [key for key in state_dict.keys() if key.startswith('model.')]
    if model_prefix_keys:
        issues.append("'model.' prefix found")
        print(f"   ❌ Found {len(model_prefix_keys)} keys with 'model.' prefix")
        print(f"      Examples: {model_prefix_keys[:3]}")

    # Issue 2: Extra keys
    expected_prefixes = ['features.', 'classifier.']
    extra_keys = [key for key in state_dict.keys()
                  if not any(key.startswith(prefix) for prefix in expected_prefixes)]
    if extra_keys:
        issues.append(f"extra keys: {extra_keys}")
        print(f"   ❌ Found {len(extra_keys)} extra keys: {extra_keys}")

    # Issue 3: Missing expected keys
    expected_keys = ['features.0.0.weight', 'classifier.1.weight', 'classifier.1.bias']
    missing_keys = [key for key in expected_keys if key not in state_dict]
    if missing_keys:
        issues.append("missing expected keys")
        print(f"   ❌ Missing expected keys: {missing_keys}")

    if not issues:
        print("   ✅ Checkpoint structure looks good!")
        print(f"      Sample keys:")
        for key in list(state_dict.keys())[:5]:
            print(f"        - {key}")

    # Test loading into inference model
    print(f"\n3️⃣  Testing inference app compatibility...")
    print("   Creating EfficientNet-B0 with 4 classes...")

    try:
        # This is exactly what the inference app does
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
        print("   ✅ Model created")

        # Try to load the state_dict
        print("   Loading state_dict...")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"   ❌ Missing keys in state_dict:")
            for key in missing_keys[:10]:
                print(f"      - {key}")
            if len(missing_keys) > 10:
                print(f"      ... and {len(missing_keys) - 10} more")

        if unexpected_keys:
            print(f"   ❌ Unexpected keys in state_dict:")
            for key in unexpected_keys[:10]:
                print(f"      - {key}")
            if len(unexpected_keys) > 10:
                print(f"      ... and {len(unexpected_keys) - 10} more")

        if not missing_keys and not unexpected_keys:
            print("   ✅ State dict loaded successfully with strict=True!")
            # Test with strict=True to be sure
            model.load_state_dict(state_dict, strict=True)
            print("   ✅ FULLY COMPATIBLE!")

            # Quick forward pass test
            print("\n4️⃣  Testing forward pass...")
            model.eval()
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output = model(dummy_input)
            print(f"   ✅ Forward pass successful!")
            print(f"      Output shape: {output.shape}")
            print(f"      Expected: torch.Size([1, 4])")

            if output.shape == torch.Size([1, 4]):
                print("\n" + "="*70)
                print("✅ CHECKPOINT IS FULLY COMPATIBLE!")
                print("="*70)
                print("\nYou can safely use this checkpoint in your inference app:")
                print(f"  model.load_state_dict(torch.load('{checkpoint_path}'))")
                return True
            else:
                print(f"\n❌ Output shape mismatch!")
                return False

        else:
            print("\n" + "="*70)
            print("❌ CHECKPOINT IS NOT COMPATIBLE!")
            print("="*70)
            print(f"\nIssues found: {', '.join(issues)}")
            print("\nTo fix this:")
            print("  1. Use the convert_checkpoint.py script:")
            print(f"     python convert_checkpoint.py --input {checkpoint_path} --output best_clean.pt")
            print("\n  2. Or retrain with the fixed training scripts")
            return False

    except Exception as e:
        print(f"\n❌ Error during compatibility test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test checkpoint compatibility with inference app"
    )
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=True,
        help='Path to the checkpoint file to test'
    )

    args = parser.parse_args()

    success = test_checkpoint_compatibility(args.checkpoint)

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
