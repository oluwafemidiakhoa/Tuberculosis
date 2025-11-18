"""
Checkpoint Converter for EfficientNet Models
=============================================

This script converts training checkpoints that have:
1. "model." prefix on all keys (from wrapper class)
2. Extra keys like "activation_mask"

To clean EfficientNet checkpoints that can be loaded directly into:
    model = efficientnet_b2(num_classes=4)
    model.load_state_dict(torch.load('best.pt'))

Usage:
    python convert_checkpoint.py --input checkpoints/best_old.pt --output best.pt

Or to convert in-place:
    python convert_checkpoint.py --input checkpoints/best.pt --inplace
"""

import torch
import argparse
from pathlib import Path
from collections import OrderedDict


def convert_checkpoint(input_path, output_path=None, inplace=False):
    """
    Convert a wrapped checkpoint to a clean EfficientNet checkpoint.

    Args:
        input_path: Path to the wrapped checkpoint
        output_path: Path to save the clean checkpoint (optional if inplace=True)
        inplace: If True, overwrite the input file

    Returns:
        dict: The cleaned state_dict
    """
    print(f"Loading checkpoint from: {input_path}")

    # Load the checkpoint
    try:
        checkpoint = torch.load(input_path, map_location='cpu')
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        # Check if it's a training checkpoint with 'model_state_dict' key
        if 'model_state_dict' in checkpoint:
            print("  ℹ️  Detected training checkpoint with metadata")
            state_dict = checkpoint['model_state_dict']
            print(f"     Epoch: {checkpoint.get('epoch', 'N/A')}")
            print(f"     Val Acc: {checkpoint.get('val_acc', 'N/A')}")
        else:
            # Assume it's a state_dict
            state_dict = checkpoint
    else:
        print("❌ Unexpected checkpoint format")
        return None

    # Analyze the state_dict
    print(f"\n📊 Checkpoint Analysis:")
    print(f"  Total keys: {len(state_dict)}")

    # Check for "model." prefix
    has_prefix = any(key.startswith('model.') for key in state_dict.keys())
    print(f"  Has 'model.' prefix: {has_prefix}")

    # Check for extra keys
    extra_keys = [key for key in state_dict.keys()
                  if not key.startswith('model.') and not key.startswith('features.')
                  and not key.startswith('classifier.')]
    if extra_keys:
        print(f"  Extra keys found: {extra_keys}")

    # Convert the state_dict
    print("\n🔧 Converting checkpoint...")
    cleaned_state_dict = OrderedDict()

    for key, value in state_dict.items():
        # Remove "model." prefix if present
        if key.startswith('model.'):
            new_key = key[6:]  # Remove "model." (6 characters)
            cleaned_state_dict[new_key] = value
            print(f"  ✓ {key} → {new_key}")
        # Skip extra keys (like activation_mask)
        elif key in extra_keys:
            print(f"  ✗ Skipping: {key}")
        # Keep as-is if already clean
        else:
            cleaned_state_dict[key] = value
            print(f"  ✓ Kept: {key}")

    print(f"\n✅ Converted checkpoint:")
    print(f"  Original keys: {len(state_dict)}")
    print(f"  Cleaned keys: {len(cleaned_state_dict)}")
    print(f"  Removed: {len(state_dict) - len(cleaned_state_dict)} keys")

    # Verify the checkpoint has the expected structure
    expected_keys = ['features.0.0.weight', 'classifier.1.weight', 'classifier.1.bias']
    missing = [key for key in expected_keys if key not in cleaned_state_dict]

    if missing:
        print(f"\n⚠️  WARNING: Expected keys not found: {missing}")
        print("   This checkpoint may not be compatible with EfficientNet")
    else:
        print(f"\n✅ Checkpoint structure verified!")
        print("   Compatible with: efficientnet_b2(num_classes=4)")

    # Save the cleaned checkpoint
    if inplace:
        output_path = input_path
        print(f"\n💾 Saving cleaned checkpoint (in-place): {output_path}")
    elif output_path:
        print(f"\n💾 Saving cleaned checkpoint: {output_path}")
    else:
        print("\n⚠️  No output path specified. Checkpoint not saved.")
        return cleaned_state_dict

    # Create backup if inplace
    if inplace:
        backup_path = Path(input_path).with_suffix('.pt.backup')
        print(f"  Creating backup: {backup_path}")
        torch.save(checkpoint, backup_path)

    # Save the cleaned checkpoint
    torch.save(cleaned_state_dict, output_path)
    print(f"  ✅ Saved!")

    return cleaned_state_dict


def main():
    parser = argparse.ArgumentParser(
        description="Convert wrapped EfficientNet checkpoint to clean format"
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to the input checkpoint file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Path to save the cleaned checkpoint (default: best_clean.pt)'
    )
    parser.add_argument(
        '--inplace',
        action='store_true',
        help='Overwrite the input file (creates a .backup file)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify the cleaned checkpoint can be loaded into EfficientNet'
    )

    args = parser.parse_args()

    # Set output path
    if args.inplace:
        output_path = args.input
    elif args.output:
        output_path = args.output
    else:
        # Default output path
        input_path = Path(args.input)
        output_path = input_path.parent / f"{input_path.stem}_clean{input_path.suffix}"

    # Convert the checkpoint
    cleaned_state_dict = convert_checkpoint(args.input, output_path, args.inplace)

    if cleaned_state_dict is None:
        print("\n❌ Conversion failed!")
        return 1

    # Verify if requested
    if args.verify:
        print("\n🔍 Verifying checkpoint...")
        try:
            from torchvision import models
            import torch.nn as nn

            # Create a dummy model
            model = models.efficientnet_b2(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)

            # Try to load the checkpoint
            model.load_state_dict(cleaned_state_dict, strict=True)

            print("  ✅ Checkpoint verified! Can be loaded into EfficientNet-B2 with 4 classes")

        except Exception as e:
            print(f"  ❌ Verification failed: {e}")
            return 1

    print("\n" + "="*70)
    print("✅ CONVERSION COMPLETE!")
    print("="*70)
    print(f"\nYou can now use this checkpoint in your inference app:")
    print(f"  model.load_state_dict(torch.load('{output_path}'))")
    print()

    return 0


if __name__ == '__main__':
    exit(main())
