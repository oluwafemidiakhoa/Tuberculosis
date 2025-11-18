"""
Quick Checkpoint Compatibility Check and Converter
==================================================

This script can be imported and run in a Jupyter notebook to check
and convert checkpoints if needed.

Usage in notebook:
    %run check_and_convert_checkpoint.py
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path


def quick_check_compatibility(checkpoint_path):
    """
    Quick check if checkpoint is compatible with inference app.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        bool: True if compatible, False otherwise
    """
    print(f"🔍 Checking compatibility: {checkpoint_path}")

    try:
        # Load checkpoint
        state_dict = torch.load(checkpoint_path, map_location='cpu')

        # Handle training checkpoint format
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

        # Create inference model (exactly as in app)
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)

        # Try to load
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if not missing_keys and not unexpected_keys:
            print("   ✅ Checkpoint is COMPATIBLE!")
            return True
        else:
            print("   ❌ Checkpoint has compatibility issues")
            if missing_keys:
                print(f"      Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"      Unexpected keys: {len(unexpected_keys)}")
                print(f"      Examples: {unexpected_keys[:3]}")
            return False

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def convert_if_needed(checkpoint_path, output_path=None):
    """
    Check compatibility and convert if needed.

    Args:
        checkpoint_path: Path to checkpoint to check
        output_path: Optional output path for converted checkpoint
    """
    print("="*70)
    print("CHECKPOINT COMPATIBILITY CHECK & CONVERSION")
    print("="*70)
    print()

    # Check compatibility
    is_compatible = quick_check_compatibility(checkpoint_path)

    if is_compatible:
        print("\n✅ No conversion needed! Checkpoint is ready to use.")
        return True

    # Convert if needed
    print("\n🔧 Converting checkpoint...")

    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Clean up keys (remove 'model.' prefix if present)
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                new_key = key.replace('model.', '', 1)
                cleaned_state_dict[new_key] = value
            else:
                cleaned_state_dict[key] = value

        # Save
        if output_path is None:
            output_path = checkpoint_path.replace('.pt', '_clean.pt')

        torch.save(cleaned_state_dict, output_path)
        print(f"   ✅ Converted checkpoint saved to: {output_path}")

        # Verify
        is_compatible = quick_check_compatibility(output_path)

        if is_compatible:
            print("\n✅ SUCCESS! Converted checkpoint is compatible!")
            return True
        else:
            print("\n❌ Conversion failed to fix compatibility issues")
            return False

    except Exception as e:
        print(f"   ❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    # Default: check the optimized checkpoint
    checkpoint_path = 'checkpoints_multiclass_optimized/best.pt'

    if Path(checkpoint_path).exists():
        convert_if_needed(checkpoint_path)
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please update the checkpoint_path variable.")
