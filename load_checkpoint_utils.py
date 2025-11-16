"""
Utility functions for loading PyTorch checkpoints correctly.

This module provides helper functions to handle different checkpoint formats,
including those with 'model.' prefixes and additional metadata keys.
"""

import torch
from typing import Dict, Any


def load_checkpoint(checkpoint_path: str, device: torch.device = None) -> Dict[str, Any]:
    """
    Load a PyTorch checkpoint and extract the model state dict.

    Handles different checkpoint formats:
    - Direct state dict
    - State dict with 'model' key
    - State dict with 'model.' prefix in keys
    - State dict with extra metadata (e.g., 'activation_mask')

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the checkpoint on (default: CPU)

    Returns:
        Clean state dict ready to load into a model

    Example:
        >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        >>> model = models.efficientnet_b0(weights=None)
        >>> model.classifier[1] = nn.Linear(1280, 4)
        >>> state_dict = load_checkpoint('checkpoints_multiclass/best.pt', device)
        >>> model.load_state_dict(state_dict)
    """
    if device is None:
        device = torch.device('cpu')

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        # If checkpoint has 'model' key, extract it
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Remove 'model.' prefix from keys if present
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                new_key = key[6:]  # Remove 'model.' prefix
                new_state_dict[new_key] = value
            elif key in ['activation_mask', 'optimizer', 'epoch', 'scheduler']:
                # Skip non-model keys (common metadata keys)
                continue
            else:
                new_state_dict[key] = value

        return new_state_dict
    else:
        # If checkpoint is already a state dict
        return checkpoint


def load_model_from_checkpoint(model: torch.nn.Module, checkpoint_path: str,
                               device: torch.device = None, strict: bool = True):
    """
    Load a model from a checkpoint file.

    Args:
        model: The model to load weights into
        checkpoint_path: Path to the checkpoint file
        device: Device to load the checkpoint on (default: CPU)
        strict: Whether to strictly enforce that the keys in state_dict match

    Returns:
        The model with loaded weights

    Example:
        >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        >>> model = models.efficientnet_b0(weights=None)
        >>> model.classifier[1] = nn.Linear(1280, 4)
        >>> model = load_model_from_checkpoint(model, 'checkpoints_multiclass/best.pt', device)
        >>> model = model.to(device)
        >>> model.eval()
    """
    if device is None:
        device = torch.device('cpu')

    state_dict = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(state_dict, strict=strict)
    return model


if __name__ == '__main__':
    # Example usage
    import torch.nn as nn
    from torchvision import models

    print("Checkpoint Loading Utility - Example Usage")
    print("=" * 60)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create model
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(1280, 4)

    # Load checkpoint
    checkpoint_path = 'checkpoints_multiclass/best.pt'

    print(f"\nLoading checkpoint from: {checkpoint_path}")
    try:
        model = load_model_from_checkpoint(model, checkpoint_path, device)
        model = model.to(device)
        model.eval()
        print("✓ Checkpoint loaded successfully!")

        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Total parameters: {total_params:,}")
    except FileNotFoundError:
        print(f"✗ Checkpoint file not found: {checkpoint_path}")
    except RuntimeError as e:
        print(f"✗ Error loading checkpoint: {e}")
