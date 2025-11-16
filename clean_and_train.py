"""
Clean the dataset and restart training.
This script will:
1. Remove all corrupted images
2. Retrain the model with clean data
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from fix_corrupted_images import verify_and_fix_images

print("=" * 80)
print("STEP 1: Cleaning corrupted images from dataset")
print("=" * 80)

# Fix corrupted images in the multiclass dataset
corrupted = verify_and_fix_images('data_multiclass', backup=True)

print(f"\n✓ Removed {len(corrupted)} corrupted images")
print("\n" + "=" * 80)
print("STEP 2: Starting training with clean dataset")
print("=" * 80)

# Now import and run the training
print("\nImporting training modules...")
import subprocess
import sys

# Run the training script
result = subprocess.run([sys.executable, 'train_multiclass_simple.py'],
                        capture_output=False, text=True)

if result.returncode == 0:
    print("\n✓ Training completed successfully!")
else:
    print(f"\n✗ Training failed with return code: {result.returncode}")
    sys.exit(result.returncode)
