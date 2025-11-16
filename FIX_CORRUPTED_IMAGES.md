# How to Fix Corrupted Images Issue

## Problem
Your training is encountering hundreds of corrupted Pneumonia images that cannot be loaded by PIL/Pillow. This causes:
- Images to be skipped during training
- Reduced effective dataset size
- Inconsistent batch sizes
- Potential training instability

## Solution

### Option 1: Quick Fix (Recommended)

Add this code cell **before** running `train_multiclass_simple.py`:

```python
# Fix corrupted images
import os
from PIL import Image
import shutil

def remove_corrupted_images(data_dir='data_multiclass'):
    """Remove all corrupted images from the dataset."""
    corrupted = []
    total = 0

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                total += 1
                file_path = os.path.join(root, file)

                try:
                    with Image.open(file_path) as img:
                        img.verify()
                    with Image.open(file_path) as img:
                        img.load()
                except:
                    print(f"Removing corrupted: {file_path}")
                    os.remove(file_path)
                    corrupted.append(file_path)

    print(f"\n✓ Removed {len(corrupted)} corrupted images out of {total} total")
    return corrupted

# Run the cleanup
corrupted_files = remove_corrupted_images()
```

### Option 2: Use the Provided Script

Run the cleanup script:

```python
# In your notebook
!python fix_corrupted_images_notebook.py
```

This will:
1. Scan all images in `data_multiclass/`
2. Backup corrupted images to `data_multiclass_corrupted_backup/`
3. Remove them from the training set
4. Print a summary of what was removed

### Option 3: Rebuild the Dataset

If you want to start fresh, you can rebuild the multiclass dataset:

```python
# Re-run the data preparation
!python prepare_data_multiclass.py --train-size 2000 --val-size 500
```

## Expected Results

Based on your training output, you should expect:
- **~500-700 corrupted Pneumonia images** to be removed
- This represents a small fraction of your dataset
- Training will run faster without trying to load corrupted files
- No more "Warning: Corrupted image found" messages

## After Cleanup

Once the corrupted images are removed, restart training:

```python
!python train_multiclass_simple.py
```

You should see:
- No corruption warnings
- Consistent batch processing
- Faster training iterations
- Better GPU utilization

## Prevention

To prevent this issue in future datasets:

1. **Verify images during download:**
   ```python
   def verify_image(path):
       try:
           Image.open(path).verify()
           return True
       except:
           return False
   ```

2. **Add verification to data preparation scripts** (already included in updated `prepare_data_multiclass.py`)

3. **Use the error handling in the training script** (already implemented)

## Troubleshooting

If you still see corrupted image warnings after cleanup:
1. Re-run the cleanup script
2. Check if new corrupted images appeared
3. Consider re-downloading the Pneumonia dataset:
   ```python
   !kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
   ```

## Technical Details

The corrupted images cannot be loaded because:
- File headers are malformed
- Image data is incomplete
- Compression errors
- Download interruptions

The cleanup script uses PIL's `verify()` and `load()` methods to detect these issues.
