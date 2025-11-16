# 🔧 Troubleshooting Guide

## Common Issues & Solutions

### Issue 1: FileNotFoundError for metrics_ast.csv

**Error**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'checkpoints_multiclass/metrics_ast.csv'
```

**Cause**: Training script hasn't run yet or failed

**Solution**:
1. Make sure training cell completed successfully
2. Check if `checkpoints_multiclass/` folder exists
3. Look for error messages in training output
4. If training failed, scroll up to see the actual error

---

### Issue 2: Unrecognized arguments error

**Error**:
```
train_ast_multiclass.py: error: unrecognized arguments: --data_dir data_multiclass...
```

**Cause**: Old training script doesn't support those arguments

**Solution**:
✅ **FIXED** - The notebook now uses `train_multiclass_simple.py` which doesn't need arguments!

Just run:
```python
!python train_multiclass_simple.py
```

---

### Issue 3: Pie chart ValueError

**Error**:
```
ValueError: 'explode' must be of length 'x', not 4
```

**Cause**: Dataset doesn't have exactly 4 classes

**Solution**:
✅ **FIXED** - Notebook now uses dynamic explode:
```python
explode = tuple([0.05] * len(class_counts))
```

This automatically matches the number of classes found.

---

### Issue 4: No images in dataset folders

**Error**:
```
TRAIN: 0 images
VAL: 0 images
TEST: 0 images
```

**Cause**: Dataset download or organization failed

**Solution**:
1. Check if datasets downloaded:
   ```bash
   !ls data_covid/
   !ls data_pneumonia/
   !ls data_tb/
   ```

2. Check Kaggle API:
   - Re-upload `kaggle.json`
   - Verify it's in `~/.kaggle/`
   - Check permissions: `!ls -la ~/.kaggle/`

3. Manually check dataset structure:
   ```bash
   !find data_covid -name "*.png" | head -10
   !find data_pneumonia -name "*.png" | head -10
   !find data_tb -name "*.png" | head -10
   ```

4. If downloads failed, try manual download:
   ```bash
   # Download individually
   !kaggle datasets download -d tawsifurrahman/covid19-radiography-database
   !unzip -q covid19-radiography-database.zip -d data_covid
   ```

---

### Issue 5: CUDA out of memory

**Error**:
```
RuntimeError: CUDA out of memory
```

**Solution**:
1. Reduce batch size in training script:
   ```python
   # Edit train_multiclass_simple.py
   CONFIG = {
       'batch_size': 16,  # Changed from 32
       ...
   }
   ```

2. Or restart runtime and clear memory:
   - Runtime → Factory reset runtime
   - Re-run from beginning

3. Use Colab Pro for more RAM

---

### Issue 6: Model not converging

**Symptoms**:
- Accuracy stuck at ~25% (random for 4 classes)
- Loss not decreasing

**Solutions**:
1. Check if data loaded correctly:
   ```python
   print(f"Train samples: {len(train_dataset)}")
   print(f"Val samples: {len(val_dataset)}")
   ```

2. Verify images are readable:
   ```python
   img, label = train_dataset[0]
   print(f"Image shape: {img.shape}")
   print(f"Label: {CLASSES[label]}")
   ```

3. Check learning rate:
   ```python
   # Try lower learning rate
   CONFIG['lr'] = 0.0001  # Instead of 0.0003
   ```

---

### Issue 7: Training too slow

**Symptoms**:
- Each epoch takes >30 minutes
- Progress bar barely moving

**Solutions**:
1. Check GPU is enabled:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   print(torch.cuda.get_device_name(0))  # Should show GPU name
   ```

2. Enable GPU in Colab:
   - Runtime → Change runtime type → Hardware accelerator → GPU

3. Reduce dataset size:
   ```python
   # In data organization cell
   max_count=1500  # Instead of 3000
   ```

---

### Issue 8: Grad-CAM showing blank/black heatmap

**Cause**: Gradients not flowing properly

**Solution**:
1. Make sure `torch.set_grad_enabled(True)`:
   ```python
   with torch.set_grad_enabled(True):
       cam, output = grad_cam.generate(input_tensor)
   ```

2. Check if model is in eval mode:
   ```python
   model.eval()  # Should be called before Grad-CAM
   ```

3. Verify target layer:
   ```python
   target_layer = model.features[-1]  # For EfficientNet
   print(target_layer)  # Should show Conv2d layer
   ```

---

### Issue 9: Download fails at end

**Error**:
```
MessageError: TypeError: Failed to fetch
```

**Cause**: Browser popup blocker or timeout

**Solution**:
1. Manually download from Files panel:
   - Click folder icon on left
   - Navigate to files
   - Right-click → Download

2. Or use Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')

   # Copy files to Drive
   !cp checkpoints_multiclass/best.pt /content/drive/MyDrive/
   !cp *.png /content/drive/MyDrive/
   ```

3. Or create zip:
   ```python
   !zip -r results.zip checkpoints_multiclass/ *.png
   files.download('results.zip')
   ```

---

### Issue 10: Confusion matrix shows poor results

**Symptoms**:
- Diagonal not darkest
- High off-diagonal values
- Pneumonia → TB misclassification

**Solutions**:
1. Check if using best model with proper loading:

   **Option A: Use the helper utility (recommended)**
   ```python
   from load_checkpoint_utils import load_model_from_checkpoint

   model = load_model_from_checkpoint(model, 'checkpoints_multiclass/best.pt', device)
   ```

   **Option B: Manual loading**
   ```python
   # Load checkpoint with proper key handling
   checkpoint = torch.load('checkpoints_multiclass/best.pt', map_location=device)

   # Handle different checkpoint formats
   if isinstance(checkpoint, dict):
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
           elif key == 'activation_mask':
               continue  # Skip non-model keys
           else:
               new_state_dict[key] = value

       state_dict = new_state_dict
   else:
       state_dict = checkpoint

   model.load_state_dict(state_dict)
   ```

2. Verify test data is correct:
   ```python
   for cls in CLASSES:
       path = Path(f'data_multiclass/test/{cls}')
       count = len(list(path.glob('*.png')))
       print(f"{cls}: {count} test images")
   ```

3. Check if need more training:
   - Look at training loss - still decreasing?
   - Try more epochs (e.g., 100 instead of 50)

4. Review class balance:
   - Make sure each class has similar number of samples
   - Imbalanced data leads to poor performance

---

## Quick Diagnostic Checklist

Run these checks if anything goes wrong:

```python
# 1. Check GPU
import torch
print(f"CUDA: {torch.cuda.is_available()}")

# 2. Check datasets downloaded
from pathlib import Path
for d in ['data_covid', 'data_pneumonia', 'data_tb']:
    if Path(d).exists():
        files = list(Path(d).rglob('*.png'))
        print(f"{d}: {len(files)} images")
    else:
        print(f"{d}: NOT FOUND")

# 3. Check data organization
for split in ['train', 'val', 'test']:
    for cls in ['Normal', 'TB', 'Pneumonia', 'COVID']:
        path = Path(f'data_multiclass/{split}/{cls}')
        if path.exists():
            count = len(list(path.glob('*.png')))
            print(f"{split}/{cls}: {count}")

# 4. Check training outputs
import os
if os.path.exists('checkpoints_multiclass'):
    files = os.listdir('checkpoints_multiclass')
    print(f"Checkpoint files: {files}")
else:
    print("checkpoints_multiclass NOT FOUND")

# 5. Check if metrics file exists
if os.path.exists('checkpoints_multiclass/metrics_ast.csv'):
    import pandas as pd
    df = pd.read_csv('checkpoints_multiclass/metrics_ast.csv')
    print(f"Metrics: {len(df)} epochs")
    print(f"Best accuracy: {df['val_acc'].max()*100:.2f}%")
else:
    print("metrics_ast.csv NOT FOUND - training didn't complete")
```

---

## Still Having Issues?

### Option 1: Restart Everything
1. Runtime → Factory reset runtime
2. Re-run all cells from the beginning
3. Make sure to upload kaggle.json again

### Option 2: Check GitHub Issues
https://github.com/oluwafemidiakhoa/Tuberculosis/issues

Create a new issue with:
- Error message (full traceback)
- Which cell failed
- Output from diagnostic checklist above

### Option 3: Use Alternative Dataset

If dataset download keeps failing, use a simpler approach:

```python
# Option: Use smaller, pre-organized dataset
# (You'll need to find and download separately)

# Or reduce scope to 2-3 classes instead of 4
# Just to verify the training pipeline works
```

---

## Success Checklist

You know it's working when you see:

✅ All 3 datasets downloaded (COVID, Pneumonia, TB)
✅ Data organized into 4 classes with ~2000-3000 images each
✅ Training starts and shows progress bar
✅ Loss decreasing over epochs
✅ Accuracy increasing over epochs
✅ Activation rate ~10%
✅ Energy savings ~89%
✅ Best model saved
✅ metrics_ast.csv created
✅ Visualizations generated successfully
✅ Grad-CAM showing colored heatmaps
✅ Confusion matrix showing strong diagonal
✅ **Pneumonia correctly identified, NOT as TB!**

---

## Performance Expectations

### Good Results:
- Overall accuracy: 92-97%
- Per-class accuracy: >90% for all classes
- Energy savings: 85-92%
- Training time: 3-4 hours (50 epochs, Colab T4 GPU)

### If Results Are Lower:
- 80-90% accuracy: Acceptable, but could improve with more epochs
- 70-80% accuracy: Check data quality, class balance
- <70% accuracy: Something is wrong, check diagnostics

### Red Flags:
- Accuracy stuck at 25% (random guessing for 4 classes)
- Loss not decreasing after 10 epochs
- Energy savings <50% (AST not working)
- Activation rate >50% (too dense, not sparse)

---

**Remember**: The goal is not just high accuracy, but **correct specificity**!

✅ Pneumonia should predict as Pneumonia (not TB)
✅ Each disease should have distinct Grad-CAM pattern
✅ Confusion matrix diagonal should be darkest

**The specificity fix is the KEY improvement, not just raw accuracy!**
