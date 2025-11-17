# 📦 best.pt Checkpoint Guide for HuggingFace Space Deployment

## 🎯 Overview

Your Gradio app (`gradio_app/app.py`) expects a `best.pt` checkpoint file for deployment to HuggingFace Spaces. This file contains the trained model weights for the multi-class chest X-ray classifier.

## 📍 Current Situation

Based on your training results (TRAINING_RESULTS.md):
- ✅ **Model trained**: 90.17% validation accuracy achieved
- ✅ **Architecture**: EfficientNet-B0 with 4 classes (Normal, TB, Pneumonia, COVID-19)
- ❌ **Checkpoint missing**: The trained weights are not in this repository

## 🔍 Where the App Looks for Checkpoints

From `gradio_app/app.py` (lines 34-48):

```python
try:
    # Try loading best.pt from root directory (HuggingFace Spaces location)
    model.load_state_dict(torch.load('best.pt', map_location=device))
    print("✅ Multi-class model loaded successfully from best.pt!")
except Exception as e:
    print(f"⚠️  Error loading model from best.pt: {e}")
    try:
        # Fallback to checkpoints directory
        model.load_state_dict(torch.load('checkpoints/best_multiclass.pt', map_location=device))
        print("✅ Multi-class model loaded successfully from checkpoints/best_multiclass.pt!")
    except Exception as e2:
        # Error - no model found
```

**Expected locations:**
1. **Primary**: `best.pt` (root directory) ← **For HuggingFace Space**
2. **Fallback**: `checkpoints/best_multiclass.pt`

## 📋 Options to Get best.pt

### Option 1: Use Your Trained Model (RECOMMENDED)

If you trained the model and achieved 90.17% accuracy:

**If you trained on Google Colab:**
```python
# In your Colab notebook after training
from google.colab import files

# Download the checkpoint
files.download('checkpoints_multiclass_optimized/best.pt')
```

**If you trained locally:**
```bash
# Copy the trained checkpoint
cp checkpoints_multiclass_optimized/best.pt best.pt

# Or wherever your training script saved it
find . -name "best.pt" -type f
```

Then commit and push:
```bash
git add best.pt
git commit -m "📦 Add trained model checkpoint (90.17% accuracy)"
git push -u origin claude/add-tb-checkpoint-01U8pa3ygjx7W1S2kiepNhVg
```

### Option 2: Generate Demo Weights (For Testing Only)

If you don't have the trained weights yet, create a demo checkpoint:

**I've created a script** at `create_best_checkpoint.py` that generates a properly structured checkpoint.

**To use it:**
```bash
# Install PyTorch (if not already installed)
pip install torch torchvision

# Run the script
python create_best_checkpoint.py
```

This creates `best.pt` with:
- ✅ Correct architecture (EfficientNet-B0, 4 classes)
- ✅ Pretrained ImageNet weights (base)
- ⚠️ NOT trained on chest X-rays yet (random medical predictions)

**Note**: This is only for testing the deployment infrastructure. Replace with trained weights for production.

### Option 3: Train the Model Now

Run the training script to create a new checkpoint:

```bash
# Option A: Quick training (2-3 hours on GPU)
python train_multiclass_simple.py

# Option B: Best performance training (6-8 hours on GPU)
python train_best.py

# Option C: Optimized training (aims for 90-95%)
python train_optimized_90_95.py
```

After training, copy the checkpoint:
```bash
# The script saves to checkpoints_multiclass_best/best.pt
cp checkpoints_multiclass_best/best.pt best.pt
```

## 🚀 Deployment to HuggingFace Space

Once you have `best.pt`:

### Step 1: Verify the Checkpoint

```python
import torch
from torchvision import models
import torch.nn as nn

# Load checkpoint
checkpoint = torch.load('best.pt', map_location='cpu')

# Create model
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)

# Load weights
model.load_state_dict(checkpoint)
print("✅ Checkpoint loads successfully!")
```

### Step 2: Push to GitHub

```bash
git add best.pt
git commit -m "📦 Add trained model checkpoint for deployment"
git push -u origin claude/add-tb-checkpoint-01U8pa3ygjx7W1S2kiepNhVg
```

### Step 3: Deploy to HuggingFace Space

**Method A: Direct Upload**
1. Go to your HuggingFace Space: https://huggingface.co/spaces/mgbam/Tuberculosis
2. Click "Files" → "Add file" → "Upload files"
3. Upload `best.pt` (will be ~21 MB)
4. Upload `gradio_app/app.py`
5. Upload `gradio_app/requirements.txt`

**Method B: Git Push**
```bash
# Clone your HF Space
git clone https://huggingface.co/spaces/mgbam/Tuberculosis hf_space
cd hf_space

# Copy files
cp ../best.pt .
cp ../gradio_app/app.py .
cp ../gradio_app/requirements.txt .

# Commit and push
git add .
git commit -m "🚀 Deploy Multi-Class TB Detection v1.0

- Model: EfficientNet-B0
- Accuracy: 90.17%
- Classes: Normal, TB, Pneumonia, COVID-19
- Energy Efficient: 77% savings with AST"

git push
```

## 📊 Checkpoint File Details

**Expected checkpoint specifications:**

| Property | Value |
|----------|-------|
| **Format** | PyTorch state_dict |
| **Architecture** | EfficientNet-B0 |
| **Input size** | 224×224×3 |
| **Output classes** | 4 (Normal, TB, Pneumonia, COVID-19) |
| **File size** | ~21 MB |
| **File type** | .pt (PyTorch) |

**Model architecture:**
```python
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(1280, 4)  # 1280 input features, 4 output classes
```

## ⚠️ Important Notes

### File Size for HuggingFace
- **HuggingFace Spaces**: Supports files up to 500MB directly
- **Git LFS**: For files >10MB, HF automatically uses Git LFS
- **best.pt size**: ~21 MB (within limits, no special handling needed)

### Version Control
```bash
# Check if Git LFS is needed (it should auto-configure on HF)
git lfs track "*.pt"
git add .gitattributes
```

### Testing Before Deployment
```bash
# Test the Gradio app locally
cd gradio_app
python app.py

# Visit http://localhost:7860 to test
```

## 🔧 Troubleshooting

### "best.pt not found" error in HF Space
**Solution**: Upload `best.pt` to the root of your HF Space repository

### "Cannot load model" error
**Cause**: Architecture mismatch between checkpoint and app
**Solution**: Verify the model architecture matches:
```python
# In app.py
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(1280, 4)  # Must match checkpoint
```

### "Out of memory" during loading
**Cause**: HF Space has limited RAM
**Solution**: Use CPU-only deployment (already configured in app.py with `map_location=device`)

### Checkpoint file is corrupt
**Solution**: Re-download or regenerate the checkpoint
```bash
# Test checkpoint integrity
python -c "import torch; torch.load('best.pt', map_location='cpu'); print('OK')"
```

## 📚 Related Documentation

- `DEPLOYMENT_INSTRUCTIONS.md` - Full deployment guide
- `TRAINING_RESULTS.md` - Training performance metrics
- `gradio_app/app.py` - Gradio application code
- `create_best_checkpoint.py` - Script to generate demo weights

## ✅ Next Steps

1. **Choose your option** above (use trained model, demo weights, or train now)
2. **Create/obtain best.pt**
3. **Verify it loads** with the test script above
4. **Push to GitHub** on your branch
5. **Deploy to HuggingFace Space**
6. **Test the deployment** with example X-rays

## 🤝 Need Help?

**If you have the trained model but can't find it:**
- Check `checkpoints_multiclass_optimized/` directory
- Check Google Colab downloads folder
- Check your training logs for save locations

**If you need to train from scratch:**
- Use Google Colab with GPU for faster training
- Run `train_best.py` for best results
- Training time: 6-8 hours on GPU

**If deploying fails:**
- Check HuggingFace Space logs
- Verify checkpoint file size
- Test app.py locally first

---

**Ready to deploy!** 🚀

Follow the steps above to get `best.pt` and deploy your multi-class TB detection model to HuggingFace Spaces.
