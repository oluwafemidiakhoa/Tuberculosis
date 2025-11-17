# Quick Guide: best.pt for HuggingFace Deployment

## 🎯 What You Need

Your HuggingFace Space deployment needs a `best.pt` file in the **root directory** of your space.

## 📦 What is best.pt?

`best.pt` contains the trained weights for your multi-class chest X-ray classifier:
- Architecture: EfficientNet-B0
- Classes: Normal, Tuberculosis, Pneumonia, COVID-19
- File size: ~21 MB
- Format: PyTorch state_dict

## 🚀 Quick Start (3 Options)

### Option 1: Upload Your Trained Model ⭐ RECOMMENDED

If you already trained the model (90.17% accuracy from TRAINING_RESULTS.md):

1. **Find your checkpoint:**
   - Location: `checkpoints_multiclass_optimized/best.pt`
   - Or wherever your training saved it

2. **Copy to root:**
   ```bash
   cp path/to/your/trained/best.pt ./best.pt
   ```

3. **Commit and push:**
   ```bash
   git add best.pt
   git commit -m "Add trained checkpoint (90.17% accuracy)"
   git push
   ```

4. **Upload to HuggingFace:**
   - Visit: https://huggingface.co/spaces/mgbam/Tuberculosis
   - Files → Upload `best.pt`

### Option 2: Create Demo Weights (Testing Only)

For testing the deployment without real training:

1. **Run the generator script:**
   ```bash
   python create_best_checkpoint.py
   ```
   This creates `best.pt` with ImageNet pretrained weights.

2. **Deploy:**
   ```bash
   git add best.pt
   git commit -m "Add demo checkpoint for testing"
   git push
   ```

⚠️ **Note**: Demo weights will give random medical predictions. Replace with trained weights for production.

### Option 3: Train the Model

Train a new model to create `best.pt`:

```bash
# Quick training (2-3 hours on GPU)
python train_multiclass_simple.py

# OR best performance training (6-8 hours on GPU)
python train_best.py
```

After training completes:
```bash
cp checkpoints_multiclass_best/best.pt ./best.pt
```

## ✅ Verify Your Checkpoint

Before deploying, test that it loads correctly:

```python
import torch

# Load and verify
checkpoint = torch.load('best.pt', map_location='cpu')
print(f"✅ Checkpoint loaded!")
print(f"   Keys: {list(checkpoint.keys())[:5]}...")  # Should show model layer names
```

## 🌐 Deploy to HuggingFace

Once you have `best.pt`:

```bash
# Upload to your HF Space
# Method 1: Via web interface
# - Go to https://huggingface.co/spaces/mgbam/Tuberculosis
# - Upload best.pt, app.py, requirements.txt

# Method 2: Via git
git clone https://huggingface.co/spaces/mgbam/Tuberculosis hf_space
cd hf_space
cp ../best.pt .
cp ../gradio_app/app.py .
git add .
git commit -m "Deploy multi-class TB detector"
git push
```

## 📋 Files Needed for Deployment

Your HuggingFace Space needs:

```
hf_space/
├── best.pt                  # ← Model weights (this file!)
├── app.py                   # ← From gradio_app/app.py
├── requirements.txt         # ← From gradio_app/requirements.txt
└── README.md               # ← Space description (optional)
```

## 🔧 Troubleshooting

**"ModuleNotFoundError: No module named 'torch'"** when running create_best_checkpoint.py
```bash
pip install torch torchvision
```

**"FileNotFoundError: best.pt" in HuggingFace Space**
- Make sure `best.pt` is in the root directory of your Space, not in a subdirectory

**"RuntimeError: Error loading state_dict"**
- Your checkpoint architecture doesn't match the app
- Make sure you're using EfficientNet-B0 with 4 output classes

## 📖 More Information

See `CHECKPOINT_GUIDE.md` for detailed instructions and advanced options.

## ⏭️ What's Next?

1. ✅ Get or create `best.pt` (choose an option above)
2. ✅ Verify it loads correctly
3. ✅ Upload to HuggingFace Space
4. ✅ Test your deployment!

---

**Questions?** Check the full guide in `CHECKPOINT_GUIDE.md`
