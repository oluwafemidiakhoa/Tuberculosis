# 🔧 Troubleshooting: Model Producing Random Predictions

## 🚨 Problem: All Predictions Show ~25% Confidence

If your Gradio app is showing predictions like this:

```
COVID-19:     25.1%
Normal:       25.1%
Tuberculosis: 25.0%
Pneumonia:    24.7%
```

**This means the model is producing RANDOM predictions** (25% is what you'd expect from random guessing across 4 classes).

## 🔍 Root Cause

The model checkpoint file `checkpoints/best_multiclass.pt` is either:
1. **Missing** - File doesn't exist
2. **Untrained** - Contains ImageNet weights only (not trained on medical data)
3. **Corrupted** - File exists but can't be loaded properly

## ✅ Solution Steps

### Step 1: Verify the Problem

```bash
# Check if checkpoint exists
ls -lh gradio_app/checkpoints/best_multiclass.pt

# OR from repository root:
ls -lh checkpoints/best_multiclass.pt
```

**Expected file size:** ~17-20 MB (for trained EfficientNet-B0)
**If missing or <1MB:** Model is not trained

### Step 2: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Or minimal install:
pip install torch torchvision gradio opencv-python pillow numpy matplotlib
```

### Step 3: Generate Model Checkpoint Structure

```bash
# From repository root:
python setup_model.py
```

This creates the correct model architecture file at `checkpoints/best_multiclass.pt`.

**⚠️ IMPORTANT:** This creates an **untrained** model with ImageNet weights. It will still produce random predictions for medical images!

### Step 4: Train the Model (Required for Accurate Predictions)

#### Option A: Quick Training (~2-3 hours, ~85% accuracy)

```bash
# Download dataset first:
# https://www.kaggle.com/datasets/anasmohammedtahir/covidqu

# Extract to ./data/ directory

# Run quick training:
python train_multiclass_simple.py
```

#### Option B: Best Performance (~8-12 hours, ~90-95% accuracy)

```bash
python train_optimized_90_95.py
```

#### Option C: Use Pre-trained Weights (If Available)

If you have a previously trained model:

```bash
# Copy to the correct location:
cp /path/to/your/trained_model.pt checkpoints/best_multiclass.pt

# OR for Gradio app:
cp /path/to/your/trained_model.pt gradio_app/checkpoints/best_multiclass.pt
```

### Step 5: Verify the Fix

```bash
# Run the Gradio app
cd gradio_app
python app.py
```

**Expected behavior:**
- ✅ No warning banner about untrained model
- ✅ Predictions vary significantly (not all ~25%)
- ✅ One class has clearly higher confidence than others
- ✅ Console shows: "✅ Multi-class model loaded successfully!"

## 📊 How to Tell if Model is Trained

### Untrained Model Signs:
- ❌ All predictions ~25% (±2%)
- ❌ Red warning banner in app UI
- ❌ Warning in interpretation text
- ❌ Console: "⚠️ WARNING: Model checkpoint not found"

### Trained Model Signs:
- ✅ Predictions vary: one class 60-95%, others much lower
- ✅ No warning banners
- ✅ Console: "✅ Multi-class model loaded successfully!"
- ✅ Reasonable predictions that match image content

## 🎯 Quick Test Cases

### Test with Known Images:

1. **Normal X-ray** → Should predict "Normal" with >60% confidence
2. **TB X-ray** → Should predict "Tuberculosis" with >70% confidence
3. **Pneumonia X-ray** → Should predict "Pneumonia" with >70% confidence
4. **COVID X-ray** → Should predict "COVID-19" with >60% confidence

If all images show ~25% for all classes → **Model is untrained**

## 🔄 For Hugging Face Spaces Deployment

### Problem: Space shows random predictions

**Solution:**

1. **Train model locally** (see Step 4 above)

2. **Upload checkpoint to Hugging Face Space:**

   ```bash
   # Install Git LFS (for files >10MB)
   git lfs install

   # Clone your Space
   git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE
   cd YOUR_SPACE

   # Track model files with LFS
   git lfs track "checkpoints/*.pt"
   git add .gitattributes

   # Copy trained model
   mkdir -p checkpoints
   cp /path/to/trained/best_multiclass.pt checkpoints/

   # Commit and push
   git add checkpoints/best_multiclass.pt
   git commit -m "Add trained model checkpoint"
   git push
   ```

3. **Alternative: Upload via Hugging Face UI:**
   - Go to your Space's "Files" tab
   - Create `checkpoints` folder
   - Upload `best_multiclass.pt` (drag & drop)
   - Space will automatically restart

## 📚 Dataset Information

**Required Dataset:** COVID-QU-Ex
**Source:** https://www.kaggle.com/datasets/anasmohammedtahir/covidqu
**Size:** ~33,920 chest X-ray images
**Classes:** Normal, Tuberculosis, Pneumonia, COVID-19

**Directory structure after download:**
```
data/
├── train/
│   ├── Normal/
│   ├── Tuberculosis/
│   ├── Pneumonia/
│   └── COVID-19/
├── val/
│   └── (same structure)
└── test/
    └── (same structure)
```

## 🆘 Still Having Issues?

### Check Console Output:

```bash
cd gradio_app
python app.py
```

Look for these messages:
- ✅ "✅ Multi-class model loaded successfully!" → Good!
- ❌ "⚠️ WARNING: Model checkpoint not found" → Need to create/train model
- ❌ "⚠️ Error loading model: ..." → Check error message for details

### Common Errors:

**Error: "No such file or directory: 'checkpoints/best_multiclass.pt'"**
- **Fix:** Run `python ../setup_model.py` from gradio_app directory
- **OR:** Run `python setup_model.py` from repository root

**Error: "RuntimeError: Error(s) in loading state_dict"**
- **Fix:** Model architecture mismatch. Delete checkpoint and regenerate:
  ```bash
  rm checkpoints/best_multiclass.pt
  python setup_model.py
  ```

**Error: "ModuleNotFoundError: No module named 'torch'"**
- **Fix:** Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## 📞 Getting Help

If you're still experiencing issues:

1. **Check you have:**
   - ✅ Python 3.8+
   - ✅ All dependencies installed (`pip install -r requirements.txt`)
   - ✅ Sufficient disk space (~5GB for dataset + models)
   - ✅ GPU (optional but recommended for training)

2. **Verify paths:**
   - Model file location matches what app expects
   - Dataset directory structure is correct

3. **Open an issue on GitHub:**
   - Include console output
   - Include screenshot of predictions
   - Include output of `ls -lh checkpoints/`

---

**Remember:** An untrained model will ALWAYS produce ~25% predictions. You MUST train it on medical data to get meaningful results!
