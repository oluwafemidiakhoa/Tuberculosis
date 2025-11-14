# 🚀 Push TB Detection to GitHub

Follow these steps to push your TB detection project to GitHub (excluding Gradio app files).

## Step 1: Initialize Git Repository

```bash
cd c:/Users/adminidiakhoa/malaria_ast_starter/tb_detection_ast

# Initialize git
git init

# Add .gitignore (already created - excludes app.py, gradio_app/, large model files)
git add .gitignore

# Add all files (except those in .gitignore)
git add .
```

## Step 2: Create First Commit

```bash
# Commit with descriptive message
git commit -m "Initial commit: TB Detection with AST

- 99.29% accuracy, 89.52% energy savings
- Complete training pipeline with proven AST code
- Comprehensive visualizations (4-panel + headline)
- Grad-CAM explainability
- Training notebooks (TB_Training_Complete.ipynb)
- Metrics and results documentation

Excludes:
- Model checkpoints (too large for GitHub)
- Dataset files
- Gradio app (will be pushed to HF Space separately)
"
```

## Step 3: Connect to GitHub Remote

```bash
# Create new repository on GitHub first:
# Go to: https://github.com/oluwafemidiakhoa
# Click "New repository"
# Name: "Tuberculosis"
# Description: "TB Detection with AST - 99% accuracy, 89% energy savings"
# Public
# DO NOT initialize with README (we already have one)

# Then connect to remote:
git remote add origin https://github.com/oluwafemidiakhoa/Tuberculosis.git

# Verify remote
git remote -v
```

## Step 4: Push to GitHub

```bash
# Push to main branch
git branch -M main
git push -u origin main
```

## ✅ What Gets Pushed:

### Included ✅:
- `README.md` - Complete documentation with actual results
- `TB_Training_Complete.ipynb` - All-in-one training + visualization notebook
- `create_visualizations.py` - Visualization generation script
- `requirements.txt` - Python dependencies
- `.gitignore` - Excludes large files
- `checkpoints/metrics_ast.csv` - Training metrics (small file)
- `checkpoints/metrics_ast.jsonl` - Training history
- `visualizations/*.png` - Result plots
- `docs/` - Documentation
- `src/` - Source code
- `configs/` - Configuration files

### Excluded ❌ (per .gitignore):
- `checkpoints/*.pt` - Model weights (too large, will use Git LFS or HF)
- `data/` - Dataset (too large)
- `gradio_app/` - Gradio app (goes to HF Space separately)
- `app.py` - Gradio app file
- `*.pth`, `*.onnx` - Large model files

## Step 5: Verify Upload

```bash
# Check GitHub repo
# https://github.com/oluwafemidiakhoa/Tuberculosis

# You should see:
# - README with visualizations
# - Training notebook
# - Metrics files
# - Source code
```

## Alternative: Use GitHub Desktop

If you prefer a GUI:

1. Open GitHub Desktop
2. File → Add Local Repository
3. Choose: `c:/Users/adminidiakhoa/malaria_ast_starter/tb_detection_ast`
4. Create commit with message above
5. Publish repository to GitHub
6. Repository name: "Tuberculosis"
7. Push

---

## 📝 Notes:

- **Model checkpoints** are excluded (too large for GitHub)
  - Upload to Hugging Face Hub separately
  - Or use Git LFS: `git lfs track "*.pt"`

- **Visualizations** are included (PNG files are small)
  - Will display in README on GitHub

- **Gradio app** goes to Hugging Face Space
  - Separate deployment (next step)

---

## 🔒 If You Want to Include Model Checkpoints:

### Option A: Git LFS (Large File Storage)

```bash
# Install Git LFS
git lfs install

# Track model files
git lfs track "checkpoints/*.pt"
git add .gitattributes

# Now add and commit
git add checkpoints/best.pt
git commit -m "Add trained model checkpoint"
git push
```

### Option B: Upload to Hugging Face Hub

```bash
pip install huggingface_hub

# Upload model
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='checkpoints_tb_ast/best.pt',
    path_in_repo='best.pt',
    repo_id='mgbam/tb-detection-ast',
    repo_type='model'
)
"
```

Then reference it in README:
```markdown
Download model: https://huggingface.co/mgbam/tb-detection-ast/resolve/main/best.pt
```

---

## ✅ After Pushing:

1. **Verify README displays correctly** on GitHub
2. **Check visualizations show up** in README
3. **Test Colab notebook** link works
4. **Add topics**: `tuberculosis`, `medical-ai`, `adaptive-sparse-training`, `energy-efficient`, `chest-xray`
5. **Star your own repo** ⭐

---

**Ready to push? Run the commands above!** 🚀
