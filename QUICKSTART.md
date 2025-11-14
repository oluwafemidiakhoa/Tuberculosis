# 🚀 TB Detection - Quick Start Guide

Get started with TB detection in 5 minutes!

---

## ✅ What's Already Done

- ✅ Project structure created
- ✅ Kaggle credentials configured
- ✅ README and documentation written
- ✅ Requirements file ready
- ✅ Data exploration notebook created
- 🔄 TBX11K dataset downloading (in progress)

---

## 🎯 Next Steps

### 1. Install Dependencies

```bash
cd tb_detection_ast
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Wait for Dataset Download

The TBX11K dataset (11,200 chest X-rays) is currently downloading to `data/raw/`.

**Check download status:**
```bash
ls -lh data/raw/
```

**Expected size:** ~2-3 GB

### 3. Extract Dataset

```bash
cd data/raw
unzip tbx-11.zip
cd ../..
```

### 4. Explore the Data

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

This notebook will:
- ✅ Analyze class distribution
- ✅ Check image properties
- ✅ Visualize sample X-rays
- ✅ Create train/val/test splits (70/15/15)

### 5. Copy AST Code from Malaria Project

```bash
# Copy your proven AST implementation
cp -r ../malaria_ast_starter/malaria_ast_starter/ast_lib ./
```

### 6. Train Baseline Model (No AST)

```bash
# Coming soon - baseline training script
python src/train_baseline.py
```

**Expected Results:**
- Accuracy: 85-92%
- Training time: ~2 hours on GPU
- Energy: 100% (baseline)

### 7. Train with AST

```bash
# Coming soon - AST training script
python src/train_ast.py
```

**Target Results:**
- Accuracy: 90-94%
- Training time: ~2-3 hours on GPU
- Energy savings: 85-90%

### 8. Run Demo

```bash
cd gradio_app
python app.py
```

Opens Gradio interface for TB detection with Grad-CAM visualizations.

---

## 📂 Project Structure Overview

```
tb_detection_ast/
├── data/
│   ├── raw/              # TBX11K dataset (downloading)
│   ├── processed/        # Preprocessed X-rays
│   └── splits/           # train.csv, val.csv, test.csv
│
├── notebooks/
│   └── 01_data_exploration.ipynb  # ✅ Ready to use
│
├── src/                  # Training scripts (coming soon)
├── ast_lib/              # Copy from malaria project
├── gradio_app/           # Demo app (coming soon)
│
├── README.md             # ✅ Complete
├── requirements.txt      # ✅ Ready
└── QUICKSTART.md         # This file
```

---

## 🎯 Your 5-Week Plan

### Week 1: Data & Setup ✅
- [x] Download TBX11K dataset
- [x] Create project structure
- [x] Explore data
- [ ] Preprocess images

### Week 2: Baseline Model
- [ ] Train EfficientNet-B0 (no AST)
- [ ] Evaluate on test set
- [ ] Establish baseline metrics

### Week 3: AST Integration
- [ ] Copy AST code from malaria project
- [ ] Adapt for chest X-rays
- [ ] Train with AST
- [ ] Compare with baseline

### Week 4: Explainability
- [ ] Add Grad-CAM visualizations
- [ ] Analyze model predictions
- [ ] Document failure cases

### Week 5: Deployment
- [ ] Build Gradio demo
- [ ] Deploy to Hugging Face
- [ ] Write Medium article
- [ ] Share on social media

---

## 💡 Key Differences from Malaria Project

| Aspect | Malaria | TB Detection |
|--------|---------|--------------|
| **Images** | Blood cells | Chest X-rays |
| **Size** | 128x128 | 512x512 |
| **Colors** | RGB | Grayscale |
| **Dataset** | 27K | 11K |
| **Task** | Binary | Multi-class |

**AST stays the same!** Your Sundew algorithm works across modalities.

---

## 🆘 Troubleshooting

### Dataset Not Downloading?
```bash
# Manual download
kaggle datasets download -d usmanshams/tbx-11
```

### Kaggle API Issues?
```bash
# Verify credentials
cat ~/.kaggle/kaggle.json

# Re-download from https://www.kaggle.com/settings
```

### Out of Memory?
```python
# Reduce batch size in training config
batch_size = 16  # instead of 32
```

### Slow Training?
```python
# Use smaller image size
input_size = 224  # instead of 512
```

---

## 📊 Expected Timeline

- **Data prep**: 1-2 days
- **Baseline training**: 1 day
- **AST integration**: 2-3 days
- **Demo creation**: 1 day
- **Total**: ~1 week to working prototype

---

## 🎯 Success Criteria

### Minimum Viable Product (MVP):
- ✅ 90%+ accuracy on test set
- ✅ 85%+ energy savings
- ✅ Working Gradio demo
- ✅ Deployed on Hugging Face

### Stretch Goals:
- 🎯 95%+ sensitivity (critical for TB)
- 🎯 Grad-CAM bounding box localization
- 🎯 Multi-disease classifier (TB + Pneumonia + COVID)
- 🎯 Mobile app deployment

---

## 🚀 Ready to Code?

Start with:
```bash
# 1. Check dataset download
ls -lh data/raw/

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter
jupyter notebook notebooks/01_data_exploration.ipynb
```

---

## 🤝 Questions?

Check the main [README.md](README.md) or [TB_DETECTION_PLAN.md](../TB_DETECTION_PLAN.md) for detailed information.

**Let's build something impactful!** 🌍✨
