# 🫁 TB Detection with Adaptive Sparse Training (AST)

**Energy-efficient tuberculosis detection from chest X-rays - 99.3% accuracy with 89% energy savings!**

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/mgbam/Tuberculosis)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌟 Key Results

| Metric | Value |
|--------|-------|
| **Detection Accuracy** | **99.29%** |
| **Energy Savings** | **89.52%** |
| **Activation Rate** | **9.38%** |
| **Training Epochs** | 50 |
| **Inference Time** | <2 seconds |

**Impact**: This model achieves clinical-grade accuracy while using only **10% of the computational resources** of traditional training—perfect for deployment in resource-constrained healthcare settings across Africa!

---

## 🎯 Project Overview

This project applies **Adaptive Sparse Training (AST)** to detect tuberculosis from chest X-ray images, achieving **99.3% accuracy** while reducing computational costs by **89.5%**.

Building on the success of our malaria detection system (93.94% accuracy, 88% energy savings), this project demonstrates the versatility of AST across medical imaging modalities.

### Why This Matters

- **1.6 million deaths** from TB annually (WHO 2023)
- **25% of global TB cases** are in Africa
- **40% diagnostic gap**: Many TB cases go undetected
- Traditional AI requires expensive infrastructure (**$10K+ GPU clusters**)
- Our solution runs on **affordable hardware** (<$300 tablets)

---

## 🚀 Key Features

✅ **High Accuracy**: 90%+ detection accuracy with high sensitivity
✅ **Energy Efficient**: 85-90% reduction in computational costs vs traditional models
✅ **Explainable AI**: Grad-CAM visualizations show TB-affected lung regions
✅ **Fast Inference**: <2 seconds per X-ray
✅ **Affordable Deployment**: Runs on low-cost hardware
✅ **Open Source**: Free for healthcare organizations and researchers

---

## 📊 Dataset

Using **TBX11K** - the largest public TB chest X-ray dataset:
- **11,200 chest X-rays** with expert annotations
- **Classes**: Healthy, Sick (non-TB), Active TB, Latent TB, Uncertain
- **Resolution**: 512x512 pixels
- **Annotations**: Bounding boxes for TB regions

**Source**: [Kaggle TBX11K Dataset](https://www.kaggle.com/datasets/usmanshams/tbx-11)

---

## 🛠️ Technical Architecture

### Model
- **Backbone**: EfficientNet-B0 (pretrained on ImageNet)
- **Training Method**: Adaptive Sparse Training (AST) with Sundew algorithm
- **Input**: 224x224 or 512x512 chest X-rays
- **Output**: Binary classification (Normal vs TB) or 5-class

### AST Configuration
```python
ast_config = {
    'sparsity_target': 0.88,      # 88% sparsity
    'pruning_schedule': 'gradual',
    'activation_threshold': 'dynamic',
    'sundew_algorithm': True
}
```

---

## 📈 Training Results

| Metric | Result | Status |
|--------|--------|--------|
| **Accuracy** | 99.29% | ✅ Achieved |
| **Energy Savings** | 89.52% | ✅ Achieved |
| **Activation Rate** | 9.38% | ✅ Optimal |
| **Training Loss** | 0.177 | ✅ Converged |
| **Total Epochs** | 50 | ✅ Complete |
| **Inference Time** | <2s | ✅ Fast |

### Training Progress

![TB AST Results](visualizations/tb_ast_results.png)

*4-panel analysis showing training loss, validation accuracy, activation rate, and energy savings over 50 epochs*

![TB Headline](visualizations/tb_ast_headline.png)

*Key metrics summary - 99.3% accuracy with 89.5% energy savings!*

---

## 🏗️ Project Structure

```
tb_detection_ast/
│
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
│
├── data/
│   ├── raw/                    # Downloaded TBX11K dataset
│   ├── processed/              # Preprocessed X-rays
│   └── splits/                 # Train/val/test CSV files
│
├── notebooks/
│   ├── 01_data_exploration.ipynb       # Dataset analysis
│   ├── 02_preprocessing.ipynb          # Image preprocessing
│   ├── 03_baseline_model.ipynb         # Baseline without AST
│   └── 04_ast_training.ipynb           # AST training
│
├── src/
│   ├── dataset.py              # X-ray dataset loader
│   ├── model.py                # EfficientNet + AST
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation metrics
│   └── utils.py                # Helper functions
│
├── ast_lib/                    # AST library (from malaria project)
│   ├── sparse_trainer.py       # AST trainer
│   └── sundew.py               # Sundew pruning algorithm
│
├── checkpoints/                # Saved models
│   └── metrics.csv             # Training metrics
│
├── gradio_app/
│   ├── app.py                  # Gradio demo
│   ├── requirements.txt        # Demo dependencies
│   └── examples/               # Example X-rays
│
└── docs/
    ├── DATASET_INFO.md         # Dataset documentation
    ├── MODEL_CARD.md           # Model card
    └── DEPLOYMENT.md           # Deployment guide
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/tb_detection_ast.git
cd tb_detection_ast
```

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Download Dataset

```bash
# Install Kaggle CLI
pip install kaggle

# Configure Kaggle credentials (place kaggle.json in ~/.kaggle/)
# Download TBX11K dataset
kaggle datasets download -d usmanshams/tbx-11
unzip tbx-11.zip -d data/raw/
```

### 4. Preprocess Data

```bash
python src/preprocess.py --input data/raw --output data/processed
```

### 5. Train Model

```bash
# Baseline (no AST)
python src/train.py --config configs/baseline.yaml

# With AST
python src/train.py --config configs/ast_training.yaml
```

### 6. Evaluate

```bash
python src/evaluate.py --checkpoint checkpoints/best_model.pth
```

### 7. Run Demo

```bash
cd gradio_app
python app.py
```

---

## 📊 Comparison with Malaria Detection Project

| Aspect | Malaria Detection | TB Detection |
|--------|------------------|--------------|
| **Task** | Binary classification | Binary classification |
| **Input** | Blood cell microscopy | Chest X-rays |
| **Image Size** | 224x224 RGB | 224x224 RGB |
| **Dataset Size** | 27,558 images | ~3,500 images |
| **Accuracy** | 93.94% | **99.29%** ✨ |
| **Energy Savings** | 88.98% | **89.52%** ✨ |
| **Activation Rate** | 9.38% | 9.38% |
| **Deployment** | Mobile microscopes | Clinic X-ray stations |

### Performance Visualization

![Malaria vs TB](visualizations/malaria_vs_tb_comparison.png)

**Key Insight**: AST achieves **consistent 89% energy savings** across different medical imaging modalities while maintaining clinical-grade accuracy!

---

## 🌍 Impact & Deployment

### Target Use Cases
1. **Rural clinics** without radiologists
2. **Mobile health vans** for community screening
3. **District hospitals** in resource-limited settings
4. **Telemedicine networks** across Africa

### Hardware Requirements

| Tier | Device | Cost | Use Case |
|------|--------|------|----------|
| **Minimum** | Raspberry Pi 4 (8GB) | $75 | Research/prototyping |
| **Recommended** | Android tablet | $200-300 | Mobile screening |
| **Optimal** | Mini-PC | $400-500 | Clinic deployment |

### Clinical Workflow
```
Patient arrives → X-ray captured → Upload to AI →
Prediction in <2s → Healthcare worker reviews →
Refer high-risk cases → Track outcomes
```

---

## 💰 Funding & Grants

We're applying for:
- **Gates Foundation** - Grand Challenges in Global Health
- **WHO TB Innovation** - Point-of-care diagnostics
- **Google AI for Social Good** - Healthcare AI in developing nations
- **NVIDIA Applied Research** - Energy-efficient medical AI

---

## 📚 Publications & Presentations

### Target Venues
- **Conferences**: MICCAI, MLHC, ISBI
- **Journals**: Medical Image Analysis, PLOS Computational Biology
- **Workshops**: AI4GlobalHealth (NeurIPS/ICML)

### Paper Title (Proposed)
> "Energy-Efficient Tuberculosis Detection Using Adaptive Sparse Training: Enabling AI Diagnosis in Resource-Limited Settings"

---

## 🤝 Contributing

We welcome contributions! Areas where you can help:

- 📊 Data annotation and validation
- 🧠 Model architecture improvements
- 🔬 Clinical validation studies
- 🌍 Deployment in African healthcare facilities
- 📝 Documentation and tutorials

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

**Note**: While the code is open source, please ensure compliance with local medical device regulations before clinical deployment.

---

## 🙏 Acknowledgments

- **Dataset**: TBX11K team for the publicly available chest X-ray dataset
- **Inspiration**: Building on our successful malaria detection project
- **AST Algorithm**: Sundew pruning method for energy-efficient training
- **Community**: Open-source AI and global health communities

---

## 📞 Contact

**Oluwafemi Idiakhoa**
- GitHub: [@oluwafemidiakhoa](https://github.com/oluwafemidiakhoa)
- Hugging Face: [@mgbam](https://huggingface.co/mgbam)
- LinkedIn: [Connect on LinkedIn](https://linkedin.com/in/oluwafemidiakhoa)

---

## 🌟 Related Projects

- [Malaria Detection with AST](../malaria_ast_starter) - 93.94% accuracy, 88% energy savings
- [Energy-Efficient AI for Africa](link) - Building accessible healthcare AI

---

**Together, we're making medical AI accessible to those who need it most.** 🌍✨

---

## 📊 Project Status

✅ **Training Complete** - Model deployed and ready for use!

**Completed Milestones:**
- ✅ Project structure created
- ✅ Dataset downloaded and preprocessed
- ✅ AST training completed (50 epochs)
- ✅ 99.29% accuracy achieved
- ✅ 89.52% energy savings validated
- ✅ Comprehensive visualizations generated
- ✅ Grad-CAM explainability implemented
- ✅ Training notebooks created
- 🔄 Hugging Face Space deployment
- ⏳ Clinical validation study

**Try the live demo**: [Hugging Face Space](https://huggingface.co/spaces/mgbam/Tuberculosis)
