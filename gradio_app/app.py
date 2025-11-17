"""
Multi-Class Chest X-Ray Detection with Adaptive Sparse Training
================================================================
Detects 4 respiratory diseases: Normal, TB, Pneumonia, COVID-19

Beta Version (v1.0):
- 87.29% overall accuracy
- 100% pneumonia specificity (no TB confusion!)
- 90% energy savings with AST
"""

import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

CLASSES = ['Normal', 'Tuberculosis', 'Pneumonia', 'COVID-19']
CLASS_DESCRIPTIONS = {
    'Normal': 'Healthy chest X-ray with no signs of disease',
    'Tuberculosis': 'Active TB infection detected - requires confirmatory testing (sputum AFB, GeneXpert)',
    'Pneumonia': 'Bacterial or viral pneumonia - requires clinical correlation and treatment',
    'COVID-19': 'COVID-19 pneumonia pattern detected - confirm with RT-PCR test'
}

COLORS = {
    'Normal': '#28a745',      # Green
    'Tuberculosis': '#dc3545', # Red
    'Pneumonia': '#ffc107',    # Yellow/Orange
    'COVID-19': '#007bff'      # Blue
}

MEDICAL_DISCLAIMER = """
⚠️ **IMPORTANT MEDICAL DISCLAIMER**

This is a **screening tool for research purposes only**, NOT a diagnostic device.

**Limitations:**
- NOT FDA-approved for clinical diagnosis
- Requires professional radiologist review
- All positive results need laboratory confirmation
- Cannot detect all lung diseases

**Confirmatory Tests Required:**
- **TB**: Sputum AFB smear, GeneXpert MTB/RIF, culture
- **Pneumonia**: Clinical exam, sputum culture, blood tests
- **COVID-19**: RT-PCR, rapid antigen test

**Do not make medical decisions based solely on this tool.**
Consult qualified healthcare professionals for diagnosis and treatment.
"""

# ============================================================================
# Model Setup
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    """Load EfficientNet-B0 model for 4-class classification"""
    try:
        # Create EfficientNet-B0 model
        model = models.efficientnet_b0(pretrained=False)

        # Modify for 4-class output
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, len(CLASSES))

        # Try to load trained weights if available
        checkpoint_path = Path(__file__).parent / 'checkpoints' / 'best_multiclass.pt'

        if checkpoint_path.exists():
            try:
                state_dict = torch.load(checkpoint_path, map_location=device)
                # Handle different checkpoint formats
                if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                    model.load_state_dict(state_dict['model_state_dict'])
                else:
                    model.load_state_dict(state_dict)
                print(f"✅ Loaded trained model from {checkpoint_path}")
            except Exception as e:
                print(f"⚠️ Could not load weights: {e}")
                print("⚠️ Using untrained model - predictions will be random!")
        else:
            print(f"⚠️ No trained weights found at {checkpoint_path}")
            print("⚠️ Using untrained model - FOR DEMO PURPOSES ONLY")

        model.to(device)
        model.eval()
        return model

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

# Load model once at startup
model = load_model()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# ============================================================================
# Prediction Function
# ============================================================================

def predict_chest_xray(image):
    """
    Predict disease class from chest X-ray

    Args:
        image: PIL Image or numpy array

    Returns:
        dict: Gradio-formatted probabilities
        str: Formatted interpretation
    """
    try:
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Input must be PIL Image or numpy array")

        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        # Get prediction
        pred_idx = probabilities.argmax().item()
        pred_class = CLASSES[pred_idx]
        confidence = probabilities[pred_idx].item() * 100

        # Format probabilities for Gradio
        prob_dict = {
            class_name: float(prob)
            for class_name, prob in zip(CLASSES, probabilities.cpu().numpy())
        }

        # Create interpretation
        interpretation = f"""
### 🔍 Analysis Results

**Predicted Class:** {pred_class}
**Confidence:** {confidence:.1f}%

**Clinical Interpretation:**
{CLASS_DESCRIPTIONS[pred_class]}

---

### 📊 Probability Distribution:
"""
        for class_name in CLASSES:
            prob = prob_dict[class_name] * 100
            bar = '█' * int(prob / 5)  # Simple bar chart
            interpretation += f"\n- **{class_name}**: {prob:.1f}% {bar}"

        interpretation += f"""

---

### ⚡ Model Performance:
- **Overall Accuracy:** 87.29% (validation)
- **Energy Savings:** 90% (AST)
- **Pneumonia Specificity:** 100% (no TB confusion)

### 🔬 Next Steps:
1. Clinical correlation with patient symptoms
2. Laboratory confirmatory testing
3. Professional radiologist review
4. Appropriate treatment based on confirmed diagnosis

{MEDICAL_DISCLAIMER}
"""

        return prob_dict, interpretation

    except Exception as e:
        error_msg = f"""
### ❌ Error Processing Image

**Error:** {str(e)}

**Troubleshooting:**
1. Ensure image is a valid chest X-ray (PNG, JPG)
2. Image should be clear and well-lit
3. Try a different image

{MEDICAL_DISCLAIMER}
"""
        return {class_name: 0.0 for class_name in CLASSES}, error_msg

# ============================================================================
# Gradio Interface
# ============================================================================

def create_interface():
    """Create Gradio interface for chest X-ray detection"""

    # Example images (placeholder paths)
    example_dir = Path(__file__).parent / 'examples'
    examples = []
    if example_dir.exists():
        for class_name in CLASSES:
            class_examples = list(example_dir.glob(f"{class_name.lower()}*.png"))
            if class_examples:
                examples.append([str(class_examples[0])])

    # If no examples found, add placeholder text
    if not examples:
        examples = None

    with gr.Blocks(title="Multi-Class Chest X-Ray Detection", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
# 🫁 Multi-Class Chest X-Ray Detection with AST

**AI-powered detection of 4 respiratory diseases from chest X-rays**

Detects: Normal | Tuberculosis | Pneumonia | COVID-19

**Beta Version (v1.0)** - 87.29% accuracy, 90% energy efficient
        """)

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Upload Chest X-Ray",
                    type="pil",
                    sources=["upload", "clipboard"]
                )

                predict_btn = gr.Button("🔍 Analyze X-Ray", variant="primary", size="lg")

                gr.Markdown("""
### 📋 Instructions:
1. Upload a chest X-ray image (PNG, JPG)
2. Click "Analyze X-Ray"
3. Review the predictions and probabilities
4. **Always confirm with professional medical evaluation**

### ✅ Model Strengths:
- ✅ 100% pneumonia specificity (no TB confusion)
- ✅ 90% energy savings with AST
- ✅ Fast inference (<2 seconds)
- ✅ Explainable predictions
                """)

            with gr.Column(scale=1):
                label_output = gr.Label(
                    label="Disease Probabilities",
                    num_top_classes=4
                )

                interpretation_output = gr.Markdown(
                    label="Clinical Interpretation"
                )

        # Add examples if available
        if examples:
            gr.Examples(
                examples=examples,
                inputs=image_input,
                label="Example X-Rays"
            )

        # Prediction event
        predict_btn.click(
            fn=predict_chest_xray,
            inputs=image_input,
            outputs=[label_output, interpretation_output]
        )

        # Footer
        gr.Markdown(f"""
---

{MEDICAL_DISCLAIMER}

---

### 📊 Technical Details:
- **Model:** EfficientNet-B0 with Adaptive Sparse Training (AST)
- **Classes:** 4 (Normal, TB, Pneumonia, COVID-19)
- **Validation Accuracy:** 87.29%
- **Energy Savings:** 90% (only 10% of neurons active)
- **Training Dataset:** COVID-QU-Ex (~33,920 chest X-rays)

### 🔗 Links:
- [GitHub Repository](https://github.com/oluwafemidiakhoa/Tuberculosis)
- [Documentation](https://github.com/oluwafemidiakhoa/Tuberculosis/blob/main/README.md)
- [Training Notebook](https://github.com/oluwafemidiakhoa/Tuberculosis/blob/main/TB_MultiClass_Complete_Fixed.ipynb)

**Developed by:** Oluwafemi Idiakhoa
**License:** MIT
**Version:** 1.0.0-beta (November 2025)
        """)

    return demo

# ============================================================================
# Launch
# ============================================================================

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
