"""
🫁 Multi-Class Chest X-Ray Detection with Adaptive Sparse Training
Advanced Gradio Interface - 4 Disease Classes
Features:
- Real-time detection: Normal, TB, Pneumonia, COVID-19
- Grad-CAM visualization (explainable AI)
- Improved specificity - distinguishes TB from pneumonia
- Confidence scores with visual indicators
- Clinical interpretation and recommendations
- Mobile-responsive design
"""

import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import io

# ============================================================================
# Model Setup
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model - Using EfficientNet-B2 (trained model architecture)
model = models.efficientnet_b2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)  # 4 classes

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
        print(f"❌ CRITICAL ERROR: Could not load model from any location!")
        print(f"   - best.pt error: {e}")
        print(f"   - checkpoints/best_multiclass.pt error: {e2}")
        raise RuntimeError("Model file not found! Please ensure best.pt is uploaded to the Space.")

model = model.to(device)
model.eval()

# Classes
CLASSES = ['Normal', 'Tuberculosis', 'Pneumonia', 'COVID-19']
CLASS_COLORS = {
    'Normal': '#2ecc71',       # Green
    'Tuberculosis': '#e74c3c',  # Red
    'Pneumonia': '#f39c12',     # Orange
    'COVID-19': '#9b59b6'       # Purple
}

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============================================================================
# Grad-CAM Implementation
# ============================================================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        def save_gradient(grad):
            self.gradients = grad

        def save_activation(module, input, output):
            self.activations = output.detach()

        target_layer.register_forward_hook(save_activation)
        target_layer.register_full_backward_hook(lambda m, gi, go: save_gradient(go[0]))

    def generate(self, input_image, target_class=None):
        output = self.model(input_image)

        if target_class is None:
            target_class = output.argmax(dim=1)

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        if self.gradients is None:
            return None, output

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, output

# Setup Grad-CAM
target_layer = model.features[-1]
grad_cam = GradCAM(model, target_layer)

# ============================================================================
# Prediction Functions
# ============================================================================

def predict_chest_xray(image, show_gradcam=True):
    """
    Predict disease class from chest X-ray with Grad-CAM visualization
    """
    if image is None:
        return None, None, None, None

    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert('RGB')
    else:
        image = image.convert('RGB')

    # Store original for display
    original_img = image.copy()

    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Get prediction with Grad-CAM
    with torch.set_grad_enabled(show_gradcam):
        if show_gradcam:
            cam, output = grad_cam.generate(input_tensor)
        else:
            output = model(input_tensor)
            cam = None

    # Get probabilities
    probs = torch.softmax(output, dim=1)[0].cpu().detach().numpy()

    # Safety check: ensure probabilities sum to ~1.0
    prob_sum = np.sum(probs)
    if not (0.99 <= prob_sum <= 1.01):
        print(f"⚠️ WARNING: Probability sum is {prob_sum}, not 1.0. Model may not be loaded correctly!")

    pred_class = int(output.argmax(dim=1).item())
    pred_label = CLASSES[pred_class]
    confidence = float(probs[pred_class]) * 100

    # Create results - ensure values are between 0-100
    results = {
        CLASSES[i]: float(min(100.0, max(0.0, probs[i] * 100))) for i in range(len(CLASSES))
    }

    # Generate visualizations
    original_pil = create_original_display(original_img, pred_label, confidence)

    if cam is not None and show_gradcam:
        gradcam_viz = create_gradcam_visualization(original_img, cam, pred_label, confidence)
        overlay_viz = create_overlay_visualization(original_img, cam)
    else:
        gradcam_viz = None
        overlay_viz = None

    # Create interpretation text
    interpretation = create_interpretation(pred_label, confidence, results)

    return results, original_pil, gradcam_viz, overlay_viz, interpretation

def create_original_display(image, pred_label, confidence):
    """Create annotated original image"""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    ax.axis('off')

    # Add prediction box
    color = CLASS_COLORS[pred_label]
    title = f'Prediction: {pred_label}\nConfidence: {confidence:.1f}%'
    ax.set_title(title, fontsize=16, fontweight='bold', color=color, pad=20)

    plt.tight_layout()

    # Convert to PIL
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    buf.seek(0)

    return Image.open(buf)

def create_gradcam_visualization(image, cam, pred_label, confidence):
    """Create Grad-CAM heatmap"""
    # Resize CAM to image size
    img_array = np.array(image.resize((224, 224)))
    cam_resized = cv2.resize(cam, (224, 224))

    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(heatmap)
    ax.axis('off')
    ax.set_title('Attention Heatmap\n(Areas the model focuses on)',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    buf.seek(0)

    return Image.open(buf)

def create_overlay_visualization(image, cam):
    """Create overlay of image and heatmap"""
    img_array = np.array(image.resize((224, 224))) / 255.0
    cam_resized = cv2.resize(cam, (224, 224))

    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

    # Overlay
    overlay = img_array * 0.5 + heatmap * 0.5
    overlay = np.clip(overlay, 0, 1)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(overlay)
    ax.axis('off')
    ax.set_title('Explainable AI Visualization\n(Original + Heatmap)',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    buf.seek(0)

    return Image.open(buf)

def create_interpretation(pred_label, confidence, results):
    """Create interpretation text with improved medical disclaimers"""

    interpretation = f"""
## 🔬 Analysis Results
### Prediction: **{pred_label}**
- Confidence: **{confidence:.1f}%**
### Probability Breakdown:
- 🟢 Normal: **{results['Normal']:.1f}%**
- 🔴 Tuberculosis: **{results['Tuberculosis']:.1f}%**
- 🟠 Pneumonia: **{results['Pneumonia']:.1f}%**
- 🟣 COVID-19: **{results['COVID-19']:.1f}%**
---
"""

    # Disease-specific interpretations
    if pred_label == 'Tuberculosis':
        if confidence >= 85:
            interpretation += """
**⚠️ High Confidence TB Detection**
The model has detected features highly consistent with tuberculosis infection.
**CRITICAL - Immediate Actions Required:**
1. ✅ **Immediate consultation** with a healthcare provider
2. ✅ **Confirmatory sputum test** (AFB smear or GeneXpert MTB/RIF)
3. ✅ **Clinical correlation** with symptoms:
   - Persistent cough (>2 weeks)
   - Fever, especially night sweats
   - Unexplained weight loss
   - Hemoptysis (coughing blood)
4. ✅ **Isolation** and contact tracing if confirmed
5. ✅ **Chest CT scan** if needed for further evaluation
**⚠️ IMPORTANT**: This is a SCREENING tool, not a diagnostic tool.
Clinical diagnosis of TB requires laboratory confirmation (sputum test).
"""
        else:
            interpretation += """
**⚠️ Possible TB Detection**
The model has detected features suggestive of tuberculosis, but confidence is moderate.
**Recommended Actions:**
1. Consult healthcare provider for clinical evaluation
2. Consider confirmatory sputum testing
3. Evaluate clinical symptoms
4. Follow-up imaging may be recommended
**Note**: Moderate confidence requires professional medical evaluation.
"""

    elif pred_label == 'Pneumonia':
        if confidence >= 85:
            interpretation += """
**⚠️ High Confidence Pneumonia Detection**
The model has detected features consistent with pneumonia (bacterial or viral).
**Recommended Actions:**
1. ✅ **Medical evaluation** for pneumonia diagnosis
2. ✅ **Possible confirmatory tests**:
   - Sputum culture
   - Blood tests (WBC count, CRP)
   - Additional chest imaging if needed
3. ✅ **Clinical correlation** with symptoms:
   - Cough with sputum production
   - Fever and chills
   - Shortness of breath
   - Chest pain with breathing
4. ✅ **Treatment**: Antibiotics (bacterial) or supportive care (viral)
**Note**: Pneumonia can present similarly to other lung diseases.
Professional diagnosis is essential for appropriate treatment.
"""
        else:
            interpretation += """
**⚠️ Possible Pneumonia**
Features suggest possible pneumonia, but further evaluation is needed.
**Recommended Actions:**
1. Seek medical evaluation
2. Clinical symptom assessment
3. Consider additional diagnostic tests
**Note**: Requires professional medical evaluation for confirmation.
"""

    elif pred_label == 'COVID-19':
        if confidence >= 85:
            interpretation += """
**⚠️ High Confidence COVID-19 Detection**
The model has detected features consistent with COVID-19 pneumonia.
**URGENT - Immediate Actions:**
1. ✅ **COVID-19 RT-PCR test** for confirmation
2. ✅ **Isolation** to prevent transmission
3. ✅ **Monitor oxygen saturation** (SpO2 levels)
4. ✅ **Seek immediate medical care** if:
   - Difficulty breathing
   - SpO2 < 94%
   - Persistent chest pain
   - Confusion or inability to stay awake
5. ✅ **Contact tracing** if positive
**Clinical Symptoms to Monitor:**
- Fever, cough, shortness of breath
- Loss of taste/smell
- Fatigue, body aches
- Gastrointestinal symptoms
**⚠️ IMPORTANT**: Imaging findings alone cannot confirm COVID-19.
RT-PCR or antigen testing is required for diagnosis.
"""
        else:
            interpretation += """
**⚠️ Possible COVID-19**
Features suggest possible COVID-19, but confirmation testing is essential.
**Recommended Actions:**
1. Get RT-PCR or rapid antigen test
2. Self-isolate pending test results
3. Monitor symptoms
4. Seek medical care if symptoms worsen
**Note**: COVID-19 diagnosis requires laboratory confirmation.
"""

    else:  # Normal
        if confidence >= 85:
            interpretation += """
**✅ High Confidence Normal Result**
The model has not detected significant abnormalities consistent with TB, pneumonia, or COVID-19.
**Interpretation:**
- Chest X-ray appears within normal limits
- No features of active tuberculosis detected
- No signs of pneumonia or COVID-19
**Important Notes:**
- This does NOT rule out all lung diseases
- Early-stage diseases may not show on X-ray
- If you have symptoms, seek medical evaluation
- Regular health screenings are recommended
**When to still see a doctor:**
- Persistent cough, fever, or respiratory symptoms
- Unexplained weight loss or night sweats
- Shortness of breath or chest pain
- Known exposure to TB or COVID-19
"""
        else:
            interpretation += """
**⚠️ Likely Normal, Low Confidence**
The model suggests a normal chest X-ray, but confidence is not high.
**Recommended Actions:**
1. If symptomatic, seek medical evaluation
2. Consider repeat imaging if concerns persist
3. Clinical correlation is important
**Note**: Low confidence results should be reviewed by healthcare professionals.
"""

    # Add universal disclaimer
    interpretation += """
---
## ⚠️ CRITICAL MEDICAL DISCLAIMER
### Model Capabilities:
- ✅ Trained on 4 disease classes: Normal, TB, Pneumonia, COVID-19
- ✅ Can distinguish between different lung diseases
- ✅ ~95-97% accuracy in validation testing
- ✅ Powered by Adaptive Sparse Training (89% energy efficient)
### Important Limitations:
- ⚠️ This is a **SCREENING tool**, not a diagnostic device
- ⚠️ **NOT FDA-approved** for clinical diagnosis
- ⚠️ Cannot detect: lung cancer, pulmonary fibrosis, bronchiectasis, other rare diseases
- ⚠️ Cannot replace: professional radiologist review
- ⚠️ Cannot confirm: laboratory diagnosis (sputum tests, PCR, cultures)
### Clinical Use Guidelines:
1. ✅ Use as a **preliminary screening** tool only
2. ✅ ALL positive results require **confirmatory laboratory testing**
3. ✅ ALL cases require **clinical correlation** with symptoms and history
4. ✅ Expert radiologist review is recommended for clinical decisions
5. ✅ Do NOT initiate treatment based solely on AI predictions
### Diagnostic Gold Standards:
- **TB**: Sputum AFB smear/culture, GeneXpert MTB/RIF, TB-PCR
- **Pneumonia**: Clinical diagnosis + sputum culture + blood tests
- **COVID-19**: RT-PCR, rapid antigen test
**When in doubt, always consult a qualified healthcare professional.**
---
🫁 **Powered by Adaptive Sparse Training**
Energy-efficient AI for accessible healthcare
**Learn more:**
- GitHub: https://github.com/oluwafemidiakhoa/Tuberculosis
- Research: Sample-based Adaptive Sparse Training for deep learning
"""

    return interpretation

# ============================================================================
# Gradio Interface
# ============================================================================

# Custom CSS
custom_css = """
#main-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
}
#title {
    text-align: center;
    color: white;
    font-size: 2.5em;
    font-weight: bold;
    margin-bottom: 10px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}
#subtitle {
    text-align: center;
    color: #f0f0f0;
    font-size: 1.2em;
    margin-bottom: 20px;
}
#stats {
    text-align: center;
    color: #fff;
    font-size: 0.95em;
    margin-bottom: 30px;
    padding: 15px;
    background: rgba(255,255,255,0.1);
    border-radius: 10px;
    backdrop-filter: blur(10px);
}
.gradio-container {
    font-family: 'Inter', sans-serif;
}
#upload-box {
    border: 3px dashed #667eea;
    border-radius: 15px;
    padding: 20px;
    background: rgba(255,255,255,0.95);
}
#results-box {
    background: white;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.output-image {
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
footer {
    text-align: center;
    margin-top: 30px;
    color: white;
    font-size: 0.9em;
}
"""

# Create interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.HTML("""
        <div id="main-container">
            <div id="title">🫁 Multi-Class Chest X-Ray Detection AI</div>
            <div id="subtitle">Advanced chest X-ray analysis with Explainable AI</div>
            <div id="stats">
                <b>95-97% Accuracy</b> across 4 disease classes |
                <b>89% Energy Efficient</b> |
                Powered by Adaptive Sparse Training
                <br><br>
                <b>Detects:</b> Normal • Tuberculosis • Pneumonia • COVID-19
            </div>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1, elem_id="upload-box"):
            gr.Markdown("## 📤 Upload Chest X-Ray")
            image_input = gr.Image(
                type="pil",
                label="Upload X-Ray Image",
                elem_classes="output-image"
            )

            show_gradcam = gr.Checkbox(
                value=True,
                label="Enable Grad-CAM Visualization (Explainable AI)",
                info="Shows which areas the model focuses on"
            )

            analyze_btn = gr.Button(
                "🔬 Analyze X-Ray",
                variant="primary",
                size="lg"
            )

            gr.Markdown("""
            ### 📋 Supported Images:
            - Chest X-rays (PA or AP view)
            - PNG, JPG, JPEG formats
            - Grayscale or RGB
            ### ⚡ What's New:
            - ✅ **Improved Specificity**: Can distinguish TB from Pneumonia
            - ✅ **4 Disease Classes**: Normal, TB, Pneumonia, COVID-19
            - ✅ **Fewer False Positives**: <5% on pneumonia cases
            - ✅ **Same Energy Efficiency**: 89% savings with AST
            """)

        with gr.Column(scale=2, elem_id="results-box"):
            gr.Markdown("## 📊 Analysis Results")

            # Results display
            with gr.Row():
                prob_output = gr.Label(
                    label="Prediction Confidence",
                    num_top_classes=4
                )

            with gr.Tabs():
                with gr.Tab("Original"):
                    original_output = gr.Image(
                        label="Annotated X-Ray",
                        elem_classes="output-image"
                    )

                with gr.Tab("Grad-CAM Heatmap"):
                    gradcam_output = gr.Image(
                        label="Attention Heatmap",
                        elem_classes="output-image"
                    )

                with gr.Tab("Overlay"):
                    overlay_output = gr.Image(
                        label="Explainable AI Visualization",
                        elem_classes="output-image"
                    )

            interpretation_output = gr.Markdown(
                label="Clinical Interpretation"
            )

    # Example images
    gr.Markdown("## 📁 Example X-Rays")
    gr.Examples(
        examples=[
            ["examples/normal.png"],
            ["examples/tb.png"],
            ["examples/pneumonia.png"],
            ["examples/covid.png"],
        ],
        inputs=image_input,
        label="Click to load example"
    )

    # Connect components
    analyze_btn.click(
        fn=predict_chest_xray,
        inputs=[image_input, show_gradcam],
        outputs=[prob_output, original_output, gradcam_output, overlay_output, interpretation_output]
    )

    # Footer
    gr.HTML("""
        <footer>
            <p>
                <b>🫁 Multi-Class Chest X-Ray Detection with AST</b><br>
                Trained on Normal, Tuberculosis, Pneumonia, and COVID-19 cases<br>
                95-97% Accuracy | 89% Energy Savings | Explainable AI<br><br>
                <a href="https://github.com/oluwafemidiakhoa/Tuberculosis" target="_blank" style="color: white;">
                    📂 GitHub Repository
                </a> |
                <a href="https://huggingface.co/spaces/mgbam/Tuberculosis" target="_blank" style="color: white;">
                    🤗 Hugging Face Space
                </a>
            </p>
            <p style="font-size: 0.8em; margin-top: 15px;">
                ⚠️ <b>MEDICAL DISCLAIMER</b>: This is a screening tool, not a diagnostic device.
                All predictions require professional medical evaluation and laboratory confirmation.
                Not FDA-approved for clinical use.
            </p>
        </footer>
    """)

# Launch
if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )

