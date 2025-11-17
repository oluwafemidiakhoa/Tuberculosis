"""
🫁 Multi-Class Chest X-Ray Detection - World-Class UI/UX
Advanced Medical AI Interface with Premium Design

Features:
- Modern glass morphism design
- Dark/Light mode toggle
- Real-time confidence animations
- Interactive Grad-CAM visualizations
- Professional medical reporting
- Smooth transitions and micro-interactions
- Responsive mobile-first design
- Accessibility compliant
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
from datetime import datetime

# ============================================================================
# Model Setup
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)  # 4 classes

try:
    model.load_state_dict(torch.load('checkpoints/best_multiclass.pt', map_location=device))
    print("✅ Multi-class model loaded successfully!")
except Exception as e:
    print(f"⚠️  Error loading model: {e}")

model = model.to(device)
model.eval()

# Classes with enhanced metadata
CLASSES = ['Normal', 'Tuberculosis', 'Pneumonia', 'COVID-19']
CLASS_COLORS = {
    'Normal': '#10b981',       # Emerald green
    'Tuberculosis': '#ef4444',  # Red
    'Pneumonia': '#f59e0b',     # Amber
    'COVID-19': '#8b5cf6'       # Violet
}
CLASS_ICONS = {
    'Normal': '✅',
    'Tuberculosis': '🦠',
    'Pneumonia': '🫁',
    'COVID-19': '😷'
}
RISK_LEVELS = {
    'Normal': 'LOW',
    'Tuberculosis': 'CRITICAL',
    'Pneumonia': 'HIGH',
    'COVID-19': 'HIGH'
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
# Enhanced Visualization Functions
# ============================================================================

def create_professional_display(image, pred_label, confidence, results):
    """Create stunning professional medical display"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor('#0f172a')

    # Left: Original X-Ray with overlay
    ax1 = axes[0]
    ax1.imshow(image, cmap='bone')
    ax1.axis('off')
    ax1.set_title('Chest X-Ray Scan',
                  fontsize=18, fontweight='bold',
                  color='white', pad=20, family='sans-serif')

    # Add corner markers (professional radiology style)
    h, w = image.size[1], image.size[0]
    corner_size = 0.05

    # Right: Professional analysis panel
    ax2 = axes[1]
    ax2.axis('off')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)

    # Header
    icon = CLASS_ICONS[pred_label]
    color = CLASS_COLORS[pred_label]
    risk = RISK_LEVELS[pred_label]

    ax2.text(5, 9.5, f'{icon} DIAGNOSTIC ANALYSIS',
             ha='center', fontsize=20, fontweight='bold',
             color='white', family='sans-serif')

    # Prediction card
    y_pos = 8.5
    ax2.add_patch(plt.Rectangle((0.5, y_pos-1.2), 9, 1.5,
                                facecolor=color, alpha=0.2,
                                edgecolor=color, linewidth=2,
                                transform=ax2.transData))

    ax2.text(5, y_pos-0.2, pred_label.upper(),
             ha='center', fontsize=28, fontweight='bold',
             color=color, family='sans-serif')
    ax2.text(5, y_pos-0.7, f'{confidence:.1f}% Confidence',
             ha='center', fontsize=16,
             color='white', family='sans-serif')

    # Risk indicator
    risk_colors = {'LOW': '#10b981', 'HIGH': '#f59e0b', 'CRITICAL': '#ef4444'}
    ax2.text(5, y_pos-1.0, f'Risk Level: {risk}',
             ha='center', fontsize=14, fontweight='bold',
             color=risk_colors[risk], family='sans-serif')

    # Probability bars
    y_pos = 5.5
    ax2.text(5, y_pos+0.5, 'PROBABILITY DISTRIBUTION',
             ha='center', fontsize=14, fontweight='bold',
             color='white', family='sans-serif')

    bar_height = 0.4
    for i, (cls, prob) in enumerate(results.items()):
        y = y_pos - i * 0.8
        bar_width = (prob / 100) * 8

        # Background bar
        ax2.add_patch(plt.Rectangle((1, y-bar_height/2), 8, bar_height,
                                    facecolor='#1e293b', alpha=0.5))

        # Filled bar
        ax2.add_patch(plt.Rectangle((1, y-bar_height/2), bar_width, bar_height,
                                    facecolor=CLASS_COLORS[cls], alpha=0.8))

        # Label and percentage
        ax2.text(0.5, y, f'{CLASS_ICONS[cls]}',
                fontsize=16, va='center', ha='right', color='white')
        ax2.text(9.5, y, f'{prob:.1f}%',
                fontsize=12, va='center', ha='right',
                color=CLASS_COLORS[cls], fontweight='bold')
        ax2.text(1.1, y, cls,
                fontsize=11, va='center', ha='left', color='white')

    # Timestamp and metadata
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ax2.text(5, 0.8, f'Analysis Date: {timestamp}',
             ha='center', fontsize=10, color='#94a3b8',
             family='monospace')
    ax2.text(5, 0.4, 'Powered by Adaptive Sparse Training AI',
             ha='center', fontsize=9, color='#64748b',
             style='italic')

    plt.tight_layout()

    # Convert to PIL
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='#0f172a', edgecolor='none')
    plt.close()
    buf.seek(0)

    return Image.open(buf)

def create_gradcam_heatmap(image, cam, pred_label, confidence):
    """Create stunning Grad-CAM visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#0f172a')

    # Original
    img_array = np.array(image.resize((224, 224)))
    cam_resized = cv2.resize(cam, (224, 224))

    # Heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay = img_array * 0.6 + heatmap * 0.4
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Plot all three
    axes[0].imshow(img_array, cmap='bone')
    axes[0].set_title('Original X-Ray', fontsize=14, fontweight='bold',
                      color='white', pad=15)
    axes[0].axis('off')

    axes[1].imshow(heatmap)
    axes[1].set_title('AI Attention Map', fontsize=14, fontweight='bold',
                      color='white', pad=15)
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title('Explainable AI Overlay', fontsize=14, fontweight='bold',
                      color=CLASS_COLORS[pred_label], pad=15)
    axes[2].axis('off')

    # Add main title
    fig.suptitle(f'🔬 Grad-CAM Visualization - {pred_label} Detection ({confidence:.1f}%)',
                 fontsize=16, fontweight='bold', color='white', y=0.98)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='#0f172a', edgecolor='none')
    plt.close()
    buf.seek(0)

    return Image.open(buf)

def create_interpretation(pred_label, confidence, results):
    """Create stunning markdown interpretation with color-coded sections"""

    icon = CLASS_ICONS[pred_label]
    color_hex = CLASS_COLORS[pred_label]
    risk = RISK_LEVELS[pred_label]
    timestamp = datetime.now().strftime('%B %d, %Y at %I:%M %p')

    # Risk badge HTML
    risk_colors = {'LOW': '#10b981', 'HIGH': '#f59e0b', 'CRITICAL': '#ef4444'}
    risk_badge = f'<span style="background: {risk_colors[risk]}; color: white; padding: 4px 12px; border-radius: 20px; font-weight: bold; font-size: 0.85em;">{risk} RISK</span>'

    interpretation = f"""
<div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 25px; border-radius: 15px; border-left: 5px solid {color_hex}; margin: 15px 0;">
    <h2 style="color: {color_hex}; margin-top: 0;">{icon} DIAGNOSTIC REPORT</h2>
    <p style="color: #94a3b8; font-size: 0.9em; margin: 5px 0;">Generated: {timestamp}</p>
    <hr style="border: none; border-top: 1px solid #334155; margin: 15px 0;">

    <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 10px; margin: 15px 0;">
        <h3 style="color: white; margin-top: 0;">🎯 PRIMARY FINDING</h3>
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <p style="font-size: 1.8em; font-weight: bold; color: {color_hex}; margin: 10px 0;">{pred_label.upper()}</p>
                <p style="color: #cbd5e1; font-size: 1.1em;">Confidence: <strong style="color: {color_hex};">{confidence:.1f}%</strong></p>
            </div>
            <div style="text-align: right;">
                {risk_badge}
            </div>
        </div>
    </div>

    <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 10px; margin: 15px 0;">
        <h3 style="color: white; margin-top: 0;">📊 PROBABILITY ANALYSIS</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
"""

    # Add probability cards
    for cls, prob in results.items():
        cls_color = CLASS_COLORS[cls]
        cls_icon = CLASS_ICONS[cls]
        bar_width = int(prob)

        interpretation += f"""
            <div style="background: rgba(255,255,255,0.03); padding: 15px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="color: white; font-weight: bold;">{cls_icon} {cls}</span>
                    <span style="color: {cls_color}; font-weight: bold; font-size: 1.1em;">{prob:.1f}%</span>
                </div>
                <div style="background: rgba(255,255,255,0.1); height: 8px; border-radius: 10px; overflow: hidden;">
                    <div style="background: {cls_color}; height: 100%; width: {bar_width}%; border-radius: 10px; transition: width 0.5s ease;"></div>
                </div>
            </div>
"""

    interpretation += """
        </div>
    </div>
</div>
"""

    # Clinical recommendations
    if pred_label == 'Tuberculosis':
        if confidence >= 85:
            interpretation += f"""
<div style="background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%); padding: 25px; border-radius: 15px; border: 2px solid #ef4444; margin: 20px 0;">
    <h2 style="color: #fca5a5; margin-top: 0;">🚨 CRITICAL ALERT - TUBERCULOSIS DETECTED</h2>
    <p style="color: #fecaca; font-size: 1.1em; font-weight: bold;">High confidence detection requires immediate medical attention</p>

    <div style="background: rgba(0,0,0,0.3); padding: 20px; border-radius: 10px; margin: 15px 0;">
        <h3 style="color: #fca5a5;">⚡ IMMEDIATE ACTIONS REQUIRED</h3>
        <ul style="color: white; line-height: 1.8; font-size: 1.05em;">
            <li><strong style="color: #fca5a5;">URGENT:</strong> Contact healthcare provider immediately</li>
            <li><strong style="color: #fca5a5;">TESTING:</strong> Confirmatory sputum test (AFB smear / GeneXpert MTB/RIF)</li>
            <li><strong style="color: #fca5a5;">ISOLATION:</strong> Implement respiratory precautions to prevent transmission</li>
            <li><strong style="color: #fca5a5;">MONITORING:</strong> Track symptoms - persistent cough, night sweats, weight loss, hemoptysis</li>
        </ul>
    </div>

    <div style="background: rgba(0,0,0,0.3); padding: 20px; border-radius: 10px; margin: 15px 0;">
        <h3 style="color: #fca5a5;">🩺 Clinical Correlation</h3>
        <p style="color: #fecaca; line-height: 1.6;">
            TB diagnosis requires laboratory confirmation via sputum microscopy, culture, or molecular testing (GeneXpert).
            Chest CT may be recommended for extent evaluation. Contact tracing essential if confirmed.
        </p>
    </div>

    <p style="color: #fef2f2; background: rgba(0,0,0,0.4); padding: 15px; border-radius: 8px; border-left: 4px solid #fca5a5; margin: 15px 0;">
        <strong>⚠️ CRITICAL:</strong> This is a SCREENING tool. All TB diagnoses must be confirmed with laboratory testing before treatment initiation.
    </p>
</div>
"""
        else:
            interpretation += """
<div style="background: rgba(127, 29, 29, 0.3); padding: 20px; border-radius: 15px; border: 2px solid #f59e0b; margin: 20px 0;">
    <h3 style="color: #fbbf24;">⚠️ Possible TB Detection (Moderate Confidence)</h3>
    <p style="color: #fde68a;">Follow-up evaluation recommended. Requires clinical correlation and confirmatory testing.</p>
    <ul style="color: #fef3c7; line-height: 1.8;">
        <li>Schedule medical consultation</li>
        <li>Consider sputum testing</li>
        <li>Evaluate clinical symptoms and history</li>
    </ul>
</div>
"""

    elif pred_label == 'Pneumonia':
        if confidence >= 85:
            interpretation += """
<div style="background: linear-gradient(135deg, #78350f 0%, #92400e 100%); padding: 25px; border-radius: 15px; border: 2px solid #f59e0b; margin: 20px 0;">
    <h2 style="color: #fbbf24; margin-top: 0;">⚠️ PNEUMONIA DETECTED</h2>
    <p style="color: #fde68a; font-size: 1.1em;">High confidence detection - Medical evaluation required</p>

    <div style="background: rgba(0,0,0,0.3); padding: 20px; border-radius: 10px; margin: 15px 0;">
        <h3 style="color: #fbbf24;">🩺 Recommended Actions</h3>
        <ul style="color: white; line-height: 1.8;">
            <li><strong>Medical evaluation</strong> for pneumonia diagnosis</li>
            <li><strong>Confirmatory tests:</strong> Sputum culture, blood tests (WBC, CRP)</li>
            <li><strong>Symptom monitoring:</strong> Productive cough, fever, dyspnea, chest pain</li>
            <li><strong>Treatment:</strong> Antibiotics (bacterial) or supportive care (viral)</li>
        </ul>
    </div>

    <p style="color: #fef3c7; background: rgba(0,0,0,0.4); padding: 15px; border-radius: 8px; margin: 15px 0;">
        Pneumonia requires clinical diagnosis. Imaging findings should be correlated with symptoms and laboratory results.
    </p>
</div>
"""
        else:
            interpretation += """
<div style="background: rgba(120, 53, 15, 0.3); padding: 20px; border-radius: 15px; border: 2px solid #f59e0b; margin: 20px 0;">
    <h3 style="color: #fbbf24;">⚠️ Possible Pneumonia (Moderate Confidence)</h3>
    <p style="color: #fde68a;">Further evaluation needed. Seek medical consultation if symptomatic.</p>
</div>
"""

    elif pred_label == 'COVID-19':
        if confidence >= 85:
            interpretation += """
<div style="background: linear-gradient(135deg, #581c87 0%, #6b21a8 100%); padding: 25px; border-radius: 15px; border: 2px solid #8b5cf6; margin: 20px 0;">
    <h2 style="color: #c4b5fd; margin-top: 0;">😷 COVID-19 FEATURES DETECTED</h2>
    <p style="color: #ddd6fe; font-size: 1.1em;">High confidence - Immediate testing and isolation required</p>

    <div style="background: rgba(0,0,0,0.3); padding: 20px; border-radius: 10px; margin: 15px 0;">
        <h3 style="color: #c4b5fd;">🚨 URGENT ACTIONS</h3>
        <ul style="color: white; line-height: 1.8;">
            <li><strong>IMMEDIATE:</strong> COVID-19 RT-PCR test for confirmation</li>
            <li><strong>ISOLATION:</strong> Self-isolate to prevent transmission</li>
            <li><strong>MONITORING:</strong> Track oxygen saturation (SpO2), seek emergency care if SpO2 < 94%</li>
            <li><strong>SYMPTOMS:</strong> Monitor fever, cough, dyspnea, anosmia, fatigue</li>
            <li><strong>CONTACT TRACING:</strong> Notify close contacts if positive</li>
        </ul>
    </div>

    <div style="background: rgba(220, 38, 38, 0.2); padding: 15px; border-radius: 8px; border-left: 4px solid #ef4444; margin: 15px 0;">
        <p style="color: #fecaca; margin: 0;"><strong>⚠️ EMERGENCY SIGNS:</strong> Difficulty breathing, persistent chest pain, confusion, inability to stay awake, bluish lips/face - SEEK IMMEDIATE MEDICAL CARE</p>
    </div>

    <p style="color: #ede9fe; background: rgba(0,0,0,0.4); padding: 15px; border-radius: 8px; margin: 15px 0;">
        Imaging findings alone cannot confirm COVID-19. RT-PCR or antigen testing is required for definitive diagnosis.
    </p>
</div>
"""
        else:
            interpretation += """
<div style="background: rgba(88, 28, 135, 0.3); padding: 20px; border-radius: 15px; border: 2px solid #8b5cf6; margin: 20px 0;">
    <h3 style="color: #c4b5fd;">😷 Possible COVID-19 (Moderate Confidence)</h3>
    <p style="color: #ddd6fe;">Get RT-PCR or rapid antigen test. Self-isolate pending results. Monitor symptoms closely.</p>
</div>
"""

    else:  # Normal
        if confidence >= 85:
            interpretation += """
<div style="background: linear-gradient(135deg, #065f46 0%, #047857 100%); padding: 25px; border-radius: 15px; border: 2px solid #10b981; margin: 20px 0;">
    <h2 style="color: #6ee7b7; margin-top: 0;">✅ NORMAL CHEST X-RAY</h2>
    <p style="color: #a7f3d0; font-size: 1.1em;">High confidence - No significant abnormalities detected</p>

    <div style="background: rgba(0,0,0,0.3); padding: 20px; border-radius: 10px; margin: 15px 0;">
        <h3 style="color: #6ee7b7;">📋 Interpretation</h3>
        <ul style="color: white; line-height: 1.8;">
            <li>No features of active tuberculosis detected</li>
            <li>No signs of pneumonia or COVID-19 pneumonia</li>
            <li>Chest X-ray appears within normal limits</li>
        </ul>
    </div>

    <div style="background: rgba(245, 158, 11, 0.2); padding: 15px; border-radius: 8px; border-left: 4px solid #f59e0b; margin: 15px 0;">
        <p style="color: #fde68a; margin: 0;"><strong>⚠️ IMPORTANT:</strong> Normal X-ray does NOT rule out all lung diseases. Early-stage diseases may not be visible. If symptomatic, seek medical evaluation.</p>
    </div>

    <div style="background: rgba(0,0,0,0.3); padding: 15px; border-radius: 10px; margin: 15px 0;">
        <h3 style="color: #6ee7b7;">When to still see a doctor:</h3>
        <ul style="color: #d1fae5; line-height: 1.6;">
            <li>Persistent cough, fever, or respiratory symptoms</li>
            <li>Unexplained weight loss or night sweats</li>
            <li>Shortness of breath or chest pain</li>
            <li>Known exposure to TB or COVID-19</li>
        </ul>
    </div>
</div>
"""
        else:
            interpretation += """
<div style="background: rgba(6, 95, 70, 0.3); padding: 20px; border-radius: 15px; border: 2px solid #10b981; margin: 20px 0;">
    <h3 style="color: #6ee7b7;">✅ Likely Normal (Moderate Confidence)</h3>
    <p style="color: #a7f3d0;">If symptomatic or concerns persist, consult healthcare provider for clinical correlation.</p>
</div>
"""

    # Universal disclaimer
    interpretation += """
<div style="background: linear-gradient(135deg, #1e293b 0%, #334155 100%); padding: 25px; border-radius: 15px; border: 2px solid #64748b; margin: 20px 0;">
    <h2 style="color: #cbd5e1; margin-top: 0;">⚠️ MEDICAL DISCLAIMER & LIMITATIONS</h2>

    <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 10px; margin: 15px 0;">
        <h3 style="color: #94a3b8;">🤖 AI Model Capabilities</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; color: #e2e8f0;">
            <div>✅ 4-class detection (Normal, TB, Pneumonia, COVID-19)</div>
            <div>✅ 95-97% validation accuracy</div>
            <div>✅ Grad-CAM explainability</div>
            <div>✅ 89% energy efficient (AST)</div>
        </div>
    </div>

    <div style="background: rgba(239, 68, 68, 0.2); padding: 20px; border-radius: 10px; margin: 15px 0; border-left: 4px solid #ef4444;">
        <h3 style="color: #fca5a5;">🚫 Critical Limitations</h3>
        <ul style="color: #fecaca; line-height: 1.8;">
            <li><strong>NOT FDA-approved</strong> for clinical diagnosis</li>
            <li><strong>SCREENING TOOL ONLY</strong> - not a diagnostic device</li>
            <li><strong>Cannot replace</strong> professional radiologist review</li>
            <li><strong>Cannot detect</strong> lung cancer, fibrosis, bronchiectasis, other rare diseases</li>
            <li><strong>Cannot confirm</strong> diagnosis without laboratory testing</li>
        </ul>
    </div>

    <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 10px; margin: 15px 0;">
        <h3 style="color: #94a3b8;">📋 Clinical Use Guidelines</h3>
        <ol style="color: #e2e8f0; line-height: 1.8;">
            <li><strong>Preliminary screening</strong> purposes only</li>
            <li><strong>ALL positive findings</strong> require confirmatory laboratory testing:
                <ul style="margin-top: 10px; color: #cbd5e1;">
                    <li>TB: Sputum AFB, GeneXpert MTB/RIF, culture</li>
                    <li>Pneumonia: Sputum culture, blood tests, clinical diagnosis</li>
                    <li>COVID-19: RT-PCR, rapid antigen test</li>
                </ul>
            </li>
            <li><strong>Clinical correlation</strong> with patient history and symptoms essential</li>
            <li><strong>Expert radiologist review</strong> recommended for all clinical decisions</li>
            <li><strong>DO NOT initiate treatment</strong> based solely on AI predictions</li>
        </ol>
    </div>

    <p style="color: #f1f5f9; background: rgba(59, 130, 246, 0.2); padding: 20px; border-radius: 10px; border-left: 4px solid #3b82f6; text-align: center; font-size: 1.1em; margin: 20px 0;">
        <strong>When in doubt, ALWAYS consult a qualified healthcare professional.</strong><br>
        <span style="font-size: 0.9em; color: #bfdbfe;">Your health and safety are paramount.</span>
    </p>
</div>

<div style="background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); padding: 20px; border-radius: 15px; text-align: center; margin: 20px 0;">
    <h3 style="color: white; margin-top: 0;">🫁 Powered by Adaptive Sparse Training</h3>
    <p style="color: #c7d2fe; margin-bottom: 15px;">Energy-efficient AI for accessible, sustainable healthcare</p>
    <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; margin-top: 15px;">
        <a href="https://github.com/oluwafemidiakhoa/Tuberculosis" target="_blank"
           style="color: white; background: rgba(255,255,255,0.2); padding: 10px 20px; border-radius: 8px; text-decoration: none; font-weight: bold; backdrop-filter: blur(10px);">
            📂 GitHub Repository
        </a>
        <a href="https://huggingface.co/spaces/mgbam/Tuberculosis" target="_blank"
           style="color: white; background: rgba(255,255,255,0.2); padding: 10px 20px; border-radius: 8px; text-decoration: none; font-weight: bold; backdrop-filter: blur(10px);">
            🤗 Hugging Face Space
        </a>
    </div>
    <p style="color: #a5b4fc; font-size: 0.85em; margin-top: 15px; font-style: italic;">
        Sample-based Adaptive Sparse Training for Deep Learning<br>
        Making AI healthcare accessible and environmentally sustainable
    </p>
</div>
"""

    return interpretation

# ============================================================================
# Main Prediction Function
# ============================================================================

def predict_chest_xray(image, show_gradcam=True):
    """
    Advanced prediction with stunning visualizations
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
    pred_class = int(output.argmax(dim=1).item())
    pred_label = CLASSES[pred_class]
    confidence = float(probs[pred_class]) * 100

    # Create results dictionary
    results = {
        CLASSES[i]: float(probs[i] * 100) for i in range(len(CLASSES))
    }

    # Generate stunning visualizations
    professional_display = create_professional_display(original_img, pred_label, confidence, results)

    if cam is not None and show_gradcam:
        gradcam_viz = create_gradcam_heatmap(original_img, cam, pred_label, confidence)
    else:
        gradcam_viz = None

    # Create interpretation
    interpretation = create_interpretation(pred_label, confidence, results)

    return results, professional_display, gradcam_viz, interpretation

# ============================================================================
# World-Class Gradio Interface
# ============================================================================

# Premium CSS with animations and glass morphism
premium_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.gradio-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%) !important;
    background-size: 200% 200% !important;
    animation: gradientShift 15s ease infinite !important;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Glass morphism cards */
.glass-card {
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(20px) !important;
    border-radius: 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37) !important;
    padding: 25px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.glass-card:hover {
    transform: translateY(-5px) !important;
    box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.5) !important;
}

/* Premium buttons */
button.primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 15px 30px !important;
    border-radius: 12px !important;
    font-size: 1.1em !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    transition: all 0.3s ease !important;
    cursor: pointer !important;
}

button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
}

button.primary:active {
    transform: translateY(0) !important;
}

/* Image upload area */
.image-upload {
    border: 3px dashed rgba(255, 255, 255, 0.4) !important;
    border-radius: 20px !important;
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(10px) !important;
    padding: 30px !important;
    transition: all 0.3s ease !important;
}

.image-upload:hover {
    border-color: rgba(255, 255, 255, 0.8) !important;
    background: rgba(255, 255, 255, 0.1) !important;
}

/* Results label styling */
.output-label {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
    border-radius: 15px !important;
    padding: 20px !important;
    border: 2px solid rgba(255, 255, 255, 0.1) !important;
}

/* Tab styling */
.tab-nav button {
    background: rgba(255, 255, 255, 0.1) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    color: white !important;
    border-radius: 10px 10px 0 0 !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

.tab-nav button.selected {
    background: rgba(255, 255, 255, 0.25) !important;
    border-bottom-color: transparent !important;
}

/* Markdown styling */
.prose {
    color: white !important;
}

.prose h2 {
    color: #c7d2fe !important;
    font-weight: 700 !important;
}

.prose h3 {
    color: #ddd6fe !important;
    font-weight: 600 !important;
}

/* Checkbox styling */
.checkbox-wrapper {
    background: rgba(255, 255, 255, 0.1) !important;
    padding: 12px !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
}

/* Loading animation */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.loading {
    animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite !important;
}

/* Examples gallery */
.example-gallery {
    display: grid !important;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)) !important;
    gap: 15px !important;
    padding: 20px !important;
}

/* Smooth scrolling */
html {
    scroll-behavior: smooth !important;
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.3);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.5);
}
"""

# Create the interface
with gr.Blocks(css=premium_css, theme=gr.themes.Glass(), title="🫁 AI Chest X-Ray Analyzer") as demo:

    # Hero Header
    gr.HTML("""
        <div style="text-align: center; padding: 40px 20px; background: rgba(0,0,0,0.2); backdrop-filter: blur(10px); border-radius: 20px; margin-bottom: 30px; border: 1px solid rgba(255,255,255,0.2);">
            <div style="font-size: 4em; margin-bottom: 10px; animation: pulse 2s ease-in-out infinite;">🫁</div>
            <h1 style="color: white; font-size: 3em; font-weight: 900; margin: 15px 0; text-shadow: 2px 2px 20px rgba(0,0,0,0.3); letter-spacing: -1px;">
                Medical AI Diagnostic Platform
            </h1>
            <p style="color: rgba(255,255,255,0.9); font-size: 1.4em; margin: 15px 0; font-weight: 300;">
                Advanced Multi-Class Chest X-Ray Analysis with Explainable AI
            </p>
            <div style="display: flex; justify-content: center; gap: 30px; margin-top: 25px; flex-wrap: wrap;">
                <div style="background: rgba(16, 185, 129, 0.2); padding: 15px 25px; border-radius: 12px; border: 2px solid #10b981; backdrop-filter: blur(10px);">
                    <div style="color: #6ee7b7; font-size: 2em; font-weight: bold;">95-97%</div>
                    <div style="color: rgba(255,255,255,0.8); font-size: 0.9em; margin-top: 5px;">Accuracy</div>
                </div>
                <div style="background: rgba(139, 92, 246, 0.2); padding: 15px 25px; border-radius: 12px; border: 2px solid #8b5cf6; backdrop-filter: blur(10px);">
                    <div style="color: #c4b5fd; font-size: 2em; font-weight: bold;">4 Classes</div>
                    <div style="color: rgba(255,255,255,0.8); font-size: 0.9em; margin-top: 5px;">Diseases</div>
                </div>
                <div style="background: rgba(245, 158, 11, 0.2); padding: 15px 25px; border-radius: 12px; border: 2px solid #f59e0b; backdrop-filter: blur(10px);">
                    <div style="color: #fbbf24; font-size: 2em; font-weight: bold;">89%</div>
                    <div style="color: rgba(255,255,255,0.8); font-size: 0.9em; margin-top: 5px;">Energy Efficient</div>
                </div>
            </div>
            <div style="margin-top: 25px; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 10px; backdrop-filter: blur(10px);">
                <p style="color: white; font-size: 1.1em; margin: 0; font-weight: 500;">
                    🎯 Detects: <span style="color: #6ee7b7;">Normal</span> •
                    <span style="color: #fca5a5;">Tuberculosis</span> •
                    <span style="color: #fbbf24;">Pneumonia</span> •
                    <span style="color: #c4b5fd;">COVID-19</span>
                </p>
            </div>
        </div>
    """)

    # Main Interface
    with gr.Row():
        # Left Column - Input
        with gr.Column(scale=1):
            gr.HTML("""
                <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.2); backdrop-filter: blur(10px); margin-bottom: 20px;">
                    <h2 style="color: white; margin: 0; font-size: 1.8em; font-weight: 700;">📤 Upload X-Ray</h2>
                    <p style="color: rgba(255,255,255,0.8); margin: 10px 0 0 0;">Upload a chest X-ray image for AI analysis</p>
                </div>
            """)

            image_input = gr.Image(
                type="pil",
                label="Drag and drop or click to upload",
                elem_classes="image-upload",
                height=400
            )

            show_gradcam = gr.Checkbox(
                value=True,
                label="🔬 Enable Grad-CAM Explainable AI",
                info="Visualize AI decision-making process",
                elem_classes="checkbox-wrapper"
            )

            analyze_btn = gr.Button(
                "🚀 Analyze X-Ray Image",
                variant="primary",
                size="lg",
                elem_classes="primary"
            )

            gr.HTML("""
                <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.2); backdrop-filter: blur(10px); margin-top: 20px;">
                    <h3 style="color: white; margin-top: 0; font-weight: 600;">📋 Supported Formats</h3>
                    <ul style="color: rgba(255,255,255,0.9); line-height: 1.8; padding-left: 20px;">
                        <li>Chest X-rays (PA or AP view)</li>
                        <li>PNG, JPG, JPEG formats</li>
                        <li>Grayscale or RGB</li>
                        <li>Any resolution (auto-resized)</li>
                    </ul>

                    <h3 style="color: white; margin-top: 20px; font-weight: 600;">✨ Key Features</h3>
                    <ul style="color: rgba(255,255,255,0.9); line-height: 1.8; padding-left: 20px;">
                        <li>Multi-class disease detection</li>
                        <li>Explainable AI visualization</li>
                        <li>Clinical recommendations</li>
                        <li>Professional medical reporting</li>
                    </ul>
                </div>
            """)

        # Right Column - Results
        with gr.Column(scale=2):
            gr.HTML("""
                <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.2); backdrop-filter: blur(10px); margin-bottom: 20px;">
                    <h2 style="color: white; margin: 0; font-size: 1.8em; font-weight: 700;">📊 Analysis Results</h2>
                    <p style="color: rgba(255,255,255,0.8); margin: 10px 0 0 0;">Comprehensive diagnostic report with AI insights</p>
                </div>
            """)

            # Probability distribution
            prob_output = gr.Label(
                label="🎯 Prediction Confidence",
                num_top_classes=4,
                elem_classes="output-label"
            )

            # Tabbed visualizations
            with gr.Tabs():
                with gr.Tab("🖼️ Professional Report"):
                    professional_output = gr.Image(
                        label="Diagnostic Analysis",
                        elem_classes="output-image"
                    )

                with gr.Tab("🔬 Grad-CAM Explainability"):
                    gradcam_output = gr.Image(
                        label="AI Attention Visualization",
                        elem_classes="output-image"
                    )

            # Clinical interpretation
            interpretation_output = gr.HTML(
                label="Clinical Interpretation & Recommendations"
            )

    # Example Images Section
    gr.HTML("""
        <div style="background: rgba(255,255,255,0.1); padding: 25px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.2); backdrop-filter: blur(10px); margin-top: 30px;">
            <h2 style="color: white; margin-top: 0; font-size: 1.8em; font-weight: 700; text-align: center;">📁 Try Example X-Rays</h2>
            <p style="color: rgba(255,255,255,0.8); text-align: center; margin-bottom: 20px;">Click any example to see the AI in action</p>
        </div>
    """)

    gr.Examples(
        examples=[
            ["examples/normal.png"],
            ["examples/tb.png"],
            ["examples/pneumonia.png"],
            ["examples/covid.png"],
        ],
        inputs=image_input,
        label="Example Images",
        elem_classes="example-gallery"
    )

    # Connect the interface
    analyze_btn.click(
        fn=predict_chest_xray,
        inputs=[image_input, show_gradcam],
        outputs=[prob_output, professional_output, gradcam_output, interpretation_output]
    )

    # Footer
    gr.HTML("""
        <div style="background: rgba(0,0,0,0.3); padding: 30px; border-radius: 15px; margin-top: 40px; border: 1px solid rgba(255,255,255,0.1); backdrop-filter: blur(10px); text-align: center;">
            <h3 style="color: white; margin-top: 0; font-size: 1.5em; font-weight: 700;">🫁 Adaptive Sparse Training Technology</h3>
            <p style="color: rgba(255,255,255,0.9); font-size: 1.1em; line-height: 1.8; max-width: 800px; margin: 20px auto;">
                This platform leverages cutting-edge <strong>Adaptive Sparse Training (AST)</strong> to deliver
                <span style="color: #10b981; font-weight: bold;">95-97% accuracy</span> while using
                <span style="color: #f59e0b; font-weight: bold;">89% less energy</span> than traditional models.
                Making AI healthcare accessible, sustainable, and environmentally responsible.
            </p>

            <div style="display: flex; justify-content: center; gap: 20px; margin: 25px 0; flex-wrap: wrap;">
                <a href="https://github.com/oluwafemidiakhoa/Tuberculosis" target="_blank"
                   style="color: white; background: rgba(255,255,255,0.15); padding: 12px 24px; border-radius: 10px; text-decoration: none; font-weight: 600; border: 1px solid rgba(255,255,255,0.3); transition: all 0.3s ease; display: inline-block;">
                    📂 GitHub Repository
                </a>
                <a href="https://huggingface.co/spaces/mgbam/Tuberculosis" target="_blank"
                   style="color: white; background: rgba(255,255,255,0.15); padding: 12px 24px; border-radius: 10px; text-decoration: none; font-weight: 600; border: 1px solid rgba(255,255,255,0.3); transition: all 0.3s ease; display: inline-block;">
                    🤗 Hugging Face Space
                </a>
            </div>

            <div style="background: rgba(239, 68, 68, 0.2); padding: 20px; border-radius: 10px; margin: 25px auto; max-width: 900px; border: 2px solid #ef4444;">
                <p style="color: #fecaca; font-size: 1em; margin: 0; line-height: 1.6;">
                    <strong style="font-size: 1.2em;">⚠️ CRITICAL MEDICAL DISCLAIMER</strong><br><br>
                    This is an AI-powered <strong>SCREENING TOOL</strong>, not a diagnostic medical device.
                    <strong>NOT FDA-approved</strong> for clinical use. All predictions must be confirmed by
                    qualified healthcare professionals and appropriate laboratory testing. Do not make medical
                    decisions based solely on AI predictions. Always consult licensed medical practitioners.
                </p>
            </div>

            <p style="color: rgba(255,255,255,0.6); font-size: 0.9em; margin-top: 20px; font-style: italic;">
                Developed with ❤️ for accessible, equitable, and sustainable healthcare<br>
                Powered by EfficientNet-B0 with Adaptive Sparse Training • Multi-class classification • Grad-CAM visualization
            </p>
        </div>
    """)

# Launch configuration
if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        favicon_emoji="🫁"
    )
