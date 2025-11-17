"""
Specificity Test + Grad-CAM Visualization (FIXED)
--------------------------------------------------
Tests the model's ability to distinguish between diseases
and generates Grad-CAM visualizations.

Fixes:
1. Correct checkpoint loading (removes "model." prefix properly)
2. Handles both stage1 and stage2 checkpoints
3. Better visualization with all 4 classes
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    'data_dir': 'data_multiclass',
    'checkpoint_path': 'checkpoints_multiclass_best/best.pt',  # or best_stage1.pt
    'output_dir': 'gradcam_results',
    'model_variant': 'b2',  # Must match training
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

CLASSES = ['Normal', 'TB', 'Pneumonia', 'COVID']

Path(CONFIG['output_dir']).mkdir(exist_ok=True)

print(f"Device: {CONFIG['device']}")
print(f"Loading model from: {CONFIG['checkpoint_path']}")

# ============================================================================
# Load Model (FIXED)
# ============================================================================

def load_model(checkpoint_path, model_variant='b2', num_classes=4, device='cpu'):
    """Load model with proper checkpoint handling"""

    # Create base model
    if model_variant == 'b0':
        model = models.efficientnet_b0(weights=None)
        in_features = 1280
    elif model_variant == 'b1':
        model = models.efficientnet_b1(weights=None)
        in_features = 1280
    elif model_variant == 'b2':
        model = models.efficientnet_b2(weights=None)
        in_features = 1408
    elif model_variant == 'b3':
        model = models.efficientnet_b3(weights=None)
        in_features = 1536
    else:
        raise ValueError(f"Unknown model variant: {model_variant}")

    # Replace classifier
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    state_dict = {}
    for key, value in checkpoint.items():
        # Skip activation_mask buffer
        if key == 'activation_mask':
            continue

        # Remove "model." prefix if present
        if key.startswith('model.'):
            new_key = key.replace('model.', '', 1)
            state_dict[new_key] = value  # FIXED: use new_key, not key!
        else:
            state_dict[key] = value

    # Load state dict
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully")
    return model

model = load_model(
    CONFIG['checkpoint_path'],
    CONFIG['model_variant'],
    len(CLASSES),
    CONFIG['device']
)

# ============================================================================
# Transforms
# ============================================================================

if CONFIG['model_variant'] == 'b0':
    img_size = 224
elif CONFIG['model_variant'] in ['b1', 'b2']:
    img_size = 260
else:
    img_size = 300

transform = transforms.Compose([
    transforms.Resize(img_size + 28),
    transforms.CenterCrop(img_size),
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
        target_layer.register_full_backward_hook(
            lambda m, gi, go: save_gradient(go[0])
        )

    def generate(self, input_img):
        # Forward pass
        output = self.model(input_img)
        pred_class = output.argmax(dim=1)

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][pred_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        if self.gradients is None:
            print("Warning: No gradients captured")
            return None, output

        # Generate CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam, output

# Setup Grad-CAM on last feature layer
if CONFIG['model_variant'] in ['b0', 'b1', 'b2', 'b3']:
    target_layer = model.features[-1]
else:
    target_layer = model.features[-1]

grad_cam = GradCAM(model, target_layer)

print("✓ Grad-CAM setup complete")

# ============================================================================
# Prediction Function
# ============================================================================

def predict(img_path):
    """Predict class and confidence"""
    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(CONFIG['device'])

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()

    pred_idx = out.argmax(dim=1).item()
    return CLASSES[pred_idx], float(probs[pred_idx] * 100), probs

# ============================================================================
# SPECIFICITY TEST
# ============================================================================

print("\n" + "="*60)
print("SPECIFICITY TEST - Can we distinguish diseases?")
print("="*60 + "\n")

specificity_results = {}

for cls in CLASSES:
    test_path = Path(CONFIG['data_dir']) / 'test' / cls

    # Get test images (check if directory exists)
    if not test_path.exists():
        print(f"Warning: {test_path} does not exist. Skipping {cls}.")
        continue

    test_imgs = list(test_path.glob('*.png'))[:5]

    if len(test_imgs) == 0:
        print(f"Warning: No test images found for {cls}. Skipping.")
        continue

    print(f"\nTesting {cls}:")
    correct = 0
    predictions = []

    for img_path in test_imgs:
        pred, conf, probs = predict(img_path)
        is_correct = pred == cls
        correct += is_correct
        symbol = "✓" if is_correct else "✗"
        print(f"  {symbol} Predicted: {pred:12s} ({conf:.1f}%)")
        predictions.append((pred, conf, probs))

    accuracy = (correct / len(test_imgs)) * 100
    print(f"  Accuracy: {accuracy:.1f}% ({correct}/{len(test_imgs)})")

    specificity_results[cls] = {
        'accuracy': accuracy,
        'correct': correct,
        'total': len(test_imgs),
        'predictions': predictions
    }

print("\n" + "="*60)

# Calculate overall specificity
if specificity_results:
    total_correct = sum(r['correct'] for r in specificity_results.values())
    total_samples = sum(r['total'] for r in specificity_results.values())
    overall_acc = (total_correct / total_samples) * 100 if total_samples > 0 else 0

    print(f"Overall Specificity: {overall_acc:.1f}% ({total_correct}/{total_samples})")
    print("="*60)

    # Check for specific confusions
    print("\n🔍 Key Checks:")
    if 'Pneumonia' in specificity_results:
        pneu_acc = specificity_results['Pneumonia']['accuracy']
        if pneu_acc >= 80:
            print(f"  ✅ Pneumonia distinction: GOOD ({pneu_acc:.1f}%)")
        else:
            print(f"  ⚠️  Pneumonia distinction: NEEDS IMPROVEMENT ({pneu_acc:.1f}%)")

    if 'TB' in specificity_results:
        tb_acc = specificity_results['TB']['accuracy']
        if tb_acc >= 80:
            print(f"  ✅ TB detection: GOOD ({tb_acc:.1f}%)")
        else:
            print(f"  ⚠️  TB detection: NEEDS IMPROVEMENT ({tb_acc:.1f}%)")

# ============================================================================
# GRAD-CAM VISUALIZATION
# ============================================================================

print("\n" + "="*60)
print("Generating Grad-CAM Visualizations...")
print("="*60 + "\n")

# Get one sample from each class
samples = []
for cls in CLASSES:
    test_path = Path(CONFIG['data_dir']) / 'test' / cls
    if not test_path.exists():
        continue
    img_files = list(test_path.glob('*.png'))
    if img_files:
        samples.append((img_files[0], cls))

if len(samples) == 0:
    print("Error: No test images found. Please check data_multiclass/test/ directory.")
else:
    # Create visualization
    fig, axes = plt.subplots(len(samples), 3, figsize=(15, 4.5*len(samples)))
    if len(samples) == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('Grad-CAM Visualization - Explainable AI for Disease Detection',
                 fontsize=20, fontweight='bold', y=0.995)

    for idx, (img_path, true_class) in enumerate(samples):
        # Load and process image
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(CONFIG['device'])

        # Generate Grad-CAM
        with torch.set_grad_enabled(True):
            cam, output = grad_cam.generate(img_tensor)

        # Get prediction
        probs = torch.softmax(output, dim=1)[0].cpu().detach().numpy()
        pred_idx = output.argmax(dim=1).item()
        pred_class = CLASSES[pred_idx]
        confidence = probs[pred_idx] * 100

        # Prepare images
        img_resized = img.resize((img_size, img_size))
        img_array = np.array(img_resized)

        if cam is not None:
            cam_resized = cv2.resize(cam, (img_size, img_size))

            # Create heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            # Create overlay
            overlay = img_array * 0.5 + heatmap * 0.5
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        else:
            heatmap = np.zeros_like(img_array)
            overlay = img_array

        # Plot
        axes[idx, 0].imshow(img_resized)
        axes[idx, 0].set_title(f'Original\nTrue: {true_class}', fontsize=12, fontweight='bold')
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(heatmap)
        axes[idx, 1].set_title(f'Grad-CAM\nAttention Map', fontsize=12, fontweight='bold')
        axes[idx, 1].axis('off')

        status = '✓ CORRECT' if pred_class == true_class else '✗ WRONG'
        color = 'green' if pred_class == true_class else 'red'
        axes[idx, 2].imshow(overlay)
        axes[idx, 2].set_title(f'Overlay\nPred: {pred_class} ({confidence:.1f}%)\n{status}',
                              fontsize=12, fontweight='bold', color=color)
        axes[idx, 2].axis('off')

    plt.tight_layout()
    output_path = Path(CONFIG['output_dir']) / 'gradcam_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    print(f"\n✅ Grad-CAM visualization saved to: {output_path}")
    print("   Shows which areas the model focuses on for each disease.")

print("\n" + "="*60)
print("TESTING COMPLETE!")
print("="*60)
