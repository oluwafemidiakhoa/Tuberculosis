#!/bin/bash
# Prepare files for Hugging Face Space deployment

echo "============================================"
echo "Preparing Deployment Files"
echo "============================================"
echo ""

# Check if gradio_app exists
if [ ! -d "gradio_app" ]; then
    echo "❌ Error: gradio_app directory not found!"
    exit 1
fi

echo "✅ Found gradio_app directory"
echo ""

# List deployment files
echo "📦 Deployment Files:"
echo "-------------------"
ls -lh gradio_app/app.py
ls -lh gradio_app/requirements.txt
ls -lh gradio_app/README.md
echo ""

# Check for model weights
if [ -f "gradio_app/checkpoints/best_multiclass.pt" ]; then
    echo "✅ Model weights found:"
    ls -lh gradio_app/checkpoints/best_multiclass.pt
else
    echo "⚠️  Model weights NOT found at: gradio_app/checkpoints/best_multiclass.pt"
    echo ""
    echo "Options:"
    echo "  1. Train model using: python train_v2_improved.py"
    echo "  2. Copy existing weights: cp checkpoints_multiclass/best.pt gradio_app/checkpoints/best_multiclass.pt"
    echo "  3. Deploy without weights (will use ImageNet pretrained - demo only)"
fi
echo ""

echo "============================================"
echo "Deployment Options"
echo "============================================"
echo ""
echo "Option A: Direct to Hugging Face Space"
echo "--------------------------------------"
echo "1. Clone your space:"
echo "   git clone https://huggingface.co/spaces/mgbam/Tuberculosis hf_space"
echo ""
echo "2. Copy files:"
echo "   cp gradio_app/app.py hf_space/"
echo "   cp gradio_app/requirements.txt hf_space/"
echo "   cp gradio_app/README.md hf_space/"
echo "   cp gradio_app/checkpoints/best_multiclass.pt hf_space/checkpoints/ # if exists"
echo ""
echo "3. Commit and push:"
echo "   cd hf_space"
echo "   git add ."
echo "   git commit -m '🚀 Deploy Multi-Class TB Detection v1.0-beta'"
echo "   git push"
echo ""
echo "Option B: GitHub First, Then HF"
echo "-------------------------------"
echo "1. Commit to this repo:"
echo "   git add gradio_app/"
echo "   git commit -m 'Add Gradio deployment app'"
echo "   git push"
echo ""
echo "2. Then deploy to HF Space (clone from GitHub or copy files)"
echo ""
echo "============================================"
echo "Ready to deploy!"
echo "============================================"
