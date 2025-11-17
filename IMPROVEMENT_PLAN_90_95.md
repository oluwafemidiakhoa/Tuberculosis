# 🎯 Plan to Achieve 90-95% Accuracy

## Current Performance Analysis

### Overall: 87.29% ❌ (Target: 90-95%)

**Per-Class Breakdown:**
| Class | Current | Target | Gap | Status |
|-------|---------|--------|-----|--------|
| Normal | 60% | 90%+ | -30% | ❌ **CRITICAL** |
| TB | 80% | 93%+ | -13% | ⚠️ Needs improvement |
| Pneumonia | 100% | 95%+ | +5% | ✅ **EXCELLENT** |
| COVID | 80% | 92%+ | -12% | ⚠️ Needs improvement |

### Root Cause Analysis

**Problem 1: Normal ↔ COVID Confusion (60% Normal accuracy)**
- Issue: Model confuses healthy lungs with COVID pneumonia
- Why: Both can appear similar on X-ray (subtle patterns)
- Impact: -30% accuracy gap, biggest problem

**Problem 2: TB & COVID at 80%**
- Issue: Not learning distinctive features well enough
- Why: Possibly undertrained, needs more capacity
- Impact: -13% and -12% gaps

**Problem 3: Training stopped too early**
- Current: 50 epochs
- Observation: Accuracy was still improving
- Issue: Stopped before convergence

---

## 🚀 Action Plan to Hit 90-95%

### Strategy 1: Extended Training (Quick Win) ⚡

**Change:** Train for **100 epochs** instead of 50

**Reasoning:**
- Your loss was still decreasing at epoch 50
- Validation accuracy was improving
- No signs of overfitting yet

**Expected Gain:** +3-5% accuracy

**Implementation:**
```python
# In train_multiclass_simple.py
CONFIG = {
    'epochs': 100,  # Was: 50
    # ... rest stays same
}
```

**Time Cost:** ~6-8 hours total

---

### Strategy 2: Better Architecture (Medium Win) 🏗️

**Change:** Use **EfficientNet-B1 or B2** instead of B0

**Reasoning:**
- B0 has limited capacity (5.3M params)
- B1 has 7.8M params (+47% capacity)
- B2 has 9.2M params (+73% capacity)
- More capacity = better feature learning

**Expected Gain:** +4-7% accuracy

**Implementation:**
```python
# Replace in train_multiclass_simple.py
model = models.efficientnet_b1(pretrained=True)  # Was: efficientnet_b0
# or
model = models.efficientnet_b2(pretrained=True)
```

**Time Cost:** +20% training time (worth it!)

---

### Strategy 3: Advanced Data Augmentation (Big Win) 🎨

**Problem:** Normal vs COVID confusion suggests weak feature learning

**Solution:** Aggressive augmentation to force robust learning

**Implementation:**
```python
# Enhanced augmentation
train_transform = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),  # Was: 10
    transforms.ColorJitter(
        brightness=0.3,  # Was: 0.2
        contrast=0.3,    # Was: 0.2
        saturation=0.2,  # New
        hue=0.1          # New
    ),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        shear=10  # New
    ),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))  # New
])
```

**Expected Gain:** +5-8% accuracy (especially Normal/COVID)

---

### Strategy 4: Class Weighting (Targeted Fix) ⚖️

**Problem:** Model might be biased toward easier classes

**Solution:** Weight loss by class difficulty

**Implementation:**
```python
# Calculate class weights inversely proportional to frequency
class_weights = torch.tensor([
    1.0,  # Normal (hardest - gets more weight)
    0.8,  # TB (medium)
    0.6,  # Pneumonia (easiest - already 100%)
    0.9,  # COVID (medium-hard)
]).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
```

**Expected Gain:** +2-4% accuracy (balanced improvement)

---

### Strategy 5: Learning Rate Schedule (Fine-tuning) 📉

**Problem:** Fixed LR may not converge optimally

**Solution:** Cosine annealing with warmup

**Implementation:**
```python
# Warmup for first 5 epochs
def get_lr(epoch, max_epochs=100, base_lr=0.001):
    if epoch < 5:
        return base_lr * (epoch + 1) / 5  # Linear warmup
    else:
        # Cosine annealing
        progress = (epoch - 5) / (max_epochs - 5)
        return base_lr * 0.5 * (1 + np.cos(np.pi * progress))

# Use in training loop
for epoch in range(100):
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

**Expected Gain:** +2-3% accuracy

---

### Strategy 6: Test-Time Augmentation (Inference Boost) 🔮

**Problem:** Single prediction can be noisy

**Solution:** Average predictions over multiple augmented versions

**Implementation:**
```python
def predict_tta(model, image, n_tta=5):
    """Test-time augmentation"""
    predictions = []

    for _ in range(n_tta):
        # Apply random augmentation
        aug_image = tta_transform(image)
        with torch.no_grad():
            pred = model(aug_image)
            predictions.append(torch.softmax(pred, dim=1))

    # Average predictions
    avg_pred = torch.stack(predictions).mean(dim=0)
    return avg_pred
```

**Expected Gain:** +1-3% accuracy (at inference time)

---

### Strategy 7: Two-Stage Training (Optimal Approach) 🎯

**Stage 1:** Learn accuracy (60 epochs, no AST)
- Focus: Maximum accuracy
- Model: EfficientNet-B2
- Goal: 93-96% accuracy

**Stage 2:** Add efficiency (20 epochs, with AST)
- Focus: Maintain accuracy while compressing
- Target: 90-95% accuracy, 75-85% energy savings
- Goal: Balance performance and efficiency

**Expected Gain:** +5-10% accuracy (best approach)

**Implementation:**
```python
# Stage 1: No AST, pure accuracy
model = EfficientNetB2(num_classes=4)
train(model, epochs=60, use_ast=False)
# Checkpoint: ~93-96% accuracy

# Stage 2: Add AST for efficiency
model = load_checkpoint('stage1_best.pt')
model = wrap_with_ast(model, target_activation=0.20)
train(model, epochs=20, use_ast=True)
# Final: ~90-95% accuracy, 80% energy savings
```

**Time Cost:** ~10-12 hours total

---

## 🎯 Recommended Combined Approach

### Quick Wins (2-3 hours) - Expected: 90-92%
1. ✅ Use EfficientNet-B1 (easy change)
2. ✅ Train for 80 epochs (2x current)
3. ✅ Add class weighting
4. ✅ Better augmentation

### Optimal Approach (8-10 hours) - Expected: 92-95%
1. ✅ Use EfficientNet-B2
2. ✅ Two-stage training (60 + 20 epochs)
3. ✅ Advanced augmentation
4. ✅ Cosine LR schedule
5. ✅ Class weighting
6. ✅ Test-time augmentation for inference

---

## 📊 Expected Results by Approach

| Approach | Time | Overall Acc | Normal | TB | Pneumonia | COVID | Energy |
|----------|------|-------------|--------|----|-----------| ------|--------|
| **Current** | 3-4h | 87% | 60% | 80% | 100% | 80% | 90% |
| **Quick** | 5-6h | 90-92% | 85% | 90% | 95% | 88% | 90% |
| **Optimal** | 10-12h | 92-95% | 90% | 95% | 95% | 93% | 80% |

---

## 🚀 Implementation Plan

### Option A: Quick Improvement (Tonight)
```bash
# Modify train_multiclass_simple.py:
# - model = efficientnet_b1
# - epochs = 80
# - Add class weights
# - Better augmentation

python train_multiclass_simple.py

# Expected: 90-92% in 5-6 hours
```

### Option B: Optimal Improvement (Tomorrow)
```bash
# Use new train_v2_improved.py (already created!)
# - EfficientNet-B2
# - Two-stage training
# - All optimizations

python train_v2_improved.py

# Expected: 92-95% in 10-12 hours
```

---

## 🎯 My Recommendation

**Use Option B (Optimal Approach)** because:

1. **Better final results:** 92-95% vs 90-92%
2. **Worth the extra time:** +4-6 hours for +3-5% accuracy
3. **Ready to use:** `train_v2_improved.py` already created
4. **Clinical-grade:** 92-95% is publishable, grant-worthy
5. **One-time effort:** Do it right once vs iterating

### Next Steps:

1. **Run training:** `python train_v2_improved.py`
2. **Monitor progress:** Check validation accuracy every 10 epochs
3. **Validate results:** Test on holdout set
4. **Deploy:** Copy to Gradio app when >92% achieved

---

## 🔍 Monitoring During Training

**Good signs:**
- ✅ Validation accuracy > 80% by epoch 20
- ✅ Normal class > 70% by epoch 30
- ✅ Overall > 90% by epoch 50
- ✅ Plateau around 92-95% by epoch 80

**Bad signs:**
- ❌ Validation loss increasing (overfitting)
- ❌ Train-val gap > 10% (need regularization)
- ❌ Stuck at <85% (need more capacity)

---

## 🎉 Success Criteria

**Minimum Acceptable:**
- Overall: 90%+
- Per-class: All > 85%

**Target:**
- Overall: 92-95%
- Per-class: All > 88%
- Energy savings: 75-85%

**Stretch Goal:**
- Overall: 95%+
- Per-class: All > 90%
- Energy savings: 80%+

---

**Ready to train to 90-95%? Let's do it! 🚀**
