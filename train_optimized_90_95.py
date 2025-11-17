"""
OPTIMIZED Multi-Class Training Script - Target: 90-95% Accuracy
=================================================================

Improvements over train_multiclass_simple.py:
1. EfficientNet-B2 (more capacity: 9.2M vs 5.3M params)
2. 100 epochs (was 50) - train to convergence
3. Advanced data augmentation - better Normal/COVID distinction
4. Class-weighted loss - balance learning
5. Cosine LR with warmup - optimal convergence
6. Gradient clipping - stable training
7. Mixed precision - faster training

Expected Results:
- Overall: 92-95% (was 87%)
- Normal: 90%+ (was 60%)
- TB: 95%+ (was 80%)
- Pneumonia: 95%+ (was 100%)
- COVID: 92%+ (was 80%)
- Energy: 85-90% savings
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torchvision import models, transforms
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
import math

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    # Data
    'data_dir': 'data_multiclass',
    'num_classes': 4,

    # Training - OPTIMIZED
    'epochs': 100,  # Was 50 - train to convergence
    'batch_size': 32,
    'lr': 0.001,  # Will be modulated by warmup/schedule
    'warmup_epochs': 5,  # Linear warmup

    # Model - UPGRADED
    'model': 'efficientnet_b2',  # Was b0 - more capacity
    'dropout': 0.3,  # Regularization

    # AST
    'target_activation_rate': 0.15,  # 15% = 85% savings (was 10%)
    'ast_start_epoch': 10,  # Start AST after initial learning

    # Optimization
    'weight_decay': 0.01,
    'gradient_clip': 1.0,  # Prevent gradient explosion
    'mixed_precision': True,  # Faster training on GPU

    # Checkpointing
    'checkpoint_dir': 'checkpoints_multiclass_optimized',
    'save_frequency': 10,

    # Hardware
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,
}

CLASSES = ['Normal', 'TB', 'Pneumonia', 'COVID']

# Create checkpoint directory
Path(CONFIG['checkpoint_dir']).mkdir(exist_ok=True)

print("="*80)
print("OPTIMIZED MULTI-CLASS TRAINING - Target: 90-95% Accuracy")
print("="*80)
print(f"\nDevice: {CONFIG['device']}")
print(f"Model: {CONFIG['model'].upper()}")
print(f"Epochs: {CONFIG['epochs']}")
print(f"Classes: {CLASSES}\n")

# ============================================================================
# Enhanced Data Augmentation
# ============================================================================

# TRAINING: Aggressive augmentation for robust learning
train_transform = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),  # Was 10
    transforms.ColorJitter(
        brightness=0.3,  # Was 0.2 - stronger variation
        contrast=0.3,    # Was 0.2
        saturation=0.2,  # New - color variation
        hue=0.1          # New - slight hue shift
    ),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        shear=10  # New - perspective variation
    ),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))  # New - occlusion robustness
])

# VALIDATION: Standard preprocessing only
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============================================================================
# Dataset with Class Statistics
# ============================================================================

class ChestXRayDatasetOptimized(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.samples = []

        for class_idx, class_name in enumerate(CLASSES):
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.png'):
                    self.samples.append((img_path, class_idx))

        # Calculate class distribution
        self.class_counts = Counter([label for _, label in self.samples])

        print(f"\n{split.upper()} Dataset: {len(self.samples)} images")
        for cls_idx, cls_name in enumerate(CLASSES):
            count = self.class_counts.get(cls_idx, 0)
            pct = 100 * count / len(self.samples) if self.samples else 0
            print(f"  {cls_name:12s}: {count:5d} ({pct:5.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Robust loading with fallback
        for attempt in range(3):
            try:
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return image, label
            except Exception as e:
                if attempt == 2:
                    # Final fallback: return random sample
                    idx = np.random.randint(0, len(self.samples))
                    img_path, label = self.samples[idx]

# Create datasets
train_dataset = ChestXRayDatasetOptimized(CONFIG['data_dir'], 'train', train_transform)
val_dataset = ChestXRayDatasetOptimized(CONFIG['data_dir'], 'val', val_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=True,
    num_workers=CONFIG['num_workers'],
    pin_memory=True,
    drop_last=True  # For batch norm stability
)
val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False,
    num_workers=CONFIG['num_workers'],
    pin_memory=True
)

# ============================================================================
# Model with Adaptive Sparsity
# ============================================================================

class AdaptiveSparseModelV2(nn.Module):
    def __init__(self, num_classes, model_name='efficientnet_b2', dropout=0.3, target_activation_rate=0.15):
        super().__init__()

        # Load pretrained model
        if model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(pretrained=True)
            in_features = 1280
        elif model_name == 'efficientnet_b1':
            self.model = models.efficientnet_b1(pretrained=True)
            in_features = 1280
        elif model_name == 'efficientnet_b2':
            self.model = models.efficientnet_b2(pretrained=True)
            in_features = 1408
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Replace classifier with dropout
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes)
        )

        self.target_activation_rate = target_activation_rate
        self.register_buffer('activation_mask', torch.ones(1))
        self.ast_enabled = False  # Will enable after warmup

    def forward(self, x):
        return self.model(x)

    def enable_ast(self):
        """Enable AST after warmup period"""
        self.ast_enabled = True
        print("  🔥 AST enabled - starting sparse training")

    def apply_adaptive_sparsity(self):
        """Apply sample-based sparsity (only if enabled)"""
        if not self.ast_enabled:
            return 1.0  # 100% activation when AST disabled

        total_params = 0
        active_params = 0

        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                # Calculate magnitude threshold
                threshold = torch.quantile(
                    torch.abs(param.data.flatten()),
                    1.0 - self.target_activation_rate
                )

                # Create and apply mask
                mask = (torch.abs(param.data) >= threshold).float()
                param.data = param.data * mask

                total_params += param.numel()
                active_params += mask.sum().item()

        activation_rate = active_params / total_params if total_params > 0 else 0
        return activation_rate

# Create model
model = AdaptiveSparseModelV2(
    CONFIG['num_classes'],
    CONFIG['model'],
    CONFIG['dropout'],
    CONFIG['target_activation_rate']
)
model = model.to(CONFIG['device'])

total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel Parameters: {total_params:,}")
print(f"Target Activation: {CONFIG['target_activation_rate']*100:.0f}%")
print(f"Target Energy Savings: {(1-CONFIG['target_activation_rate'])*100:.0f}%")

# ============================================================================
# Class-Weighted Loss
# ============================================================================

# Calculate class weights (inverse frequency)
class_counts = train_dataset.class_counts
total_samples = len(train_dataset)
class_weights = torch.tensor([
    total_samples / (len(CLASSES) * class_counts.get(i, 1))
    for i in range(len(CLASSES))
]).to(CONFIG['device'])

# Normalize weights
class_weights = class_weights / class_weights.sum() * len(CLASSES)

print(f"\nClass Weights (for balanced learning):")
for cls_name, weight in zip(CLASSES, class_weights):
    print(f"  {cls_name:12s}: {weight:.3f}")

criterion = nn.CrossEntropyLoss(weight=class_weights)

# ============================================================================
# Optimizer with Warmup and Cosine Schedule
# ============================================================================

optimizer = optim.AdamW(
    model.parameters(),
    lr=CONFIG['lr'],
    weight_decay=CONFIG['weight_decay']
)

# Mixed precision scaler
scaler = GradScaler() if CONFIG['mixed_precision'] and CONFIG['device'] == 'cuda' else None

def get_lr(epoch, warmup_epochs=5, max_epochs=100, base_lr=0.001):
    """Learning rate with warmup and cosine annealing"""
    if epoch < warmup_epochs:
        # Linear warmup
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine annealing
        progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
        return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

# ============================================================================
# Training Loop
# ============================================================================

metrics_history = []
best_val_acc = 0.0

print("\n" + "="*80)
print("TRAINING START")
print("="*80 + "\n")

for epoch in range(1, CONFIG['epochs'] + 1):
    # Update learning rate
    lr = get_lr(epoch, CONFIG['warmup_epochs'], CONFIG['epochs'], CONFIG['lr'])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Enable AST after warmup
    if epoch == CONFIG['ast_start_epoch']:
        model.enable_ast()

    # ========================================================================
    # Training Phase
    # ========================================================================
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']}")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])

        optimizer.zero_grad()

        # Mixed precision forward pass
        if scaler:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG['gradient_clip'])
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG['gradient_clip'])
            optimizer.step()

        # Apply AST every 100 batches
        if batch_idx % 100 == 0:
            activation_rate = model.apply_adaptive_sparsity()

        # Metrics
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*train_correct/train_total:.2f}%',
            'lr': f'{lr:.6f}'
        })

    # ========================================================================
    # Validation Phase
    # ========================================================================
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    # Per-class metrics
    class_correct = [0] * len(CLASSES)
    class_total = [0] * len(CLASSES)

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

            # Per-class accuracy
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1

    # Calculate metrics
    train_loss = train_loss / len(train_loader)
    val_loss = val_loss / len(val_loader)
    train_acc = 100. * train_correct / train_total
    val_acc = 100. * val_correct / val_total

    # Apply final AST and measure efficiency
    activation_rate = model.apply_adaptive_sparsity()
    energy_savings = (1.0 - activation_rate) * 100

    # Print epoch summary
    print(f"\nEpoch {epoch}/{CONFIG['epochs']}:")
    print(f"  LR: {lr:.6f}")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    print(f"  Per-class Val Accuracy:")
    for i, class_name in enumerate(CLASSES):
        if class_total[i] > 0:
            acc = 100. * class_correct[i] / class_total[i]
            print(f"    {class_name:12s}: {acc:5.1f}% ({class_correct[i]}/{class_total[i]})")
    print(f"  Activation: {activation_rate*100:.2f}% | Energy Savings: {energy_savings:.2f}%")

    # Save metrics
    metrics_history.append({
        'epoch': epoch,
        'timestamp': pd.Timestamp.now().timestamp(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_acc': val_acc / 100,
        'lr': lr,
        'activation_rate': activation_rate,
        'energy_savings': energy_savings,
        **{f'{cls}_acc': 100.*class_correct[i]/class_total[i] if class_total[i] > 0 else 0
           for i, cls in enumerate(CLASSES)}
    })

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        # Save only the inner EfficientNet model state_dict (compatible with inference app)
        torch.save(model.model.state_dict(),
                  Path(CONFIG['checkpoint_dir']) / 'best.pt')
        # Also save training checkpoint with metadata
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
            'per_class_acc': {
                cls: 100.*class_correct[i]/class_total[i] if class_total[i] > 0 else 0
                for i, cls in enumerate(CLASSES)
            }
        }, Path(CONFIG['checkpoint_dir']) / 'best_with_metadata.pt')
        print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")

    # Save checkpoint periodically
    if epoch % CONFIG['save_frequency'] == 0:
        torch.save(model.model.state_dict(),
                  Path(CONFIG['checkpoint_dir']) / f'checkpoint_epoch{epoch}.pt')

    print("-" * 80)

# ========================================================================
# Training Complete
# ========================================================================

# Save final model (only the inner EfficientNet model)
torch.save(model.model.state_dict(), Path(CONFIG['checkpoint_dir']) / 'final.pt')

# Save metrics
metrics_df = pd.DataFrame(metrics_history)
metrics_df.to_csv(Path(CONFIG['checkpoint_dir']) / 'metrics_optimized.csv', index=False)

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"\n📊 Final Results:")
print(f"  Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"  Average Energy Savings: {metrics_df['energy_savings'].mean():.2f}%")
print(f"  Average Activation Rate: {metrics_df['activation_rate'].mean()*100:.2f}%")

# Show best per-class performance
best_epoch_idx = metrics_df['val_acc'].idxmax()
best_metrics = metrics_df.iloc[best_epoch_idx]
print(f"\n  Best Per-Class Accuracy (Epoch {int(best_metrics['epoch'])}):")
for cls in CLASSES:
    print(f"    {cls:12s}: {best_metrics[f'{cls}_acc']:.2f}%")

print(f"\n📁 Saved Files:")
print(f"  - {CONFIG['checkpoint_dir']}/best.pt")
print(f"  - {CONFIG['checkpoint_dir']}/final.pt")
print(f"  - {CONFIG['checkpoint_dir']}/metrics_optimized.csv")

# Success check
if best_val_acc >= 90:
    print(f"\n🎉 SUCCESS! Achieved {best_val_acc:.2f}% accuracy (target: 90-95%)")
    print("   Ready for deployment!")
elif best_val_acc >= 88:
    print(f"\n✅ Good result! Achieved {best_val_acc:.2f}% accuracy")
    print("   Close to target - may need slight fine-tuning")
else:
    print(f"\n⚠️  Achieved {best_val_acc:.2f}% accuracy")
    print("   Below target - review data and hyperparameters")

print("\n" + "="*80)
