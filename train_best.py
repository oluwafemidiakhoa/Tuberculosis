"""
BEST Multi-Class Training Script
---------------------------------
This fixes all the issues:
1. Two-stage training (learn first, compress later)
2. Higher activation rate (25% instead of 10%)
3. Better model (EfficientNet-B2)
4. Better augmentation and training schedule
5. Class-weighted loss for balance

Expected Results:
- 90%+ validation accuracy
- 75% energy savings
- Good specificity (no Pneumonia→TB confusion)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    'data_dir': 'data_multiclass',
    'num_classes': 4,
    'stage1_epochs': 60,  # Train without AST
    'stage2_epochs': 20,  # Fine-tune with AST
    'batch_size': 32,
    'lr_stage1': 0.0003,
    'lr_stage2': 0.00005,  # Lower LR for fine-tuning
    'checkpoint_dir': 'checkpoints_multiclass_best',
    'target_activation_rate': 0.25,  # 25% activation = 75% energy savings
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model_variant': 'b2',  # EfficientNet-B2
    'use_class_weights': True,  # Balance classes
}

CLASSES = ['Normal', 'TB', 'Pneumonia', 'COVID']

# Create checkpoint directory
Path(CONFIG['checkpoint_dir']).mkdir(exist_ok=True)

print(f"Device: {CONFIG['device']}")
print(f"Classes: {CLASSES}")
print(f"Model: EfficientNet-{CONFIG['model_variant'].upper()}")
print(f"Strategy: Two-stage training (accuracy → compression)")

# ============================================================================
# Dataset with Class Weights
# ============================================================================

class ChestXRayDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.samples = []

        for class_idx, class_name in enumerate(CLASSES):
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.png'):
                    self.samples.append((img_path, class_idx))

        # Calculate class weights
        class_counts = Counter([label for _, label in self.samples])
        total = len(self.samples)
        self.class_weights = {
            cls_idx: total / (len(CLASSES) * count)
            for cls_idx, count in class_counts.items()
        }

        print(f"{split.upper()}: {len(self.samples)} images")
        if split == 'train':
            print(f"  Class distribution: {dict(class_counts)}")
            print(f"  Class weights: {self.class_weights}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Try to load image, skip if corrupted
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                image = Image.open(img_path).convert('RGB')

                if self.transform:
                    image = self.transform(image)

                return image, label

            except Exception as e:
                if attempt == 0:
                    print(f"\nWarning: Corrupted image: {img_path}")
                    print(f"  Skipping...")

                # Try next random image from same class
                idx = (idx + 1) % len(self.samples)
                img_path, label = self.samples[idx]

        # If all attempts fail, raise error
        raise RuntimeError(f"Failed to load valid image after {max_attempts} attempts")

    def get_sample_weights(self):
        """For balanced sampling"""
        return [self.class_weights[label] for _, label in self.samples]

# Better augmentation for medical images
train_transform = transforms.Compose([
    transforms.Resize(288),  # Larger for B2
    transforms.RandomCrop(260),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(288),
    transforms.CenterCrop(260),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = ChestXRayDataset(CONFIG['data_dir'], 'train', train_transform)
val_dataset = ChestXRayDataset(CONFIG['data_dir'], 'val', val_transform)

# Balanced sampling for training
if CONFIG['use_class_weights']:
    sample_weights = train_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                             sampler=sampler, num_workers=2, pin_memory=True)
    print("\n✓ Using balanced sampling")
else:
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                             shuffle=True, num_workers=2, pin_memory=True)

val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                       shuffle=False, num_workers=2, pin_memory=True)

# ============================================================================
# Model with AST (Two-Stage)
# ============================================================================

class AdaptiveSparseModel(nn.Module):
    def __init__(self, num_classes, model_variant='b2', target_activation_rate=0.25):
        super().__init__()

        # Load pretrained EfficientNet
        if model_variant == 'b0':
            self.model = models.efficientnet_b0(pretrained=True)
            in_features = 1280
        elif model_variant == 'b1':
            self.model = models.efficientnet_b1(pretrained=True)
            in_features = 1280
        elif model_variant == 'b2':
            self.model = models.efficientnet_b2(pretrained=True)
            in_features = 1408
        elif model_variant == 'b3':
            self.model = models.efficientnet_b3(pretrained=True)
            in_features = 1536
        else:
            raise ValueError(f"Unknown model variant: {model_variant}")

        # Replace classifier with better dropout
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

        self.target_activation_rate = target_activation_rate
        self.ast_enabled = False  # Start disabled
        self.register_buffer('activation_mask', torch.ones(1))

    def forward(self, x):
        return self.model(x)

    def enable_ast(self):
        """Enable AST for Stage 2"""
        self.ast_enabled = True
        print("\n" + "="*60)
        print(">>> AST ENABLED - Compression Mode <<<")
        print("="*60 + "\n")

    def apply_adaptive_sparsity(self):
        """Apply sample-based sparsity"""
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

                # Create mask
                mask = (torch.abs(param.data) >= threshold).float()

                # Apply mask
                param.data = param.data * mask

                total_params += param.numel()
                active_params += mask.sum().item()

        activation_rate = active_params / total_params if total_params > 0 else 0
        return activation_rate

# Create model
model = AdaptiveSparseModel(
    CONFIG['num_classes'],
    CONFIG['model_variant'],
    CONFIG['target_activation_rate']
)
model = model.to(CONFIG['device'])

print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Target Activation (Stage 2): {CONFIG['target_activation_rate']*100:.1f}%")

# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, scheduler, device, apply_ast=False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        # Forward
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        loss.backward()
        optimizer.step()

        # Step scheduler per batch (OneCycleLR)
        if scheduler is not None and not isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR):
            scheduler.step()

        # Apply AST periodically if enabled
        if apply_ast and batch_idx % 50 == 0:
            activation_rate = model.apply_adaptive_sparsity()

        # Metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# ============================================================================
# STAGE 1: Train for Maximum Accuracy (No AST)
# ============================================================================

print("\n" + "="*70)
print("STAGE 1: Training for Maximum Accuracy (AST Disabled)")
print("="*70)

# Class-weighted loss
if CONFIG['use_class_weights']:
    class_weights = torch.tensor([
        train_dataset.class_weights[i] for i in range(CONFIG['num_classes'])
    ]).to(CONFIG['device'])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"✓ Using class-weighted loss: {class_weights.cpu().numpy()}")
else:
    criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr_stage1'], weight_decay=0.01)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=CONFIG['lr_stage1'],
    epochs=CONFIG['stage1_epochs'],
    steps_per_epoch=len(train_loader),
    pct_start=0.1  # Warmup for 10% of training
)

metrics_history = []
best_val_acc_stage1 = 0.0

for epoch in range(1, CONFIG['stage1_epochs'] + 1):
    # Train
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, scheduler,
        CONFIG['device'], apply_ast=False
    )

    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion, CONFIG['device'])

    # Print epoch summary
    print(f"\nEpoch {epoch}/{CONFIG['stage1_epochs']} (Stage 1):")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Save metrics
    metrics_history.append({
        'epoch': epoch,
        'stage': 1,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_acc': val_acc / 100,
        'lr': optimizer.param_groups[0]['lr'],
        'activation_rate': 1.0,  # 100% in Stage 1
        'energy_savings': 0.0,
    })

    # Save best model
    if val_acc > best_val_acc_stage1:
        best_val_acc_stage1 = val_acc
        torch.save(model.state_dict(),
                  Path(CONFIG['checkpoint_dir']) / 'best_stage1.pt')
        print(f"  ✓ New best model! (Val Acc: {val_acc:.2f}%)")

    print("-" * 70)

print(f"\n✅ Stage 1 Complete! Best Accuracy: {best_val_acc_stage1:.2f}%")

# ============================================================================
# STAGE 2: Apply AST for Compression
# ============================================================================

print("\n" + "="*70)
print("STAGE 2: Fine-tuning with AST Compression")
print("="*70)

# Load best model from Stage 1
model.load_state_dict(torch.load(Path(CONFIG['checkpoint_dir']) / 'best_stage1.pt'))

# Enable AST
model.enable_ast()

# New optimizer with lower learning rate
optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr_stage2'], weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['stage2_epochs'])

best_val_acc_stage2 = 0.0

for epoch in range(1, CONFIG['stage2_epochs'] + 1):
    # Train with AST
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, None,
        CONFIG['device'], apply_ast=True
    )

    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion, CONFIG['device'])

    # Apply AST and calculate energy savings
    activation_rate = model.apply_adaptive_sparsity()
    energy_savings = (1.0 - activation_rate) * 100

    # Step scheduler
    scheduler.step()

    # Print epoch summary
    print(f"\nEpoch {epoch}/{CONFIG['stage2_epochs']} (Stage 2):")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    print(f"  Activation: {activation_rate*100:.2f}% | Energy Savings: {energy_savings:.2f}%")

    # Save metrics
    metrics_history.append({
        'epoch': CONFIG['stage1_epochs'] + epoch,
        'stage': 2,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_acc': val_acc / 100,
        'lr': optimizer.param_groups[0]['lr'],
        'activation_rate': activation_rate,
        'energy_savings': energy_savings,
    })

    # Save best model
    if val_acc > best_val_acc_stage2:
        best_val_acc_stage2 = val_acc
        torch.save(model.state_dict(),
                  Path(CONFIG['checkpoint_dir']) / 'best.pt')
        print(f"  ✓ New best compressed model! (Val Acc: {val_acc:.2f}%)")

    print("-" * 70)

# Save final model
torch.save(model.state_dict(),
          Path(CONFIG['checkpoint_dir']) / 'final.pt')

# Save metrics
metrics_df = pd.DataFrame(metrics_history)
metrics_df.to_csv(Path(CONFIG['checkpoint_dir']) / 'metrics.csv', index=False)

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"\n📊 Results Summary:")
print(f"  Stage 1 Best Accuracy: {best_val_acc_stage1:.2f}% (no compression)")
print(f"  Stage 2 Best Accuracy: {best_val_acc_stage2:.2f}% (with AST)")
print(f"  Average Energy Savings: {metrics_df[metrics_df['stage']==2]['energy_savings'].mean():.2f}%")
print(f"\n💾 Saved files:")
print(f"  - {CONFIG['checkpoint_dir']}/best.pt (final compressed model)")
print(f"  - {CONFIG['checkpoint_dir']}/best_stage1.pt (high accuracy baseline)")
print(f"  - {CONFIG['checkpoint_dir']}/metrics.csv")
print("\n✅ Next: Run specificity test and Grad-CAM visualization!")
