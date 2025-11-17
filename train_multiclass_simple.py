"""
Simple Multi-Class Training Script with AST
Trains 4-class chest X-ray classifier (Normal, TB, Pneumonia, COVID)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    'data_dir': 'data_multiclass',
    'num_classes': 4,
    'epochs': 50,
    'batch_size': 32,
    'lr': 0.0003,
    'checkpoint_dir': 'checkpoints_multiclass',
    'target_activation_rate': 0.10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

CLASSES = ['Normal', 'TB', 'Pneumonia', 'COVID']

# Create checkpoint directory
Path(CONFIG['checkpoint_dir']).mkdir(exist_ok=True)

print(f"Device: {CONFIG['device']}")
print(f"Classes: {CLASSES}")

# ============================================================================
# Dataset
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

        print(f"{split.upper()}: {len(self.samples)} images")

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
                    print(f"\nWarning: Corrupted image found: {img_path}")
                    print(f"  Error: {e}")
                    print(f"  Skipping to next image...")

                # Try next random image from same class
                idx = (idx + 1) % len(self.samples)
                img_path, label = self.samples[idx]

        # If all attempts fail, raise error
        raise RuntimeError(f"Failed to load valid image after {max_attempts} attempts")

# Transforms
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = ChestXRayDataset(CONFIG['data_dir'], 'train', train_transform)
val_dataset = ChestXRayDataset(CONFIG['data_dir'], 'val', val_transform)

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                         shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                       shuffle=False, num_workers=2, pin_memory=True)

# ============================================================================
# Model with AST
# ============================================================================

class AdaptiveSparseModel(nn.Module):
    def __init__(self, num_classes, target_activation_rate=0.10):
        super().__init__()

        # Load pretrained EfficientNet with modern API
        from torchvision.models import EfficientNet_B0_Weights
        self.model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Replace classifier
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

        self.target_activation_rate = target_activation_rate
        self.register_buffer('activation_mask', torch.ones(1))

    def forward(self, x):
        return self.model(x)

    def apply_adaptive_sparsity(self):
        """Apply sample-based sparsity"""
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
model = AdaptiveSparseModel(CONFIG['num_classes'], CONFIG['target_activation_rate'])
model = model.to(CONFIG['device'])

print(f"\nModel: EfficientNet-B0 with {CONFIG['num_classes']} classes")
print(f"Target Activation Rate: {CONFIG['target_activation_rate']*100:.1f}%")

# ============================================================================
# Training
# ============================================================================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])

# Metrics tracking
metrics_history = []
best_val_acc = 0.0

print("\n" + "="*60)
print("TRAINING START")
print("="*60)

for epoch in range(1, CONFIG['epochs'] + 1):
    # Training phase
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']}")
    for images, labels in pbar:
        images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])

        # Forward
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        loss.backward()
        optimizer.step()

        # Apply AST every few batches
        if train_total % 100 == 0:
            activation_rate = model.apply_adaptive_sparsity()

        # Metrics
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*train_correct/train_total:.2f}%'
        })

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    # Calculate metrics
    train_loss = train_loss / len(train_loader)
    val_loss = val_loss / len(val_loader)
    train_acc = 100. * train_correct / train_total
    val_acc = 100. * val_correct / val_total

    # Apply AST and calculate energy savings
    activation_rate = model.apply_adaptive_sparsity()
    energy_savings = (1.0 - activation_rate) * 100

    # Learning rate step
    scheduler.step()

    # Print epoch summary
    print(f"\nEpoch {epoch}/{CONFIG['epochs']}:")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    print(f"  Activation: {activation_rate*100:.2f}% | Energy Savings: {energy_savings:.2f}%")

    # Save metrics
    metrics_history.append({
        'epoch': epoch,
        'timestamp': pd.Timestamp.now().timestamp(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_acc': val_acc / 100,  # Store as fraction
        'lr': optimizer.param_groups[0]['lr'],
        'activation_rate': activation_rate,
        'energy_savings': energy_savings,
        'samples_processed': train_total,
        'total_samples': len(train_dataset)
    })

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(),
                  Path(CONFIG['checkpoint_dir']) / 'best.pt')
        print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")

    # Save checkpoint every 10 epochs
    if epoch % 10 == 0:
        torch.save(model.state_dict(),
                  Path(CONFIG['checkpoint_dir']) / f'checkpoint_epoch{epoch}.pt')

    print("-" * 60)

# Save final model
torch.save(model.state_dict(),
          Path(CONFIG['checkpoint_dir']) / 'final.pt')

# Save metrics
metrics_df = pd.DataFrame(metrics_history)
metrics_df.to_csv(Path(CONFIG['checkpoint_dir']) / 'metrics_ast.csv', index=False)

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")
print(f"Average Energy Savings: {metrics_df['energy_savings'].mean():.2f}%")
print(f"Average Activation Rate: {metrics_df['activation_rate'].mean()*100:.2f}%")
print(f"\nSaved files:")
print(f"  - {CONFIG['checkpoint_dir']}/best.pt")
print(f"  - {CONFIG['checkpoint_dir']}/final.pt")
print(f"  - {CONFIG['checkpoint_dir']}/metrics_ast.csv")
print("\nNext: Run visualization cells in notebook to create charts!")
