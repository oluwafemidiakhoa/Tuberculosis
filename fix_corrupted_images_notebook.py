"""
Notebook-friendly script to identify and remove corrupted images.
Run this in your Jupyter/Colab environment where PIL/Pillow is available.
"""

import os
from PIL import Image
from pathlib import Path
import shutil

def verify_and_fix_images(data_dir='data_multiclass', backup=True):
    """
    Scan all images and remove corrupted ones.

    Args:
        data_dir: Root directory containing the dataset
        backup: Whether to backup corrupted files
    """
    corrupted_files = []
    total_files = 0
    backup_dir = f"{data_dir}_corrupted_backup"

    if backup and not os.path.exists(backup_dir):
        os.makedirs(backup_dir, exist_ok=True)

    print(f"🔍 Scanning images in: {data_dir}")
    print("=" * 80)

    # Walk through all directories
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                total_files += 1
                file_path = os.path.join(root, file)

                try:
                    # Try to open and verify the image
                    with Image.open(file_path) as img:
                        img.verify()

                    # Try to actually load the image data
                    with Image.open(file_path) as img:
                        img.load()

                except Exception as e:
                    print(f"✗ CORRUPTED: {file_path}")
                    corrupted_files.append(file_path)

                    if backup:
                        # Backup the corrupted file
                        rel_path = os.path.relpath(file_path, data_dir)
                        backup_path = os.path.join(backup_dir, rel_path)
                        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                        shutil.move(file_path, backup_path)
                    else:
                        # Delete the corrupted file
                        os.remove(file_path)

                # Progress indicator
                if total_files % 500 == 0:
                    print(f"📊 Processed {total_files} files... ({len(corrupted_files)} corrupted)")

    print("\n" + "=" * 80)
    print(f"✅ SUMMARY:")
    print(f"   Total images scanned: {total_files}")
    print(f"   Corrupted images found: {len(corrupted_files)}")
    print(f"   Valid images remaining: {total_files - len(corrupted_files)}")
    print(f"   Success rate: {((total_files - len(corrupted_files)) / total_files * 100):.2f}%")

    if backup and corrupted_files:
        print(f"   Corrupted files backed up to: {backup_dir}/")

    return corrupted_files, total_files


# Run the cleanup
if __name__ == "__main__":
    print("🧹 Starting image cleanup process...")
    print()

    corrupted, total = verify_and_fix_images('data_multiclass', backup=True)

    if corrupted:
        print(f"\n📋 First 20 corrupted files:")
        for f in corrupted[:20]:
            print(f"   - {f}")
        if len(corrupted) > 20:
            print(f"   ... and {len(corrupted) - 20} more")

    print("\n✅ Cleanup complete! You can now retrain your model.")
    print("   Run: python train_multiclass_simple.py")
