"""
Script to identify and remove corrupted images from the dataset.
This will scan all image files and remove any that cannot be opened by PIL.
"""

import os
from PIL import Image
from pathlib import Path
import shutil

def verify_and_fix_images(data_dir, backup=True):
    """
    Scan all images in the directory and remove corrupted ones.

    Args:
        data_dir: Root directory containing the dataset
        backup: Whether to backup corrupted files instead of deleting
    """
    corrupted_files = []
    total_files = 0
    backup_dir = Path(data_dir).parent / f"{Path(data_dir).name}_corrupted_backup"

    if backup and not backup_dir.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning images in: {data_dir}")
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
                    print(f"  Error: {str(e)}")
                    corrupted_files.append(file_path)

                    if backup:
                        # Backup the corrupted file
                        rel_path = os.path.relpath(file_path, data_dir)
                        backup_path = backup_dir / rel_path
                        backup_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(file_path, backup_path)
                        print(f"  → Moved to: {backup_path}")
                    else:
                        # Delete the corrupted file
                        os.remove(file_path)
                        print(f"  → Deleted")

                # Progress indicator
                if total_files % 100 == 0:
                    print(f"Processed {total_files} files... ({len(corrupted_files)} corrupted)")

    print("\n" + "=" * 80)
    print(f"SUMMARY:")
    print(f"  Total images scanned: {total_files}")
    print(f"  Corrupted images found: {len(corrupted_files)}")
    print(f"  Valid images: {total_files - len(corrupted_files)}")

    if backup and corrupted_files:
        print(f"  Corrupted files backed up to: {backup_dir}")

    return corrupted_files


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fix corrupted images in dataset')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--no-backup', action='store_true',
                        help='Delete corrupted files instead of backing up')

    args = parser.parse_args()

    corrupted = verify_and_fix_images(args.data_dir, backup=not args.no_backup)

    if corrupted:
        print("\nCorrupted files list:")
        for f in corrupted[:20]:  # Show first 20
            print(f"  - {f}")
        if len(corrupted) > 20:
            print(f"  ... and {len(corrupted) - 20} more")
