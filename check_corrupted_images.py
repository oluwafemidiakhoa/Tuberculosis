"""
Utility script to scan and identify corrupted images in the dataset
"""
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import sys

def check_image(img_path):
    """Check if an image can be opened and converted to RGB"""
    try:
        with Image.open(img_path) as img:
            img.convert('RGB')
        return True, None
    except Exception as e:
        return False, str(e)

def scan_directory(data_dir):
    """Scan all images in the dataset directory"""
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"Error: Directory '{data_dir}' does not exist!")
        return

    # Find all image files
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    all_images = []
    for ext in image_extensions:
        all_images.extend(data_path.rglob(ext))

    print(f"Scanning {len(all_images)} images in '{data_dir}'...\n")

    corrupted = []

    for img_path in tqdm(all_images, desc="Checking images"):
        is_valid, error = check_image(img_path)
        if not is_valid:
            corrupted.append((str(img_path), error))

    # Report results
    print(f"\n{'='*60}")
    print(f"SCAN RESULTS")
    print(f"{'='*60}")
    print(f"Total images: {len(all_images)}")
    print(f"Valid images: {len(all_images) - len(corrupted)}")
    print(f"Corrupted images: {len(corrupted)}")

    if corrupted:
        print(f"\n{'='*60}")
        print(f"CORRUPTED IMAGES FOUND:")
        print(f"{'='*60}")
        for img_path, error in corrupted:
            print(f"\n  Path: {img_path}")
            print(f"  Error: {error}")

        print(f"\n{'='*60}")
        print(f"RECOMMENDATIONS:")
        print(f"{'='*60}")
        print(f"1. Remove these corrupted files manually")
        print(f"2. Or re-download/replace them from the source")
        print(f"3. The training script has been updated to skip these automatically")

        return corrupted
    else:
        print(f"\n✓ All images are valid!")
        return []

if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data_multiclass"
    corrupted_files = scan_directory(data_dir)

    # Exit with error code if corrupted files found
    sys.exit(len(corrupted_files))
