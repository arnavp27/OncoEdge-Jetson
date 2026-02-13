"""
Calibration data preparation for TensorRT INT8 quantization.

Collects and validates representative images for INT8 calibration.
Stores RAW JPEGs (not preprocessed tensors) to ensure single source of truth.
"""
import shutil
from PIL import Image
from pathlib import Path


def prepare_calibration_set(source_dirs, output_dir, target_count=300):
    """
    Collect and validate calibration images.

    IMPORTANT: Store RAW JPEGs, not preprocessed tensors.
    Calibrator will preprocess using same path as inference.

    Args:
        source_dirs: List of directories with source images
        output_dir: Output directory for calibration JPEGs
        target_count: Target number of images (recommended: 250-400)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    collected = 0
    for source_dir in source_dirs:
        source_path = Path(source_dir)
        if not source_path.exists():
            print(f"Warning: Source directory not found: {source_dir}")
            continue

        for img_path in source_path.glob('*.jpg'):
            # Validate image loads correctly
            try:
                img = Image.open(img_path)
                img.verify()

                # Reopen after verify (verify closes the file)
                img = Image.open(img_path)

                # Copy to calibration directory
                dest = output_path / f"calib_{collected:04d}.jpg"
                shutil.copy(img_path, dest)
                collected += 1

                if collected >= target_count:
                    break
            except Exception as e:
                print(f"Skipping {img_path}: {e}")

        if collected >= target_count:
            break

    print(f"\n[OK] Collected {collected} calibration images")
    print(f"  Output directory: {output_path}")

    if collected < 250:
        print(f"\n[WARNING] Need at least 250 images for robust calibration")
        print(f"  Currently have: {collected} images")
        print(f"  Recommendation: Add more source images to reach 250-400 images")
    elif collected < target_count:
        print(f"\n[NOTE] Target was {target_count} images, collected {collected}")
    else:
        print(f"  Target achieved: {target_count} images")


if __name__ == "__main__":
    # Example usage - modify source_dirs to point to your image datasets
    import argparse

    parser = argparse.ArgumentParser(description="Prepare calibration images for TensorRT INT8")
    parser.add_argument(
        '--source-dirs',
        nargs='+',
        required=True,
        help='List of source directories containing medical images'
    )
    parser.add_argument(
        '--output-dir',
        default='data/calibration_images',
        help='Output directory for calibration images (default: data/calibration_images)'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=300,
        help='Target number of images (recommended: 250-400, default: 300)'
    )

    args = parser.parse_args()

    prepare_calibration_set(
        source_dirs=args.source_dirs,
        output_dir=args.output_dir,
        target_count=args.count
    )
