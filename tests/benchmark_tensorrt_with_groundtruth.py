#!/usr/bin/env python3
"""
Benchmark TensorRT INT8 vs PyTorch FP32 with Ground Truth Validation.

Compares:
1. Performance: TensorRT INT8 vs PyTorch FP32 latency
2. Consistency: TensorRT vs PyTorch predictions (should agree)
3. Accuracy: Both models vs YOLO ground truth labels

Uses validation set with ground truth labels from YOLO training.
"""
import time
import numpy as np
from pathlib import Path
from PIL import Image
import sys
import json
from datetime import datetime
from collections import defaultdict

# YOLO class mapping (from data.yaml)
YOLO_CLASS_NAMES = {
    0: "OCA",
    1: "OPMD",
    2: "Benign"
}

def parse_ground_truth(label_file_path):
    """
    Parse ground truth from YOLO label file.

    Args:
        label_file_path: Path to YOLO label file

    Returns:
        list: List of ground truth class names present in image
              Empty list means "Normal" (no lesions)

    Example:
        File contains:
          2 0.xxx 0.xxx ...
          2 0.xxx 0.xxx ...
        Returns: ["Benign"]

        Empty file returns: ["Normal"]
    """
    label_path = Path(label_file_path)

    if not label_path.exists():
        # No label file = Normal
        return ["Normal"]

    # Read file
    content = label_path.read_text().strip()

    if not content:
        # Empty file = Normal (no lesions)
        return ["Normal"]

    # Parse class IDs from each line (first number)
    class_ids = set()
    for line in content.split('\n'):
        line = line.strip()
        if line:
            # First number is class ID
            class_id = int(line.split()[0])
            class_ids.add(class_id)

    # Convert to class names
    ground_truth = [YOLO_CLASS_NAMES[cid] for cid in sorted(class_ids)]

    return ground_truth


def benchmark_with_groundtruth(
    images_dir='data/oral_cancer_seg/images/val',
    labels_dir='data/oral_cancer_seg/labels/val',
    num_warmup=10,
    num_iterations=None,  # None = use all validation images
    output_file=None
):
    """
    Benchmark TensorRT vs PyTorch with ground truth validation.

    Args:
        images_dir: Directory with validation images
        labels_dir: Directory with YOLO ground truth labels
        num_warmup: Number of warmup iterations
        num_iterations: Number of images to test (None = all)
        output_file: Output JSON file path

    Returns:
        int: Return code (0 = pass, 1 = fail)
    """
    print("=" * 70)
    print("TensorRT INT8 vs PyTorch FP32 Benchmark with Ground Truth")
    print("=" * 70)

    # Check if running on Jetson
    try:
        with open('/etc/nv_tegra_release', 'r') as f:
            tegra_info = f.read()
        print(f"[OK] Running on Jetson: {tegra_info.strip().split()[0]}")
    except:
        print("[WARNING] Not running on Jetson - benchmarks may not be representative")

    # Load models
    print("\n[1/5] Loading models...")
    print("  Loading TensorRT INT8 model...")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.models.biomedclip_tensorrt import BiomedCLIPTensorRT
    trt_model = BiomedCLIPTensorRT()

    print("  Loading PyTorch FP32 model...")
    from src.models.biomedclip_classifier import BiomedCLIPClassifier
    try:
        torch_model = BiomedCLIPClassifier(device='cuda')
    except:
        print("[WARNING] Could not load PyTorch model on CUDA, trying CPU...")
        torch_model = BiomedCLIPClassifier(device='cpu')

    # Load validation images
    print(f"\n[2/5] Loading validation images from: {images_dir}")
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    # Get all image files
    image_paths = list(images_dir.glob('*.jpg'))
    if len(image_paths) == 0:
        image_paths = list(images_dir.glob('*.jpeg'))

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in {images_dir}")

    # Limit number of iterations if specified
    if num_iterations is not None:
        image_paths = image_paths[:num_iterations]

    print(f"[OK] Found {len(image_paths)} validation images")

    # Warmup
    print(f"\n[3/5] Warming up TensorRT engine ({num_warmup} iterations)...")
    for i in range(min(num_warmup, len(image_paths))):
        img = Image.open(image_paths[i])
        _ = trt_model.classify(img)
    print("[OK] Warmup complete")

    # Benchmark
    print(f"\n[4/5] Running benchmark on {len(image_paths)} images...")

    trt_latencies = []
    torch_latencies = []

    # Ground truth tracking
    ground_truths = []
    trt_predictions = []
    torch_predictions = []

    # Detailed results for analysis
    detailed_results = []

    for i, img_path in enumerate(image_paths):
        # Get corresponding label file
        label_path = labels_dir / f"{img_path.stem}.txt"
        ground_truth = parse_ground_truth(label_path)
        ground_truths.append(ground_truth)

        # Load image
        img = Image.open(img_path)

        # TensorRT inference
        start = time.time()
        trt_result = trt_model.classify(img)
        trt_latency = (time.time() - start) * 1000  # ms
        trt_latencies.append(trt_latency)
        trt_predictions.append(trt_result['class'])

        # PyTorch inference
        start = time.time()
        torch_result = torch_model.classify(img)
        torch_latency = (time.time() - start) * 1000  # ms
        torch_latencies.append(torch_latency)
        torch_predictions.append(torch_result['class'])

        # Check if predictions match ground truth
        trt_correct = trt_result['class'] in ground_truth
        torch_correct = torch_result['class'] in ground_truth
        predictions_match = trt_result['class'] == torch_result['class']

        # Store detailed result
        detailed_results.append({
            'image': img_path.name,
            'ground_truth': ground_truth,
            'trt_prediction': trt_result['class'],
            'trt_confidence': float(trt_result['max_score']),
            'trt_correct': trt_correct,
            'torch_prediction': torch_result['class'],
            'torch_confidence': float(torch_result['max_score']),
            'torch_correct': torch_correct,
            'predictions_match': predictions_match
        })

        # Progress
        if (i + 1) % 20 == 0 or (i + 1) == len(image_paths):
            print(f"  Progress: {i+1}/{len(image_paths)}")

    trt_latencies = np.array(trt_latencies)
    torch_latencies = np.array(torch_latencies)

    # Compute accuracy metrics
    print("\n[5/5] Computing accuracy metrics...")

    trt_correct_count = sum(1 for r in detailed_results if r['trt_correct'])
    torch_correct_count = sum(1 for r in detailed_results if r['torch_correct'])
    predictions_match_count = sum(1 for r in detailed_results if r['predictions_match'])

    trt_accuracy = trt_correct_count / len(detailed_results)
    torch_accuracy = torch_correct_count / len(detailed_results)
    consistency = predictions_match_count / len(detailed_results)

    # Per-class accuracy
    per_class_metrics = defaultdict(lambda: {'total': 0, 'trt_correct': 0, 'torch_correct': 0})

    for result in detailed_results:
        for gt_class in result['ground_truth']:
            per_class_metrics[gt_class]['total'] += 1
            if result['trt_prediction'] == gt_class:
                per_class_metrics[gt_class]['trt_correct'] += 1
            if result['torch_prediction'] == gt_class:
                per_class_metrics[gt_class]['torch_correct'] += 1

    # Print results
    print()
    print("=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)
    print()
    print("TensorRT INT8:")
    print(f"  Mean latency:    {trt_latencies.mean():.2f} ms")
    print(f"  Median latency:  {np.median(trt_latencies):.2f} ms")
    print(f"  Std Dev:         {trt_latencies.std():.2f} ms")
    print(f"  P95:             {np.percentile(trt_latencies, 95):.2f} ms")
    print(f"  P99:             {np.percentile(trt_latencies, 99):.2f} ms")
    print(f"  Throughput:      {1000/trt_latencies.mean():.2f} FPS")
    print()
    print("PyTorch FP32:")
    print(f"  Mean latency:    {torch_latencies.mean():.2f} ms")
    print(f"  Median latency:  {np.median(torch_latencies):.2f} ms")
    print(f"  Std Dev:         {torch_latencies.std():.2f} ms")
    print(f"  P95:             {np.percentile(torch_latencies, 95):.2f} ms")
    print(f"  P99:             {np.percentile(torch_latencies, 99):.2f} ms")
    print(f"  Throughput:      {1000/torch_latencies.mean():.2f} FPS")
    print()
    print("Speedup:")
    speedup = torch_latencies.mean() / trt_latencies.mean()
    print(f"  TensorRT is {speedup:.2f}x faster than PyTorch")
    print()

    print("=" * 70)
    print("ACCURACY vs GROUND TRUTH")
    print("=" * 70)
    print()
    print(f"TensorRT INT8 Accuracy:  {trt_accuracy*100:.2f}% ({trt_correct_count}/{len(detailed_results)})")
    print(f"PyTorch FP32 Accuracy:   {torch_accuracy*100:.2f}% ({torch_correct_count}/{len(detailed_results)})")
    print(f"Model Consistency:       {consistency*100:.2f}% (predictions match)")
    print()

    print("Per-Class Accuracy:")
    print(f"{'Class':<10} {'Count':<10} {'TensorRT':<15} {'PyTorch':<15}")
    print("-" * 50)
    for class_name in sorted(per_class_metrics.keys()):
        metrics = per_class_metrics[class_name]
        trt_class_acc = metrics['trt_correct'] / metrics['total'] * 100
        torch_class_acc = metrics['torch_correct'] / metrics['total'] * 100
        print(f"{class_name:<10} {metrics['total']:<10} {trt_class_acc:.1f}%{'':<10} {torch_class_acc:.1f}%")
    print()

    # Validation checks
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    # Check 1: TensorRT should be faster
    if speedup >= 1.2:
        print(f"[OK] TensorRT is {speedup:.2f}x faster (target: >1.2x)")
    else:
        print(f"[WARNING] TensorRT speedup is only {speedup:.2f}x (target: >1.2x)")

    # Check 2: Accuracy should be acceptable
    accuracy_threshold = 0.70  # 70% accuracy target
    if trt_accuracy >= accuracy_threshold:
        print(f"[OK] TensorRT accuracy: {trt_accuracy*100:.2f}% (target: >{accuracy_threshold*100:.0f}%)")
    else:
        print(f"[WARNING] TensorRT accuracy: {trt_accuracy*100:.2f}% (target: >{accuracy_threshold*100:.0f}%)")

    # Check 3: Models should be consistent
    if consistency >= 0.85:
        print(f"[OK] Model consistency: {consistency*100:.2f}% (target: >85%)")
    else:
        print(f"[WARNING] Model consistency: {consistency*100:.2f}% (target: >85%)")

    # Overall pass/fail
    overall_pass = (speedup >= 1.2 and trt_accuracy >= accuracy_threshold and consistency >= 0.85)
    return_code = 0 if overall_pass else 1

    print()
    if overall_pass:
        print("[OK] All validation checks passed!")
    else:
        print("[WARNING] Some validation checks failed - see above")

    print("=" * 70)

    # Save results
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"benchmark_groundtruth_{timestamp}.json"

    results = {
        "timestamp": datetime.now().isoformat(),
        "test_config": {
            "images_dir": str(images_dir),
            "labels_dir": str(labels_dir),
            "num_images": len(image_paths),
            "num_warmup": num_warmup
        },
        "performance": {
            "tensorrt": {
                "mean_ms": float(trt_latencies.mean()),
                "median_ms": float(np.median(trt_latencies)),
                "std_dev_ms": float(trt_latencies.std()),
                "p95_ms": float(np.percentile(trt_latencies, 95)),
                "p99_ms": float(np.percentile(trt_latencies, 99)),
                "throughput_fps": float(1000 / trt_latencies.mean())
            },
            "pytorch": {
                "mean_ms": float(torch_latencies.mean()),
                "median_ms": float(np.median(torch_latencies)),
                "std_dev_ms": float(torch_latencies.std()),
                "p95_ms": float(np.percentile(torch_latencies, 95)),
                "p99_ms": float(np.percentile(torch_latencies, 99)),
                "throughput_fps": float(1000 / torch_latencies.mean())
            },
            "speedup": float(speedup)
        },
        "accuracy": {
            "tensorrt_accuracy": float(trt_accuracy),
            "pytorch_accuracy": float(torch_accuracy),
            "model_consistency": float(consistency),
            "tensorrt_correct": trt_correct_count,
            "pytorch_correct": torch_correct_count,
            "total_images": len(detailed_results),
            "per_class": {
                class_name: {
                    "total": metrics['total'],
                    "tensorrt_correct": metrics['trt_correct'],
                    "pytorch_correct": metrics['torch_correct'],
                    "tensorrt_accuracy": float(metrics['trt_correct'] / metrics['total']),
                    "pytorch_accuracy": float(metrics['torch_correct'] / metrics['total'])
                }
                for class_name, metrics in per_class_metrics.items()
            }
        },
        "validation": {
            "speedup_pass": bool(speedup >= 1.2),
            "accuracy_pass": bool(trt_accuracy >= accuracy_threshold),
            "consistency_pass": bool(consistency >= 0.85),
            "overall_pass": overall_pass
        },
        "detailed_results": detailed_results[:50],  # First 50 results for review
        "return_code": return_code
    }

    # Try to get Jetson info
    try:
        with open('/etc/nv_tegra_release', 'r') as f:
            results["system_info"] = {"jetson_version": f.read().strip()}
    except:
        results["system_info"] = {"jetson_version": "Not running on Jetson"}

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Results saved to: {output_file}")

    return return_code


def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Benchmark TensorRT vs PyTorch with ground truth validation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--images-dir',
        default='data/oral_cancer_seg/images/val',
        help='Validation images directory'
    )
    parser.add_argument(
        '--labels-dir',
        default='data/oral_cancer_seg/labels/val',
        help='Ground truth labels directory'
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=10,
        help='Number of warmup iterations'
    )
    parser.add_argument(
        '--num-images',
        type=int,
        default=None,
        help='Number of images to test (None = all)'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output JSON file path'
    )

    args = parser.parse_args()

    try:
        return_code = benchmark_with_groundtruth(
            images_dir=args.images_dir,
            labels_dir=args.labels_dir,
            num_warmup=args.warmup,
            num_iterations=args.num_images,
            output_file=args.output
        )
        sys.exit(return_code)
    except Exception as e:
        print(f"\n[ERROR] Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
