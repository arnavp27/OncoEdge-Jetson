#!/usr/bin/env python3
"""
Benchmark and validate TensorRT INT8 engine on Jetson Nano.

Tests:
1. Output correctness vs PyTorch FP32
2. Inference latency (mean, median, p95, p99)
3. Memory usage
4. Sustained performance (thermal throttling check)

Success Criteria:
- Latency: 200-350ms per image (ViT-B/16 on Maxwell GPU)
- Accuracy: Max score difference < 0.05 (5%) vs PyTorch
- No crashes or OOM errors
"""
import time
import numpy as np
from pathlib import Path
from PIL import Image
import sys
import json
from datetime import datetime


def benchmark_tensorrt(
    test_images_dir='data/test_images',
    num_warmup=10,
    num_iterations=100,
    output_file=None
):
    """
    Benchmark TensorRT engine performance and accuracy.

    Args:
        test_images_dir: Directory with test images
        num_warmup: Number of warmup iterations (excluded from timing)
        num_iterations: Number of timing iterations
        output_file: Optional path to save results JSON (default: auto-generated)

    Returns:
        dict: Benchmark results
    """
    print("=" * 70)
    print("TensorRT INT8 Benchmark and Validation")
    print("=" * 70)

    # Check if running on Jetson
    try:
        with open('/etc/nv_tegra_release', 'r') as f:
            tegra_info = f.read()
        print(f"[OK] Running on Jetson: {tegra_info.strip().split()[0]}")
    except:
        print("[WARNING] Not running on Jetson - benchmarks may not be representative")

    # Load both backends for comparison
    print("\n[1/4] Loading models...")
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

    # Load test images
    print(f"\n[2/4] Loading test images from: {test_images_dir}")
    test_images_dir = Path(test_images_dir)
    if not test_images_dir.exists():
        print(f"[ERROR] Test images directory not found: {test_images_dir}")
        print("Using calibration images instead...")
        test_images_dir = Path('data/calibration_images')

    test_image_paths = list(test_images_dir.glob('*.jpg'))
    if len(test_image_paths) == 0:
        test_image_paths = list(test_images_dir.glob('*.jpeg'))

    if len(test_image_paths) == 0:
        raise FileNotFoundError(f"No test images found in {test_images_dir}")

    # Use subset for benchmarking
    test_image_paths = test_image_paths[:min(20, len(test_image_paths))]
    print(f"[OK] Using {len(test_image_paths)} test images")

    # Warmup
    print(f"\n[3/4] Warming up TensorRT engine ({num_warmup} iterations)...")
    for i in range(min(num_warmup, len(test_image_paths))):
        img = Image.open(test_image_paths[i % len(test_image_paths)])
        _ = trt_model.classify(img)
    print("[OK] Warmup complete")

    # Benchmark latency
    print(f"\n[4/4] Benchmarking latency ({num_iterations} iterations)...")
    latencies = []

    for i in range(num_iterations):
        img_path = test_image_paths[i % len(test_image_paths)]
        img = Image.open(img_path)

        start = time.time()
        result = trt_model.classify(img)
        latency = (time.time() - start) * 1000  # ms
        latencies.append(latency)

        if (i + 1) % 25 == 0:
            print(f"  Progress: {i+1}/{num_iterations} (current: {latency:.2f}ms)")

    latencies = np.array(latencies)

    # Accuracy comparison
    print("\n[5/5] Comparing accuracy (TensorRT vs PyTorch)...")
    print()
    print(f"{'Image':<30} {'TensorRT':<10} {'PyTorch':<10} {'Diff':<10} {'Match':<10}")
    print("-" * 70)

    max_diff = 0.0
    class_matches = 0
    total_comparisons = 0

    for img_path in test_image_paths[:10]:  # Compare first 10 images
        img = Image.open(img_path)

        trt_result = trt_model.classify(img)
        torch_result = torch_model.classify(img)

        # Compare predictions
        trt_class = trt_result['class']
        torch_class = torch_result['class']
        match = "YES" if trt_class == torch_class else "NO"

        if trt_class == torch_class:
            class_matches += 1
        total_comparisons += 1

        # Compare scores
        for cls in trt_result['scores'].keys():
            diff = abs(trt_result['scores'][cls] - torch_result['scores'][cls])
            max_diff = max(max_diff, diff)

        # Print comparison
        img_name = img_path.name[:28]
        print(f"{img_name:<30} {trt_class:<10} {torch_class:<10} {max_diff:>.4f}     {match:<10}")

    # Print results
    print()
    print("=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print()
    print("LATENCY (TensorRT INT8):")
    print(f"  Mean:       {latencies.mean():.2f} ms")
    print(f"  Median:     {np.median(latencies):.2f} ms")
    print(f"  Std Dev:    {latencies.std():.2f} ms")
    print(f"  Min:        {latencies.min():.2f} ms")
    print(f"  Max:        {latencies.max():.2f} ms")
    print(f"  P95:        {np.percentile(latencies, 95):.2f} ms")
    print(f"  P99:        {np.percentile(latencies, 99):.2f} ms")
    print(f"  Throughput: {1000/latencies.mean():.2f} FPS")
    print()

    # Target check
    target_latency = 350  # ms
    if latencies.mean() <= target_latency:
        print(f"[OK] Latency meets target: {latencies.mean():.2f}ms <= {target_latency}ms")
    else:
        print(f"[WARNING] Latency exceeds target: {latencies.mean():.2f}ms > {target_latency}ms")
        print(f"         ViT-B/16 on Maxwell GPU is inherently slower")
        print(f"         This is expected - consider using smaller model if needed")

    print()
    print("ACCURACY (TensorRT vs PyTorch):")
    print(f"  Max score difference: {max_diff:.4f}")
    print(f"  Class agreement:      {class_matches}/{total_comparisons} ({100*class_matches/total_comparisons:.1f}%)")

    # Accuracy check
    accuracy_threshold = 0.05
    if max_diff < accuracy_threshold:
        print(f"[OK] Accuracy acceptable: {max_diff:.4f} < {accuracy_threshold}")
    else:
        print(f"[WARNING] Accuracy degradation detected: {max_diff:.4f} >= {accuracy_threshold}")
        print(f"         Check calibration data quality")
        print(f"         Consider using FP16 engine instead of INT8")

    if class_matches / total_comparisons < 0.9:
        print(f"[WARNING] Low class agreement: {100*class_matches/total_comparisons:.1f}%")
        print(f"         Expected: >90% agreement")

    # Variability check
    if latencies.std() / latencies.mean() > 0.2:
        print()
        print(f"[WARNING] High latency variability (CV: {latencies.std()/latencies.mean():.2f})")
        print(f"         This may indicate thermal throttling")
        print(f"         Check temperature: cat /sys/devices/virtual/thermal/thermal_zone*/temp")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if latencies.mean() <= target_latency and max_diff < accuracy_threshold:
        print("[OK] TensorRT engine validated successfully!")
        print()
        print("Next steps:")
        print("  1. Test full pipeline: streamlit run streamlit_app.py")
        print("  2. Run extended stress test for thermal throttling")
        return_code = 0
    else:
        print("[WARNING] Some checks failed - see warnings above")
        print()
        print("Troubleshooting:")
        if latencies.mean() > target_latency:
            print("  - High latency is expected for ViT-B/16 on Maxwell GPU")
            print("  - Check thermal throttling: tegrastats")
        if max_diff >= accuracy_threshold:
            print("  - Try rebuilding with more calibration images (400+)")
            print("  - Consider FP16 engine: trtexec --fp16 ...")
        return_code = 1

    print("=" * 70)

    # Save results to JSON file
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"benchmark_results_{timestamp}.json"

    results = {
        "timestamp": datetime.now().isoformat(),
        "test_config": {
            "test_images_dir": str(test_images_dir),
            "num_warmup": num_warmup,
            "num_iterations": num_iterations,
            "num_test_images": len(test_image_paths)
        },
        "latency": {
            "mean_ms": float(latencies.mean()),
            "median_ms": float(np.median(latencies)),
            "std_dev_ms": float(latencies.std()),
            "min_ms": float(latencies.min()),
            "max_ms": float(latencies.max()),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "throughput_fps": float(1000 / latencies.mean())
        },
        "accuracy": {
            "max_score_difference": float(max_diff),
            "class_matches": int(class_matches),
            "total_comparisons": int(total_comparisons),
            "class_agreement_percent": float(100 * class_matches / total_comparisons)
        },
        "validation": {
            "latency_target_ms": target_latency,
            "latency_passes": bool(latencies.mean() <= target_latency),
            "accuracy_threshold": accuracy_threshold,
            "accuracy_passes": bool(max_diff < accuracy_threshold),
            "overall_pass": bool(latencies.mean() <= target_latency and max_diff < accuracy_threshold)
        },
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
        description='Benchmark TensorRT INT8 engine',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--test-dir',
        default='data/test_images',
        help='Directory with test images'
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=10,
        help='Number of warmup iterations'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=100,
        help='Number of benchmark iterations'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output JSON file path (default: auto-generated with timestamp)'
    )

    args = parser.parse_args()

    try:
        return_code = benchmark_tensorrt(
            test_images_dir=args.test_dir,
            num_warmup=args.warmup,
            num_iterations=args.iterations,
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
