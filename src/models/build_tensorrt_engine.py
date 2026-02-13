#!/usr/bin/env python3
"""
Build INT8 TensorRT engine for BiomedCLIP vision encoder.

CRITICAL: Must run on Jetson Nano for hardware-specific quantization.
Calibration uses same preprocessing path as inference to prevent drift.

Expected build time: 15-30 minutes on Jetson Nano
Expected engine size: 70-120 MB (vs 330MB FP32)
"""
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pathlib import Path
from PIL import Image
import sys


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class BiomedCLIPCalibrator(trt.IInt8EntropyCalibrator2):
    """
    INT8 calibrator for TensorRT.

    CRITICAL: Uses same preprocessing path as inference to prevent accuracy drift.
    Loads RAW JPEGs and preprocesses on-the-fly.

    Uses entropy calibration (v2) which is recommended for vision models.
    Calibration runs ON JETSON NANO using its GPU and TensorRT version.
    """

    def __init__(self, calibration_images_dir, cache_file):
        """
        Initialize INT8 calibrator.

        Args:
            calibration_images_dir: Directory with raw JPEG calibration images
            cache_file: Path to calibration cache file (for rebuilds)
        """
        trt.IInt8EntropyCalibrator2.__init__(self)

        # Load raw JPEG paths (not preprocessed tensors)
        self.image_paths = sorted(list(Path(calibration_images_dir).glob('*.jpg')))
        if len(self.image_paths) == 0:
            self.image_paths = sorted(list(Path(calibration_images_dir).glob('*.jpeg')))

        assert len(self.image_paths) >= 100, \
            f"Need at least 100 images for calibration, found {len(self.image_paths)}"

        if len(self.image_paths) < 250:
            print(f"[WARNING] Using {len(self.image_paths)} images. Recommended: 250-400 for robust calibration.")

        self.batch_size = 1
        self.current_index = 0
        self.cache_file = cache_file

        # Import shared preprocessing (avoids circular import)
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from src.models.biomedclip_preprocess import preprocess_image
        self.preprocessor = preprocess_image

        # Allocate GPU memory for calibration batch
        self.device_input = cuda.mem_alloc(1 * 3 * 224 * 224 * np.dtype(np.float32).itemsize)

        print(f"[OK] Calibrator initialized with {len(self.image_paths)} images")

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        """Get next calibration batch."""
        if self.current_index < len(self.image_paths):
            # Load and preprocess raw JPEG using exact inference path
            img_path = self.image_paths[self.current_index]

            try:
                img = Image.open(img_path)
                img_tensor = self.preprocessor(img)  # (1, 3, 224, 224) float32

                # CRITICAL: Ensure array is contiguous for CUDA memcpy
                if not img_tensor.flags['C_CONTIGUOUS']:
                    img_tensor = np.ascontiguousarray(img_tensor)
            except Exception as e:
                print(f"[WARNING] Failed to load {img_path}: {e}")
                self.current_index += self.batch_size
                return self.get_batch(names)  # Try next image

            cuda.memcpy_htod(self.device_input, img_tensor)
            self.current_index += self.batch_size

            # Progress logging
            if self.current_index % 50 == 0 or self.current_index == len(self.image_paths):
                print(f"  Calibration progress: {self.current_index}/{len(self.image_paths)}")

            return [int(self.device_input)]

        return None

    def read_calibration_cache(self):
        """Read calibration cache if exists."""
        if Path(self.cache_file).exists():
            print(f"[OK] Using existing calibration cache: {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        """Write calibration cache for future rebuilds."""
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
        print(f"[OK] Calibration cache saved: {self.cache_file}")


def build_tensorrt_int8_engine(
    onnx_path='models/biomedclip/onnx/vision_encoder.onnx',
    calibration_images_dir='data/calibration_images',
    engine_path='models/biomedclip/tensorrt/vision_encoder_int8.engine',
    workspace_size_mb=512  # Conservative - Nano has limited memory
):
    """
    Build INT8 TensorRT engine with calibration ON JETSON NANO.

    This must run on the target device (Nano) to ensure:
    - Hardware-specific quantization scales
    - GPU kernel compatibility
    - Optimal INT8 performance

    Args:
        onnx_path: Path to FP32 ONNX model
        calibration_images_dir: Directory with calibration images
        engine_path: Output engine path
        workspace_size_mb: TensorRT workspace size in MB (reduce if OOM)

    Returns:
        None (engine saved to disk)
    """
    print("=" * 70)
    print("BiomedCLIP TensorRT INT8 Engine Builder")
    print("=" * 70)

    # Verify ONNX model exists
    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    # Check for .onnx.data file (external weights)
    onnx_data_path = Path(str(onnx_path) + '.data')
    if onnx_data_path.exists():
        print(f"[OK] Found ONNX model with external data:")
        print(f"     {onnx_path} ({onnx_path.stat().st_size / 1024:.2f} KB)")
        print(f"     {onnx_data_path} ({onnx_data_path.stat().st_size / (1024**2):.2f} MB)")
    else:
        print(f"[OK] Found ONNX model: {onnx_path} ({onnx_path.stat().st_size / (1024**2):.2f} MB)")

    # Setup calibration cache path (in same directory as engine)
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    cache_file = engine_path.parent / "calibration.cache"

    # Create TensorRT builder
    print("\n[1/5] Creating TensorRT builder...")
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse FP32 ONNX model
    # CRITICAL: For models with external data, change to ONNX directory first
    # TensorRT looks for .onnx.data in CWD during parsing
    print(f"[2/5] Parsing ONNX model: {onnx_path}")

    import os
    original_cwd = os.getcwd()
    onnx_dir = onnx_path.parent

    try:
        # Change to ONNX directory so TensorRT finds .onnx.data
        os.chdir(onnx_dir)

        # Parse with just the filename (external data file is in same dir)
        with open(onnx_path.name, 'rb') as f:
            onnx_data = f.read()
            if not parser.parse(onnx_data):
                print("[ERROR] Failed to parse ONNX model:")
                for error in range(parser.num_errors):
                    print(f"  - {parser.get_error(error)}")
                raise RuntimeError("ONNX parsing failed")
    finally:
        # Restore original directory
        os.chdir(original_cwd)

    print(f"[OK] ONNX model parsed successfully")
    print(f"     Network inputs: {[network.get_input(i).name for i in range(network.num_inputs)]}")
    print(f"     Network outputs: {[network.get_output(i).name for i in range(network.num_outputs)]}")

    # Configure builder
    print(f"[3/5] Configuring builder (workspace: {workspace_size_mb}MB)...")
    config = builder.create_builder_config()

    # TensorRT 8.5+ uses set_memory_pool_limit instead of max_workspace_size
    if hasattr(config, 'set_memory_pool_limit'):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size_mb * (1 << 20))
    else:
        config.max_workspace_size = workspace_size_mb * (1 << 20)

    # Enable INT8 precision with FP16 fallback for stability
    # Some layers (LayerNorm, attention) may not quantize cleanly to INT8
    # FP16 fallback improves stability without breaking performance on Nano
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)  # Mixed precision fallback
    print("[OK] INT8 quantization enabled (with FP16 fallback for unstable layers)")

    # Set calibrator
    print(f"[4/5] Initializing INT8 calibrator...")
    calibrator = BiomedCLIPCalibrator(calibration_images_dir, cache_file)
    config.int8_calibrator = calibrator

    print(f"\n[OK] Starting TensorRT engine build...")
    print(f"     This will take 15-30 minutes on Jetson Nano")
    print(f"     Calibration will run on Nano GPU with {len(calibrator.image_paths)} images")
    print(f"     Note: JPEG decoding is CPU-bound during build (expected)")
    print()

    # Build engine
    print("[5/5] Building INT8 engine (please wait)...")

    # TensorRT 8.5+ uses build_serialized_network instead of build_engine
    if hasattr(builder, 'build_serialized_network'):
        # TensorRT 8.5+ (returns serialized engine directly)
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("\n[ERROR] Engine build failed!")
            print("Possible causes:")
            print("  - Out of memory (try --workspace 256 or 128)")
            print("  - ONNX opset incompatibility")
            print("  - TensorRT version mismatch")
            raise RuntimeError("Engine build failed")

        # Save serialized engine directly
        print("\n[OK] Engine build successful!")
        print(f"     Saving engine to: {engine_path}")
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
    else:
        # TensorRT < 8.5 (legacy API)
        engine = builder.build_engine(network, config)
        if engine is None:
            print("\n[ERROR] Engine build failed!")
            print("Possible causes:")
            print("  - Out of memory (try --workspace 256 or 128)")
            print("  - ONNX opset incompatibility (need opset 13-16 for Jetson)")
            print("  - TensorRT version mismatch")
            raise RuntimeError("Engine build failed")

        # Serialize and save
        print("\n[OK] Engine build successful!")
        print(f"     Serializing engine to: {engine_path}")
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())

    engine_size_mb = engine_path.stat().st_size / (1024**2)

    print()
    print("=" * 70)
    print("BUILD COMPLETE")
    print("=" * 70)
    print(f"[OK] TensorRT engine saved: {engine_path}")
    print(f"     Engine size: {engine_size_mb:.2f} MB (vs ~330MB FP32 ONNX)")
    print(f"     Compression: {330 / engine_size_mb:.1f}x smaller")
    print(f"[OK] Calibration cache saved: {cache_file}")
    print()
    print("Next steps:")
    print("  1. Run validation: python3 tests/benchmark_tensorrt.py")
    print("  2. Test pipeline: streamlit run streamlit_app.py")
    print("=" * 70)


def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Build INT8 TensorRT engine for BiomedCLIP vision encoder',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--onnx',
        default='models/biomedclip/onnx/vision_encoder.onnx',
        help='Path to FP32 ONNX model'
    )
    parser.add_argument(
        '--calib-dir',
        default='data/calibration_images',
        help='Directory with calibration images (250-400 JPEGs recommended)'
    )
    parser.add_argument(
        '--output',
        default='models/biomedclip/tensorrt/vision_encoder_int8.engine',
        help='Output TensorRT engine path'
    )
    parser.add_argument(
        '--workspace',
        type=int,
        default=512,
        help='Workspace size in MB (reduce to 256 or 128 if OOM)'
    )

    args = parser.parse_args()

    try:
        build_tensorrt_int8_engine(
            onnx_path=args.onnx,
            calibration_images_dir=args.calib_dir,
            engine_path=args.output,
            workspace_size_mb=args.workspace
        )
    except Exception as e:
        print(f"\n[ERROR] Build failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
