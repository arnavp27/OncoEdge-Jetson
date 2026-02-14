#!/usr/bin/env python3
"""
TensorRT INT8 inference wrapper for BiomedCLIP vision encoder.

Drop-in replacement for BiomedCLIPClassifier on Jetson Nano.
Uses precomputed text embeddings and INT8 TensorRT engine.

Interface matches BiomedCLIPClassifier exactly for seamless integration.
"""
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import json
from pathlib import Path
import sys

# Import shared preprocessing
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.biomedclip_preprocess import preprocess_image


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class BiomedCLIPTensorRT:
    """
    TensorRT INT8 inference wrapper for BiomedCLIP vision encoder.

    Interface-compatible with BiomedCLIPClassifier for drop-in replacement.

    Uses:
    - INT8 TensorRT engine for vision encoding
    - Precomputed text embeddings (3×512, already L2-normalized)
    - Extracted logit scale (learned temperature parameter)
    """

    # Class labels (must match training order and YOLO dataset)
    CLASS_NAMES = ["OCA", "OPMD", "Benign", "Normal"]

    def __init__(
        self,
        engine_path='models/biomedclip/tensorrt/vision_encoder_int8.engine',
        text_embeddings_path='models/biomedclip/text_embeddings.npy',
        logit_scale_path='models/biomedclip/logit_scale.json'
    ):
        """
        Initialize TensorRT inference wrapper.

        Args:
            engine_path: Path to INT8 TensorRT engine
            text_embeddings_path: Path to precomputed text embeddings (.npy)
            logit_scale_path: Path to logit scale JSON file
        """
        print(f"Loading TensorRT engine: {engine_path}")

        # Load TensorRT engine
        engine_path = Path(engine_path)
        if not engine_path.exists():
            raise FileNotFoundError(
                f"TensorRT engine not found: {engine_path}\n"
                f"Please build the engine first: python3 src/models/build_tensorrt_engine.py"
            )

        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine")

        # Create execution context
        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()

        # Load precomputed text embeddings (already L2-normalized)
        text_embeddings_path = Path(text_embeddings_path)
        if not text_embeddings_path.exists():
            raise FileNotFoundError(f"Text embeddings not found: {text_embeddings_path}")

        self.text_features = np.load(text_embeddings_path)  # (4, 512) - 4 classes now

        # Verify shape
        if self.text_features.shape != (4, 512):
            raise ValueError(
                f"Expected text embeddings shape (4, 512), got {self.text_features.shape}\n"
                f"Did you regenerate embeddings after updating prompts?"
            )

        # Load logit scale (learned temperature parameter)
        logit_scale_path = Path(logit_scale_path)
        if not logit_scale_path.exists():
            raise FileNotFoundError(f"Logit scale not found: {logit_scale_path}")

        with open(logit_scale_path, 'r') as f:
            logit_data = json.load(f)
            self.logit_scale = logit_data['logit_scale']

        print(f"[OK] TensorRT engine loaded: {engine_path}")
        print(f"     Engine size: {engine_path.stat().st_size / (1024**2):.2f} MB")
        print(f"     Text embeddings: {self.text_features.shape}")
        print(f"     Logit scale: {self.logit_scale:.4f}")

    def _allocate_buffers(self):
        """
        Allocate GPU memory for input/output tensors.

        CRITICAL FIX: Use trt.volume() for element count, not byte count.
        The pagelocked_empty() takes element count, not bytes.

        TensorRT 10.x API: Use tensor names instead of deprecated binding API.
        """
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        # Store tensor names for execute_async_v3
        self.input_names = []
        self.output_names = []

        # TensorRT 10.x API: Use get_tensor_name and get_tensor_shape
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(tensor_name)
            size = trt.volume(shape)  # Element count (NOT bytes)
            dtype = np.float32

            # Allocate host and device buffers
            # CRITICAL: pagelocked_empty takes ELEMENT COUNT, not bytes
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)  # nbytes for GPU allocation

            bindings.append(int(device_mem))

            # TensorRT 10.x: Use get_tensor_mode to check input/output
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append({'host': host_mem, 'device': device_mem, 'name': tensor_name})
                self.input_names.append(tensor_name)
            else:
                outputs.append({'host': host_mem, 'device': device_mem, 'name': tensor_name})
                self.output_names.append(tensor_name)

        return inputs, outputs, bindings, stream

    def classify(self, image_patch):
        """
        Classify image patch using TensorRT INT8 engine.

        Interface matches BiomedCLIPClassifier.classify() exactly.

        Args:
            image_patch: PIL Image or numpy array

        Returns:
            dict: {
                'class': str,           # Predicted class name
                'scores': dict,         # Per-class probabilities
                'max_score': float      # Maximum probability
            }
        """
        # Preprocess using shared preprocessing module (same as calibration)
        img_tensor = preprocess_image(image_patch)  # (1, 3, 224, 224) float32

        # Copy to device
        np.copyto(self.inputs[0]['host'], img_tensor.ravel())
        cuda.memcpy_htod_async(
            self.inputs[0]['device'],
            self.inputs[0]['host'],
            self.stream
        )

        # TensorRT 10.x: Set tensor addresses for execute_async_v3
        for i, inp in enumerate(self.inputs):
            self.context.set_tensor_address(inp['name'], int(inp['device']))
        for i, out in enumerate(self.outputs):
            self.context.set_tensor_address(out['name'], int(out['device']))

        # Run inference (TensorRT 10.x API)
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy output back to host
        cuda.memcpy_dtoh_async(
            self.outputs[0]['host'],
            self.outputs[0]['device'],
            self.stream
        )
        self.stream.synchronize()

        # Extract and normalize image features
        features = self.outputs[0]['host'].reshape(512)  # (512,) - NOT 768!

        # L2 normalize image features
        features_norm = np.linalg.norm(features)
        if features_norm > 0:
            features /= features_norm
        else:
            # Handle edge case of zero features (shouldn't happen)
            print("[WARNING] Zero-norm features detected, skipping normalization")

        # Compute cosine similarity with text embeddings
        # Text features are already L2-normalized during precomputation
        # Image features just normalized above
        # Each normalized exactly once - no duplicate normalization
        similarity = np.dot(features, self.text_features.T)  # (3,)

        # Apply logit scaling and softmax (numerically stable)
        scaled = similarity * self.logit_scale

        # Numerical stability: subtract max before exp
        # CRITICAL for INT8: prevents noise amplification from exp()
        scaled_shifted = scaled - scaled.max()
        exp_sim = np.exp(scaled_shifted)
        scores = exp_sim / exp_sim.sum()

        # Format output to match BiomedCLIPClassifier interface
        class_scores = {
            name: float(score)
            for name, score in zip(self.CLASS_NAMES, scores)
        }
        max_idx = scores.argmax()

        return {
            'class': self.CLASS_NAMES[max_idx],
            'scores': class_scores,
            'max_score': float(scores[max_idx])
        }

    def get_class_probabilities(self, image_patch):
        """
        Get probability distribution over all classes.

        Interface matches BiomedCLIPClassifier.get_class_probabilities() exactly.

        Args:
            image_patch: PIL Image or numpy array

        Returns:
            dict: Dictionary mapping class names to probabilities
        """
        result = self.classify(image_patch)
        return result['scores']

    def __del__(self):
        """Cleanup CUDA resources."""
        if hasattr(self, 'stream'):
            try:
                self.stream.synchronize()
            except:
                pass  # Cleanup errors are non-fatal


# For testing as standalone script
if __name__ == '__main__':
    import sys
    from PIL import Image

    if len(sys.argv) < 2:
        print("Usage: python3 biomedclip_tensorrt.py <image_path>")
        print("Tests TensorRT inference on a single image")
        sys.exit(1)

    # Load model
    print("Initializing TensorRT model...")
    model = BiomedCLIPTensorRT()

    # Load and classify image
    image_path = sys.argv[1]
    print(f"\nClassifying: {image_path}")
    img = Image.open(image_path)

    result = model.classify(img)

    print("\nResults:")
    print(f"  Predicted class: {result['class']}")
    print(f"  Confidence: {result['max_score']:.4f}")
    print("\n  Class scores:")
    for cls, score in result['scores'].items():
        print(f"    {cls}: {score:.4f}")
