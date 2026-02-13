"""
Test ONNX export of BiomedCLIP vision encoder.

Validates that ONNX model produces same outputs as PyTorch model.
"""
import pytest
import torch
import numpy as np
import onnxruntime as ort
import open_clip
from pathlib import Path

ONNX_PATH = Path('models/biomedclip/onnx/vision_encoder.onnx')
_skip_no_onnx = pytest.mark.skipif(
    not ONNX_PATH.exists(),
    reason=f"ONNX model not found at {ONNX_PATH} (run export script first)",
)


@_skip_no_onnx
def test_onnx_export_exists():
    """Test that ONNX model file exists."""
    onnx_path = Path('models/biomedclip/onnx/vision_encoder.onnx')
    assert onnx_path.exists(), f"ONNX model not found at {onnx_path}"


@_skip_no_onnx
def test_onnx_model_outputs():
    """Test that ONNX model produces similar outputs to PyTorch model."""
    # Load PyTorch model
    model, _, _ = open_clip.create_model_and_transforms(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    vision_encoder = model.visual
    vision_encoder.eval()

    # Load ONNX model
    onnx_path = 'models/biomedclip/onnx/vision_encoder.onnx'
    ort_session = ort.InferenceSession(onnx_path)

    # Create random input
    dummy_input = torch.randn(1, 3, 224, 224)

    # PyTorch inference
    with torch.no_grad():
        pytorch_output = vision_encoder(dummy_input).numpy()

    # ONNX inference
    onnx_output = ort_session.run(
        None,
        {'image': dummy_input.numpy()}
    )[0]

    # Compare outputs
    # Allow small numerical differences due to ONNX conversion
    np.testing.assert_allclose(
        pytorch_output,
        onnx_output,
        rtol=1e-3,  # 0.1% relative tolerance
        atol=1e-5   # 0.00001 absolute tolerance
    )

    print(f"\n✓ ONNX output matches PyTorch output")
    print(f"  Max absolute difference: {np.abs(pytorch_output - onnx_output).max():.6f}")
    print(f"  Mean absolute difference: {np.abs(pytorch_output - onnx_output).mean():.6f}")


@_skip_no_onnx
def test_onnx_model_shape():
    """Test ONNX model input/output shapes."""
    onnx_path = 'models/biomedclip/onnx/vision_encoder.onnx'
    ort_session = ort.InferenceSession(onnx_path)

    # Check input shape
    input_name = ort_session.get_inputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape
    assert input_name == 'image', f"Expected input name 'image', got '{input_name}'"
    assert input_shape == [1, 3, 224, 224], f"Expected shape [1, 3, 224, 224], got {input_shape}"

    # Check output shape
    output_name = ort_session.get_outputs()[0].name
    output_shape = ort_session.get_outputs()[0].shape
    assert output_name == 'image_features', f"Expected output name 'image_features', got '{output_name}'"
    assert output_shape == [1, 512], f"Expected shape [1, 512], got {output_shape}"

    print(f"\n[OK] ONNX model I/O shapes correct")
    print(f"  Input: {input_name} {input_shape}")
    print(f"  Output: {output_name} {output_shape}")


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
