"""
Export BiomedCLIP vision encoder to ONNX format for TensorRT deployment.

Exports FP32 ONNX model that will be quantized to INT8 on Jetson Nano using TensorRT.
"""
import torch
import open_clip
import onnx
from pathlib import Path


def export_vision_encoder_to_onnx(
    output_path='models/biomedclip/onnx/vision_encoder.onnx',
    opset_version=13  # CRITICAL: Use 13-16 for Jetson Nano compatibility
):
    """
    Export BiomedCLIP vision encoder to ONNX FP32.

    Args:
        output_path: Path to save ONNX model
        opset_version: ONNX opset version (13-16 for Jetson Nano, NOT 17)

    Key decisions:
    - Fixed batch size=1 (no dynamic batching for Jetson Nano)
    - Opset 13-16 (NOT 17) for JetPack 4.6.x compatibility
    - FP32 export (INT8 quantization happens on Nano via TensorRT)
    """
    print("Loading BiomedCLIP model...")
    model, _, _ = open_clip.create_model_and_transforms(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )

    # Extract vision encoder only
    vision_encoder = model.visual
    vision_encoder.eval()

    print("Setting up export parameters...")
    # FIXED batch size=1 (no dynamic batching for Jetson Nano)
    dummy_input = torch.randn(1, 3, 224, 224)

    # Create output directory
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to ONNX (opset {opset_version})...")
    torch.onnx.export(
        vision_encoder,
        dummy_input,
        str(output_file),
        export_params=True,
        opset_version=opset_version,  # 13-16 for Jetson compatibility
        do_constant_folding=True,
        input_names=['image'],
        output_names=['image_features'],
        dynamic_axes=None,  # FIXED shapes - no dynamic batching
        verbose=False
    )

    print("Validating ONNX model...")
    onnx_model = onnx.load(str(output_file))
    onnx.checker.check_model(onnx_model)

    # Print model info
    model_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"\n[OK] ONNX export successful")
    print(f"  Model size: {model_size_mb:.2f} MB")
    print(f"  Opset version: {opset_version}")
    print(f"  Input: image (1, 3, 224, 224) float32")
    print(f"  Output: image_features (1, 768) float32")
    print(f"  Path: {output_file}")

    # Verify input/output shapes
    print("\nModel I/O verification:")
    for input_tensor in onnx_model.graph.input:
        print(f"  Input '{input_tensor.name}': {[d.dim_value for d in input_tensor.type.tensor_type.shape.dim]}")
    for output_tensor in onnx_model.graph.output:
        print(f"  Output '{output_tensor.name}': {[d.dim_value for d in output_tensor.type.tensor_type.shape.dim]}")

    print(f"\n[OK] ONNX model validated")
    print(f"\nNext steps:")
    print(f"  1. Transfer this FP32 ONNX to Jetson Nano")
    print(f"  2. Build INT8 TensorRT engine using TensorRT on Nano")
    print(f"  3. INT8 quantization will happen ON NANO (not here)")

    return str(output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export BiomedCLIP vision encoder to ONNX")
    parser.add_argument(
        '--output',
        default='models/biomedclip/onnx/vision_encoder.onnx',
        help='Output path for ONNX model (default: models/biomedclip/onnx/vision_encoder.onnx)'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=13,
        choices=[13, 14, 15, 16],
        help='ONNX opset version (13-16 for Jetson Nano, default: 13)'
    )

    args = parser.parse_args()

    export_vision_encoder_to_onnx(
        output_path=args.output,
        opset_version=args.opset
    )
