"""
Extract CLIP logit scale (learned temperature parameter) from BiomedCLIP.

CRITICAL: This must be extracted from the SAME checkpoint used for ONNX export
to ensure consistent confidence calibration after quantization.
"""
import open_clip
import json
from pathlib import Path


def extract_logit_scale(output_path='models/biomedclip/logit_scale.json'):
    """
    Extract logit scale from BiomedCLIP model.

    CRITICAL: Run this IMMEDIATELY AFTER loading model for ONNX export
    to ensure checkpoint consistency. Do not load model separately.

    Args:
        output_path: Path to save logit scale JSON file
    """
    print("Loading BiomedCLIP model...")
    model, _, _ = open_clip.create_model_and_transforms(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )

    # Extract learned logit scale parameter
    # logit_scale is stored as log(scale), so we exp() to get actual scale
    logit_scale = model.logit_scale.exp().item()

    # Save with checkpoint information for verification
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        'logit_scale': logit_scale,
        'model_checkpoint': 'microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
        'description': 'Learned temperature parameter for CLIP similarity scaling'
    }

    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[OK] Logit scale extracted: {logit_scale:.4f}")
    print(f"  Model checkpoint: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    print(f"  Saved to: {output_file}")

    # Verification
    if abs(logit_scale - 100.0) > 20.0:
        print(f"\n[WARNING] Logit scale is {logit_scale:.4f}")
        print("  Expected value ~100.0 for CLIP models")
        print("  You may have loaded a different checkpoint version")
        print("  Verify this is the same checkpoint used for ONNX export!")
    else:
        print(f"\n[OK] Logit scale value looks correct (~100.0 for CLIP models)")

    return logit_scale


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract CLIP logit scale from BiomedCLIP")
    parser.add_argument(
        '--output',
        default='models/biomedclip/logit_scale.json',
        help='Output path for logit scale JSON (default: models/biomedclip/logit_scale.json)'
    )

    args = parser.parse_args()

    extract_logit_scale(output_path=args.output)
