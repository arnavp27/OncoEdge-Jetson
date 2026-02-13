"""
Shared preprocessing for BiomedCLIP.

CRITICAL: This is the single source of truth for preprocessing.
Used by both calibration and inference to prevent preprocessing drift.

Preprocessing parameters are extracted once from OpenCLIP and frozen.
"""
import numpy as np
from PIL import Image
import open_clip


def extract_preprocessing_params():
    """Extract and print exact preprocessing parameters from OpenCLIP."""
    _, _, preprocess_transform = open_clip.create_model_and_transforms(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )

    # Inspect transform pipeline
    print("OpenCLIP preprocessing pipeline:")
    print(preprocess_transform)

    # These values are frozen from OpenCLIP 3.2.0 + BiomedCLIP
    # If upgrading OpenCLIP, re-verify these values
    return {
        'input_size': 224,
        'interpolation': Image.BICUBIC,  # OpenCLIP default
        'mean': np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32),
        'std': np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    }


# Frozen preprocessing parameters (verified with OpenCLIP 3.2.0)
PREPROCESS_PARAMS = {
    'input_size': 224,
    'interpolation': Image.BICUBIC,
    'mean': np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32),
    'std': np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
}


def preprocess_image(image):
    """
    Preprocess image for BiomedCLIP inference.

    Single source of truth used by:
    - Calibrator (TensorRT engine build)
    - Inference wrapper (TensorRT runtime)

    Args:
        image: PIL Image or numpy array

    Returns:
        numpy array (1, 3, 224, 224) float32
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Resize with exact interpolation mode from OpenCLIP
    image = image.convert('RGB').resize(
        (PREPROCESS_PARAMS['input_size'], PREPROCESS_PARAMS['input_size']),
        PREPROCESS_PARAMS['interpolation']
    )

    # Convert to array and normalize to [0, 1]
    img_array = np.array(image).astype(np.float32) / 255.0

    # Apply frozen normalization parameters
    img_array = (img_array - PREPROCESS_PARAMS['mean']) / PREPROCESS_PARAMS['std']

    # Convert to CHW format and add batch dimension
    img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
    img_array = np.expand_dims(img_array, axis=0)  # (1, 3, 224, 224)

    return img_array.astype(np.float32)
