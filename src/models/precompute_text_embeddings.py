"""
Precompute text embeddings for BiomedCLIP.

Eliminates need to quantize text encoder by computing embeddings once on RTX 4050.
Text encoder outputs for 3 fixed prompts are saved as numpy array (~18KB).
"""
import torch
import numpy as np
import open_clip
from pathlib import Path


def precompute_biomedclip_text_embeddings(
    output_path='models/biomedclip/text_embeddings.npy',
    class_names_path='models/biomedclip/class_names.txt'
):
    """
    Precompute and save text embeddings for BiomedCLIP classification.

    Args:
        output_path: Path to save text embeddings (.npy file)
        class_names_path: Path to save class names mapping (.txt file)
    """
    PROMPTS = [
        "clinical photograph of oral squamous cell carcinoma",
        "clinical photograph of oral leukoplakia white patch",
        "clinical photograph of normal oral mucosa"
    ]
    CLASS_NAMES = ["OSCC", "OPMD", "Normal"]

    print("Loading BiomedCLIP model...")
    model, _, _ = open_clip.create_model_and_transforms(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    tokenizer = open_clip.get_tokenizer(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )

    model.eval()

    # Use CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"Using device: {device}")

    print("\nEncoding text prompts...")
    with torch.no_grad():
        # Tokenize each prompt individually to avoid batch_encode_plus issue
        tokenized = []
        for text in PROMPTS:
            tokens = tokenizer.tokenizer(
                text,
                max_length=256,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            tokenized.append(tokens['input_ids'])

        # Stack all tokenized prompts
        text_tokens = torch.cat(tokenized, dim=0).to(device)

        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)  # L2 normalize
        text_features_np = text_features.cpu().numpy()  # (3, 768)

    # Save text embeddings
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, text_features_np)

    # Save class names mapping
    class_names_file = Path(class_names_path)
    class_names_file.parent.mkdir(parents=True, exist_ok=True)
    with open(class_names_file, 'w') as f:
        for i, (prompt, class_name) in enumerate(zip(PROMPTS, CLASS_NAMES)):
            f.write(f"{i}: {class_name} - {prompt}\n")

    print(f"\n[OK] Text embeddings saved")
    print(f"  Shape: {text_features_np.shape}")
    print(f"  Size: {output_file.stat().st_size / 1024:.2f} KB")
    print(f"  Path: {output_file}")
    print(f"\n[OK] Class names saved: {class_names_file}")
    print("\nClass mapping:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"  {i}: {class_name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Precompute BiomedCLIP text embeddings")
    parser.add_argument(
        '--output',
        default='models/biomedclip/text_embeddings.npy',
        help='Output path for text embeddings (default: models/biomedclip/text_embeddings.npy)'
    )
    parser.add_argument(
        '--class-names',
        default='models/biomedclip/class_names.txt',
        help='Output path for class names (default: models/biomedclip/class_names.txt)'
    )

    args = parser.parse_args()

    precompute_biomedclip_text_embeddings(
        output_path=args.output,
        class_names_path=args.class_names
    )
