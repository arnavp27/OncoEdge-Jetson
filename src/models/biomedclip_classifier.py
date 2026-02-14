"""
BiomedCLIP Classifier Wrapper

Zero-shot medical image classification using BiomedCLIP.
"""
import open_clip
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from huggingface_hub import hf_hub_download
import os


class BiomedCLIPClassifier:
    """
    Wrapper for BiomedCLIP model for zero-shot medical image classification.

    Uses pre-defined clinical prompts to classify oral lesion patches into
    categories: OSCC, OPMD, or Normal.

    The model is automatically downloaded to models/biomedclip/ on first use.
    """

    # Clinical prompts for zero-shot classification
    # Updated to match YOLO dataset: OCA, OPMD, Benign, Normal
    PROMPTS = [
        "clinical photograph of oral squamous cell carcinoma",
        "clinical photograph of oral leukoplakia white patch",
        "clinical photograph of benign oral lesion aphthous ulcer",
        "clinical photograph of normal oral mucosa"
    ]

    # Class labels corresponding to prompts (matches YOLO dataset + Normal for YOLO false positives)
    CLASS_NAMES = ["OCA", "OPMD", "Benign", "Normal"]

    # Model configuration
    MODEL_ID = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    MODEL_FILE = "open_clip_pytorch_model.bin"

    def __init__(self, device='cuda', use_local=True):
        """
        Initialize BiomedCLIP classifier.

        Args:
            device: Device to run inference on ('cuda' or 'cpu')
            use_local: If True, downloads and caches model locally in models/biomedclip/
                      If False, loads directly from HuggingFace (slower)
        """
        self.device = device
        self.use_local = use_local

        # Load BiomedCLIP model
        print("Loading BiomedCLIP model...")
        try:
            if use_local:
                # Download to local models folder and load from there
                local_model_path = self._ensure_local_model()
                print(f"Loading from local path: {local_model_path}")
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    'hf-hub:' + self.MODEL_ID
                )
            else:
                # Load directly from HuggingFace Hub
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    'hf-hub:' + self.MODEL_ID
                )

            self.model.to(device)
            self.model.eval()
            print("BiomedCLIP model loaded successfully")
        except Exception as e:
            print(f"Error loading BiomedCLIP model: {e}")
            raise

        # Get tokenizer
        self.tokenizer = open_clip.get_tokenizer('hf-hub:' + self.MODEL_ID)

        # Context length for text tokenization
        self.context_length = 256

        # Precompute text embeddings for efficiency
        self.text_features = self._encode_text_prompts()

    def _ensure_local_model(self):
        """
        Download model to local models/biomedclip/ folder if not already present.

        Returns:
            Path: Path to the local model directory
        """
        # Get project root (assuming src/models/biomedclip_classifier.py)
        project_root = Path(__file__).parent.parent.parent
        local_model_dir = project_root / "models" / "biomedclip"
        local_model_dir.mkdir(parents=True, exist_ok=True)

        local_model_file = local_model_dir / self.MODEL_FILE

        # Check if model already exists locally
        if local_model_file.exists():
            print(f"✓ Using cached model from: {local_model_dir}")
            return local_model_dir

        # Download model from HuggingFace Hub
        print(f"Downloading BiomedCLIP model (~784 MB) to {local_model_dir}...")
        print("This is a one-time download and will be cached locally.")

        try:
            # Download main model file
            downloaded_path = hf_hub_download(
                repo_id=self.MODEL_ID,
                filename=self.MODEL_FILE,
                cache_dir=local_model_dir,
                local_dir=local_model_dir,
                local_dir_use_symlinks=False
            )
            print(f"✓ Model downloaded successfully to: {local_model_dir}")

            # Create info file
            info_file = local_model_dir / "model_info.txt"
            with open(info_file, 'w') as f:
                f.write(f"BiomedCLIP Model\n")
                f.write(f"=" * 50 + "\n\n")
                f.write(f"Model ID: {self.MODEL_ID}\n")
                f.write(f"Model file: {self.MODEL_FILE}\n")
                f.write(f"Size: ~784 MB\n")
                f.write(f"Architecture: ViT-B/16 + PubMedBERT\n")
                f.write(f"Parameters: ~196M (86M vision + 110M text)\n")
                f.write(f"Context length: 256 tokens\n")
                f.write(f"Input resolution: 224x224\n")

            return local_model_dir

        except Exception as e:
            print(f"Warning: Could not download to local folder: {e}")
            print("Falling back to HuggingFace Hub direct loading...")
            return None

    def _encode_text_prompts(self):
        """
        Precompute and cache text embeddings for clinical prompts.

        Returns:
            torch.Tensor: Normalized text features
        """
        import open_clip.tokenizer as clip_tokenizer

        with torch.no_grad():
            # Manual tokenization using the underlying tokenizer
            # This avoids the batch_encode_plus compatibility issue
            tokenized = []
            for text in self.PROMPTS:
                # Tokenize each text individually
                tokens = self.tokenizer.tokenizer(
                    text,
                    max_length=self.context_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                tokenized.append(tokens['input_ids'])

            # Stack all tokenized prompts
            text_tokens = torch.cat(tokenized, dim=0).to(self.device)

            # Encode text features
            text_features = self.model.encode_text(text_tokens)

            # Normalize features
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def classify(self, image_patch):
        """
        Classify a lesion patch using zero-shot learning.

        Args:
            image_patch: PIL Image or numpy array of lesion crop

        Returns:
            dict: Classification results containing:
                - 'class': Predicted class name (str)
                - 'scores': Dictionary of class probabilities (dict)
                - 'max_score': Maximum probability score (float)
        """
        # Convert numpy array to PIL Image if needed
        if isinstance(image_patch, np.ndarray):
            image_patch = Image.fromarray(image_patch)

        # Ensure RGB format
        if image_patch.mode != 'RGB':
            image_patch = image_patch.convert('RGB')

        # Preprocess and encode image
        image_tensor = self.preprocess(image_patch).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Encode image
            image_features = self.model.encode_image(image_tensor)
            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Compute similarity scores (cosine similarity)
            similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
            scores = similarity[0].cpu().numpy()

        # Format output
        class_scores = {name: float(score) for name, score in zip(self.CLASS_NAMES, scores)}
        max_idx = scores.argmax()

        return {
            'class': self.CLASS_NAMES[max_idx],
            'scores': class_scores,
            'max_score': float(scores[max_idx])
        }

    def get_class_probabilities(self, image_patch):
        """
        Get probability distribution over all classes.

        Args:
            image_patch: PIL Image or numpy array

        Returns:
            dict: Dictionary mapping class names to probabilities
        """
        result = self.classify(image_patch)
        return result['scores']
