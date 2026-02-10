"""
BiomedCLIP Classifier Wrapper

Zero-shot medical image classification using BiomedCLIP.
"""
import open_clip
import torch
import numpy as np
from PIL import Image


class BiomedCLIPClassifier:
    """
    Wrapper for BiomedCLIP model for zero-shot medical image classification.

    Uses pre-defined clinical prompts to classify oral lesion patches into
    categories: OSCC, OPMD, or Normal.
    """

    # Clinical prompts for zero-shot classification
    PROMPTS = [
        "clinical photograph of oral squamous cell carcinoma",
        "clinical photograph of oral leukoplakia white patch",
        "clinical photograph of normal oral mucosa"
    ]

    # Class labels corresponding to prompts
    CLASS_NAMES = ["OSCC", "OPMD", "Normal"]

    def __init__(self, device='cuda'):
        """
        Initialize BiomedCLIP classifier.

        Args:
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device

        # Load BiomedCLIP model from HuggingFace
        print("Loading BiomedCLIP model...")
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
            )
            self.model.to(device)
            self.model.eval()
            print("BiomedCLIP model loaded successfully")
        except Exception as e:
            print(f"Error loading BiomedCLIP model: {e}")
            raise

        # Get tokenizer
        self.tokenizer = open_clip.get_tokenizer(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )

        # Precompute text embeddings for efficiency
        self.text_features = self._encode_text_prompts()

    def _encode_text_prompts(self):
        """
        Precompute and cache text embeddings for clinical prompts.

        Returns:
            torch.Tensor: Normalized text features
        """
        with torch.no_grad():
            text_tokens = self.tokenizer(self.PROMPTS).to(self.device)
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
