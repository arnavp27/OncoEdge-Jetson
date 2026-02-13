"""OpenCLIP backend for BiomedCLIP classifier (hf-hub / .bin)."""
import time

import numpy as np
import open_clip
import torch
from PIL import Image
from pathlib import Path
from huggingface_hub import hf_hub_download

from src.interfaces.classifier import BaseClassifier, ClassificationResult
from src.backends.registry import register_classifier


@register_classifier('hf-hub')
@register_classifier('.bin')
class OpenCLIPClassifier(BaseClassifier):
    """BiomedCLIP classifier using open_clip with full model (vision + text encoders).

    Loads the complete BiomedCLIP model and performs zero-shot classification
    using configurable prompts and class names.
    """

    MODEL_ID = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    MODEL_FILE = "open_clip_pytorch_model.bin"

    def __init__(self, model_path: str = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
                 device: str = 'cuda',
                 prompts: list = None,
                 class_names: list = None,
                 **kwargs):
        self.model_path = model_path
        self.device = device
        self.prompts = prompts or [
            "clinical photograph of oral squamous cell carcinoma",
            "clinical photograph of oral leukoplakia white patch",
            "clinical photograph of normal oral mucosa",
        ]
        self.class_names_list = class_names or ["OSCC", "OPMD", "Normal"]
        self.context_length = 256
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.text_features = None

    def load(self) -> None:
        """Load BiomedCLIP model from HuggingFace Hub."""
        hub_name = self.model_path
        if not hub_name.startswith('hf-hub:'):
            hub_name = 'hf-hub:' + self.MODEL_ID

        # Ensure local cache exists
        self._ensure_local_model()

        print("Loading BiomedCLIP model...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(hub_name)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = open_clip.get_tokenizer(hub_name)
        self.text_features = self._encode_text_prompts()
        print("BiomedCLIP model loaded successfully")

    def _ensure_local_model(self):
        """Download model to local models/biomedclip/ folder if not already present."""
        project_root = Path(__file__).parent.parent.parent
        local_model_dir = project_root / "models" / "biomedclip"
        local_model_dir.mkdir(parents=True, exist_ok=True)

        local_model_file = local_model_dir / self.MODEL_FILE
        if local_model_file.exists():
            return local_model_dir

        try:
            hf_hub_download(
                repo_id=self.MODEL_ID,
                filename=self.MODEL_FILE,
                cache_dir=local_model_dir,
                local_dir=local_model_dir,
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            print(f"Warning: Could not download to local folder: {e}")
            print("Falling back to HuggingFace Hub direct loading...")

        return local_model_dir

    def _encode_text_prompts(self) -> torch.Tensor:
        """Precompute and cache text embeddings for clinical prompts."""
        with torch.no_grad():
            tokenized = []
            for text in self.prompts:
                tokens = self.tokenizer.tokenizer(
                    text,
                    max_length=self.context_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                )
                tokenized.append(tokens['input_ids'])

            text_tokens = torch.cat(tokenized, dim=0).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def classify(self, image_patch: np.ndarray) -> ClassificationResult:
        """Classify a lesion patch using zero-shot learning.

        Args:
            image_patch: numpy array (H, W, 3) uint8 RGB

        Returns:
            ClassificationResult with class probabilities
        """
        t0 = time.perf_counter()

        if isinstance(image_patch, np.ndarray):
            pil_image = Image.fromarray(image_patch)
        else:
            pil_image = image_patch

        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Use learned logit_scale instead of hardcoded 100.0
            logit_scale = self.model.logit_scale.exp().item()
            similarity = (logit_scale * image_features @ self.text_features.T).softmax(dim=-1)
            scores = similarity[0].cpu().numpy()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        class_scores = {name: float(s) for name, s in zip(self.class_names_list, scores)}
        max_idx = int(scores.argmax())

        return ClassificationResult(
            predicted_class=self.class_names_list[max_idx],
            class_scores=class_scores,
            max_score=float(scores[max_idx]),
            embedding=image_features[0].cpu().numpy(),
            inference_time_ms=elapsed_ms,
        )

    def get_class_names(self) -> list:
        return list(self.class_names_list)

    def get_model_info(self) -> dict:
        return {
            'format': 'hf-hub (open_clip)',
            'model_path': self.model_path,
            'device': self.device,
            'num_prompts': len(self.prompts),
        }
