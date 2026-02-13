"""Configuration management for OncoEdge.

Loads from YAML with environment variable overrides.
Single source of truth for all pipeline parameters.
"""
import os
import yaml
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


def _resolve_device(preferred: str) -> str:
    """Resolve device string from preference.

    Args:
        preferred: "auto", "cuda", "mps", or "cpu"

    Returns:
        Resolved device string usable by PyTorch and backends.
    """
    if preferred == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return preferred


@dataclass
class DeviceConfig:
    """Device configuration with auto-detection."""
    preferred: str = "auto"

    @property
    def resolved(self) -> str:
        return _resolve_device(self.preferred)


@dataclass
class DetectorConfig:
    """Detector model configuration."""
    model_path: str = "yolo11n-seg.pt"
    conf_threshold: float = 0.25
    imgsz: int = 640


@dataclass
class ClassifierConfig:
    """Classifier model configuration."""
    model_path: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    text_embeddings_path: str = "models/biomedclip/text_embeddings.npy"
    logit_scale_path: str = "models/biomedclip/logit_scale.json"
    prompts: List[str] = field(default_factory=lambda: [
        "clinical photograph of oral squamous cell carcinoma",
        "clinical photograph of oral leukoplakia white patch",
        "clinical photograph of normal oral mucosa",
    ])
    class_names: List[str] = field(default_factory=lambda: ["OSCC", "OPMD", "Normal"])


@dataclass
class FusionConfig:
    """Fusion module configuration."""
    method: str = "gate"
    gate_threshold: float = 0.5


@dataclass
class ClinicalConfig:
    """Clinical decision thresholds.

    Risk formula: risk_score = clip_score * 10 + alpha * tobacco_norm + beta * smoking_norm
    Where:
        tobacco_norm = min(tobacco_years / tobacco_significant_years, 1.0)
        smoking_norm = min(smoking_years / smoking_significant_years, 1.0)

    The significant_years thresholds represent the point at which
    cancer risk becomes clinically significant (norm reaches 1.0).
    These should be set based on clinical evidence / doctor guidance.
    """
    high_risk_threshold: float = 8.0
    medium_risk_threshold: float = 5.0

    # Alpha: smokeless tobacco (gutka, paan, chewing) risk weight
    alpha: float = 2.0
    tobacco_significant_years: int = 15

    # Beta: smoking (cigarettes, bidis) risk weight
    beta: float = 1.5
    smoking_significant_years: int = 15


@dataclass
class PathsConfig:
    """File system paths."""
    models_dir: str = "models"
    data_dir: str = "data"
    feedback_dir: str = "data/feedback"
    metrics_dir: str = "data/metrics"


@dataclass
class MetricsConfig:
    """Metrics tracking configuration."""
    enabled: bool = True
    track_latency: bool = True
    track_confidence_distribution: bool = True
    log_file: str = "data/metrics/inference_log.jsonl"


@dataclass
class OncoEdgeConfig:
    """Top-level configuration container."""
    device: DeviceConfig = field(default_factory=DeviceConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    clinical: ClinicalConfig = field(default_factory=ClinicalConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)


def load_config(config_path: str = None) -> OncoEdgeConfig:
    """Load configuration from YAML file with environment variable overrides.

    Priority: environment variables > YAML file > dataclass defaults.

    Environment variable overrides:
        ONCOEDGE_CONFIG   - path to YAML config file
        ONCOEDGE_DEVICE   - device preference (auto/cuda/mps/cpu)
        ONCOEDGE_DETECTOR_MODEL   - detector model path
        ONCOEDGE_CLASSIFIER_MODEL - classifier model path

    Args:
        config_path: Path to YAML config. If None, checks ONCOEDGE_CONFIG
                     env var, then falls back to config/default.yaml.

    Returns:
        OncoEdgeConfig instance with all parameters resolved.
    """
    config = OncoEdgeConfig()

    # Determine config file path
    if config_path is None:
        config_path = os.environ.get('ONCOEDGE_CONFIG', 'config/default.yaml')

    yaml_path = Path(config_path)
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        if yaml_data:
            _apply_yaml(config, yaml_data)

    # Environment variable overrides (highest priority)
    if os.environ.get('ONCOEDGE_DEVICE'):
        config.device.preferred = os.environ['ONCOEDGE_DEVICE']
    if os.environ.get('ONCOEDGE_DETECTOR_MODEL'):
        config.detector.model_path = os.environ['ONCOEDGE_DETECTOR_MODEL']
    if os.environ.get('ONCOEDGE_CLASSIFIER_MODEL'):
        config.classifier.model_path = os.environ['ONCOEDGE_CLASSIFIER_MODEL']

    return config


def _apply_yaml(config: OncoEdgeConfig, yaml_data: dict):
    """Apply YAML data to config dataclass hierarchy.

    Iterates through top-level YAML sections and sets matching
    attributes on the corresponding config dataclass.
    """
    for section_name, section_data in yaml_data.items():
        if hasattr(config, section_name) and isinstance(section_data, dict):
            section_obj = getattr(config, section_name)
            for key, value in section_data.items():
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)


# Backward-compatible global config (deprecated, use load_config() instead)
CONFIG = {
    'model': type('ModelConfig', (), {
        'yolo_model': 'yolo11n-seg.pt',
        'yolo_conf_threshold': 0.25,
        'yolo_imgsz': 640,
        'biomedclip_model': 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
        'device': 'cuda',
    })(),
    'paths': type('PathConfig', (), {
        'project_root': Path(__file__).parent.parent,
        'data_dir': Path(__file__).parent.parent / 'data',
        'models_dir': Path(__file__).parent.parent / 'models',
        'test_images_dir': Path(__file__).parent.parent / 'data' / 'test_images',
    })(),
    'clinical': ClinicalConfig(),
}
