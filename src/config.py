"""
Configuration management for OncoEdge
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration"""
    yolo_model: str = 'yolo26n-seg.pt'
    yolo_conf_threshold: float = 0.25
    yolo_imgsz: int = 640
    
    biomedclip_model: str = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    
    device: str = 'cuda'  # 'cuda' or 'cpu'


@dataclass
class PathConfig:
    """Path configuration"""
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = project_root / 'data'
    models_dir: Path = project_root / 'models'
    test_images_dir: Path = data_dir / 'test_images'


@dataclass
class ClinicalConfig:
    """Clinical decision thresholds"""
    high_risk_threshold: float = 8.0
    medium_risk_threshold: float = 5.0
    
    age_risk_threshold: int = 40
    age_risk_weight: float = 1.5
    
    tobacco_risk_threshold: int = 10
    tobacco_risk_weight: float = 2.0
    
    long_duration_weight: float = 1.5


# Global config instance
CONFIG = {
    'model': ModelConfig(),
    'paths': PathConfig(),
    'clinical': ClinicalConfig()
}
