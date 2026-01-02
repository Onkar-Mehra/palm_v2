"""
Configuration Settings - Anti-Overfitting Version
==================================================
Optimized for: 1500 people × 2 images (1 RGB + 1 IR pair each)
"""

from dataclasses import dataclass, field
from typing import Tuple, List
from pathlib import Path


@dataclass
class ImageConfig:
    """Image processing configuration."""
    input_size: Tuple[int, int] = (224, 224)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    

@dataclass
class AugmentationConfig:
    """Strong augmentation to prevent overfitting."""
    # Geometric
    rotation_degrees: int = 20
    scale_range: Tuple[float, float] = (0.85, 1.15)
    translate_range: Tuple[float, float] = (0.1, 0.1)
    horizontal_flip_prob: float = 0.5
    
    # Photometric
    brightness_range: Tuple[float, float] = (0.7, 1.3)
    contrast_range: Tuple[float, float] = (0.7, 1.3)
    saturation_range: Tuple[float, float] = (0.8, 1.2)
    
    # Noise & Blur
    gaussian_blur_prob: float = 0.3
    gaussian_blur_kernel: Tuple[int, int] = (3, 7)
    gaussian_noise_prob: float = 0.2
    gaussian_noise_std: float = 0.05
    
    # Cutout/Erasing
    random_erasing_prob: float = 0.3
    random_erasing_scale: Tuple[float, float] = (0.02, 0.2)


@dataclass
class ModelConfig:
    """Model architecture - kept simple to prevent overfitting."""
    backbone: str = "efficientnet_b0"  # Smaller model
    pretrained: bool = True
    embedding_dim: int = 256  # Smaller embedding
    dropout: float = 0.5  # High dropout
    use_attention: bool = True
    fusion_type: str = "concat"  # Simple fusion
    freeze_backbone_epochs: int = 5  # Freeze first 5 epochs


@dataclass
class TrainingConfig:
    """Training configuration with anti-overfitting measures."""
    # Basic
    num_epochs: int = 100
    batch_size: int = 32
    num_workers: int = 4
    
    # Learning rate
    initial_lr: float = 0.001
    min_lr: float = 1e-6
    weight_decay: float = 0.01  # Strong regularization
    
    # Scheduler
    scheduler_type: str = "cosine_warmup"
    warmup_epochs: int = 5
    
    # Loss
    label_smoothing: float = 0.2  # Higher smoothing
    
    # Early stopping - CRITICAL
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.001
    
    # Overfitting detection
    overfit_detection_epochs: int = 5  # Stop if val_acc=0 for 5 epochs
    max_train_val_gap: float = 30.0  # Stop if train_acc - val_acc > 30%
    
    # Checkpointing
    save_every_n_epochs: int = 5
    keep_n_checkpoints: int = 3
    
    # Gradient
    gradient_clip_val: float = 1.0
    
    # Validation
    val_split: float = 0.2


@dataclass 
class SystemConfig:
    """Complete system configuration."""
    image: ImageConfig = field(default_factory=ImageConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Paths
    data_dir: str = "final_folder"
    save_dir: str = "models_v2"
    log_dir: str = "logs_v2"
    
    # Device
    device: str = "cpu"
    num_cpu_threads: int = 16
    
    # Random seed
    seed: int = 42


# Default configuration
DEFAULT_CONFIG = SystemConfig()
