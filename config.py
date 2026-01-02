"""
Few-Shot Learning Configuration
================================
Optimized for: 1500 classes × 2 images per class
Uses: Prototypical Networks + Siamese Learning
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List, Optional


@dataclass
class Config:
    """Complete configuration for few-shot learning."""
    
    # ===========================================
    # PATHS
    # ===========================================
    data_dir: str = "final_folder"
    save_dir: str = "models_fewshot"
    log_dir: str = "logs_fewshot"
    
    # ===========================================
    # FEW-SHOT SETTINGS
    # ===========================================
    # N-way K-shot: N classes, K examples per class
    n_way: int = 30          # Number of classes per episode
    k_shot: int = 1          # Support samples per class (1 for your data)
    q_query: int = 1         # Query samples per class
    
    # ===========================================
    # MODEL SETTINGS
    # ===========================================
    backbone: str = "resnet18"      # Smaller backbone
    embedding_dim: int = 128        # Compact embeddings
    dropout: float = 0.3
    pretrained: bool = True
    
    # ===========================================
    # TRAINING SETTINGS
    # ===========================================
    num_epochs: int = 200
    episodes_per_epoch: int = 100   # Training episodes per epoch
    val_episodes: int = 50          # Validation episodes
    
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    
    # Scheduler
    scheduler: str = "cosine"
    warmup_epochs: int = 10
    min_lr: float = 1e-6
    
    # ===========================================
    # LOSS SETTINGS
    # ===========================================
    temperature: float = 0.5        # For scaling distances
    margin: float = 0.5             # Triplet margin
    
    # ===========================================
    # EARLY STOPPING
    # ===========================================
    patience: int = 30              # More patience for few-shot
    min_delta: float = 0.001
    
    # ===========================================
    # CHECKPOINTING
    # ===========================================
    save_every: int = 10
    keep_best: int = 3
    
    # ===========================================
    # SYSTEM
    # ===========================================
    device: str = "cpu"
    num_workers: int = 4
    seed: int = 42
    num_cpu_threads: int = 16
    
    # ===========================================
    # IMAGE SETTINGS
    # ===========================================
    image_size: Tuple[int, int] = (112, 112)  # Smaller for speed
    
    def __post_init__(self):
        """Create directories."""
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)


# Default config
DEFAULT_CONFIG = Config()
