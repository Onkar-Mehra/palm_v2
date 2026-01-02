"""
Few-Shot Dataset with Episodic Sampling
========================================
Creates N-way K-shot episodes for training.
"""

import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json
import logging

logger = logging.getLogger(__name__)


class Augmentation:
    """Data augmentation for few-shot learning."""
    
    def __init__(self, is_training: bool = True):
        self.is_training = is_training
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if not self.is_training:
            return image
        
        img = image.copy().astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        
        h, w = img.shape[:2]
        
        # Random horizontal flip
        if random.random() > 0.5:
            img = np.fliplr(img).copy()
        
        # Random rotation (-15 to 15 degrees)
        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Random brightness
        brightness = random.uniform(0.8, 1.2)
        img = img * brightness
        
        # Random contrast
        contrast = random.uniform(0.8, 1.2)
        mean = img.mean()
        img = (img - mean) * contrast + mean
        
        # Clip and convert back
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        
        return img


class FewShotDataset(Dataset):
    """
    Dataset for few-shot learning.
    
    Organizes data by class for episodic sampling.
    """
    
    def __init__(
        self,
        data_dir: str,
        image_size: Tuple[int, int] = (112, 112),
        augmentation: Optional[Augmentation] = None,
        mode: str = "train"
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.augmentation = augmentation
        self.mode = mode
        
        # Organize data by class
        self.class_to_samples: Dict[int, List[Dict]] = defaultdict(list)
        self.label_to_name: Dict[int, str] = {}
        self.name_to_label: Dict[str, int] = {}
        self.classes: List[int] = []
        
        self._load_data()
    
    def _load_data(self):
        """Load and organize dataset by class."""
        person_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        for idx, person_dir in enumerate(person_dirs):
            person_name = person_dir.name
            self.label_to_name[idx] = person_name
            self.name_to_label[person_name] = idx
            
            # Find RGB and IR images
            rgb_files = list(person_dir.glob("*rgb*")) + list(person_dir.glob("*RGB*"))
            ir_files = list(person_dir.glob("*ir*")) + list(person_dir.glob("*IR*"))
            
            if not rgb_files or not ir_files:
                continue
            
            # Add all pairs for this class
            for rgb_path, ir_path in zip(sorted(rgb_files), sorted(ir_files)):
                self.class_to_samples[idx].append({
                    'rgb_path': str(rgb_path),
                    'ir_path': str(ir_path),
                    'label': idx,
                    'name': person_name
                })
        
        # Only keep classes with at least 1 sample
        self.classes = [c for c in self.class_to_samples.keys() 
                       if len(self.class_to_samples[c]) >= 1]
        
        logger.info(f"Loaded {len(self.classes)} classes from {self.data_dir}")
        logger.info(f"Total samples: {sum(len(v) for v in self.class_to_samples.values())}")
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load and preprocess image."""
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to load: {path}")
        img = cv2.resize(img, self.image_size)
        return img
    
    def _to_tensor(self, image: np.ndarray, channels: int = 3) -> torch.Tensor:
        """Convert to tensor."""
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
        
        if channels == 1 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
        
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        
        return torch.from_numpy(image)
    
    def get_sample(self, class_idx: int, sample_idx: int = None) -> Dict[str, torch.Tensor]:
        """Get a sample from a specific class."""
        samples = self.class_to_samples[class_idx]
        
        if sample_idx is None:
            sample_idx = random.randint(0, len(samples) - 1)
        
        sample = samples[sample_idx % len(samples)]
        
        # Load images
        rgb = self._load_image(sample['rgb_path'])
        ir = self._load_image(sample['ir_path'])
        
        # Convert IR to grayscale
        if len(ir.shape) == 3:
            ir = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
        
        # Apply augmentation
        if self.augmentation is not None:
            rgb = self.augmentation(rgb)
            ir = self.augmentation(ir)
        
        return {
            'rgb': self._to_tensor(rgb, channels=3),
            'ir': self._to_tensor(ir, channels=1),
            'label': torch.tensor(class_idx, dtype=torch.long),
            'name': sample['name']
        }
    
    def __len__(self):
        return len(self.classes)
    
    def __getitem__(self, idx):
        """Get sample by index (for standard iteration)."""
        class_idx = self.classes[idx % len(self.classes)]
        return self.get_sample(class_idx)
    
    def get_num_classes(self) -> int:
        return len(self.classes)
    
    def save_metadata(self, path: str):
        """Save class mappings."""
        metadata = {
            'label_to_name': {str(k): v for k, v in self.label_to_name.items()},
            'name_to_label': self.name_to_label,
            'num_classes': len(self.classes),
            'classes': self.classes
        }
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)


class EpisodicSampler:
    """
    Samples N-way K-shot episodes for few-shot learning.
    
    Each episode:
    - Samples N random classes
    - For each class: K support samples + Q query samples
    """
    
    def __init__(
        self,
        dataset: FewShotDataset,
        n_way: int = 30,
        k_shot: int = 1,
        q_query: int = 1,
        num_episodes: int = 100
    ):
        self.dataset = dataset
        self.n_way = min(n_way, len(dataset.classes))
        self.k_shot = k_shot
        self.q_query = q_query
        self.num_episodes = num_episodes
    
    def sample_episode(self) -> Dict[str, torch.Tensor]:
        """
        Sample one episode.
        
        Returns:
            support_rgb: (N*K, 3, H, W)
            support_ir: (N*K, 1, H, W)
            support_labels: (N*K,)
            query_rgb: (N*Q, 3, H, W)
            query_ir: (N*Q, 1, H, W)
            query_labels: (N*Q,)
        """
        # Sample N classes
        episode_classes = random.sample(self.dataset.classes, self.n_way)
        
        support_rgb, support_ir, support_labels = [], [], []
        query_rgb, query_ir, query_labels = [], [], []
        
        for new_label, class_idx in enumerate(episode_classes):
            samples = self.dataset.class_to_samples[class_idx]
            num_samples = len(samples)
            
            # Get indices for support and query
            if num_samples >= self.k_shot + self.q_query:
                indices = random.sample(range(num_samples), self.k_shot + self.q_query)
            else:
                # Repeat samples if not enough
                indices = [random.randint(0, num_samples - 1) 
                          for _ in range(self.k_shot + self.q_query)]
            
            support_indices = indices[:self.k_shot]
            query_indices = indices[self.k_shot:self.k_shot + self.q_query]
            
            # Get support samples
            for idx in support_indices:
                sample = self.dataset.get_sample(class_idx, idx)
                support_rgb.append(sample['rgb'])
                support_ir.append(sample['ir'])
                support_labels.append(new_label)  # Use episode-local label
            
            # Get query samples
            for idx in query_indices:
                sample = self.dataset.get_sample(class_idx, idx)
                query_rgb.append(sample['rgb'])
                query_ir.append(sample['ir'])
                query_labels.append(new_label)
        
        return {
            'support_rgb': torch.stack(support_rgb),
            'support_ir': torch.stack(support_ir),
            'support_labels': torch.tensor(support_labels, dtype=torch.long),
            'query_rgb': torch.stack(query_rgb),
            'query_ir': torch.stack(query_ir),
            'query_labels': torch.tensor(query_labels, dtype=torch.long),
            'episode_classes': episode_classes
        }
    
    def __iter__(self):
        for _ in range(self.num_episodes):
            yield self.sample_episode()
    
    def __len__(self):
        return self.num_episodes


class FullEvaluationSampler:
    """
    Evaluates on all classes (not episodic).
    For final accuracy measurement.
    """
    
    def __init__(self, dataset: FewShotDataset):
        self.dataset = dataset
    
    def get_all_embeddings(self, model, device: str = 'cpu'):
        """
        Get embeddings for all samples.
        
        Returns:
            embeddings: Dict[class_idx -> List[embedding]]
            class_prototypes: Dict[class_idx -> mean_embedding]
        """
        model.eval()
        embeddings = defaultdict(list)
        
        with torch.no_grad():
            for class_idx in self.dataset.classes:
                samples = self.dataset.class_to_samples[class_idx]
                for sample_idx in range(len(samples)):
                    sample = self.dataset.get_sample(class_idx, sample_idx)
                    rgb = sample['rgb'].unsqueeze(0).to(device)
                    ir = sample['ir'].unsqueeze(0).to(device)
                    
                    emb = model.get_embedding(rgb, ir)
                    embeddings[class_idx].append(emb.cpu())
        
        # Compute prototypes (mean embedding per class)
        prototypes = {}
        for class_idx, emb_list in embeddings.items():
            stacked = torch.cat(emb_list, dim=0)
            prototypes[class_idx] = stacked.mean(dim=0)
        
        return embeddings, prototypes


def create_data_loaders(
    data_dir: str,
    n_way: int = 30,
    k_shot: int = 1,
    q_query: int = 1,
    train_episodes: int = 100,
    val_episodes: int = 50,
    image_size: Tuple[int, int] = (112, 112),
    val_split: float = 0.2,
    seed: int = 42
) -> Tuple[EpisodicSampler, EpisodicSampler, FewShotDataset]:
    """
    Create training and validation episode samplers.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Create datasets
    train_aug = Augmentation(is_training=True)
    val_aug = Augmentation(is_training=False)
    
    train_dataset = FewShotDataset(
        data_dir=data_dir,
        image_size=image_size,
        augmentation=train_aug,
        mode="train"
    )
    
    val_dataset = FewShotDataset(
        data_dir=data_dir,
        image_size=image_size,
        augmentation=val_aug,
        mode="val"
    )
    
    # Split classes for train/val
    all_classes = train_dataset.classes.copy()
    random.shuffle(all_classes)
    
    split_idx = int(len(all_classes) * (1 - val_split))
    train_classes = all_classes[:split_idx]
    val_classes = all_classes[split_idx:]
    
    # Update dataset classes
    train_dataset.classes = train_classes
    val_dataset.classes = val_classes
    
    logger.info(f"Train classes: {len(train_classes)}")
    logger.info(f"Val classes: {len(val_classes)}")
    
    # Create samplers
    train_sampler = EpisodicSampler(
        dataset=train_dataset,
        n_way=min(n_way, len(train_classes)),
        k_shot=k_shot,
        q_query=q_query,
        num_episodes=train_episodes
    )
    
    val_sampler = EpisodicSampler(
        dataset=val_dataset,
        n_way=min(n_way, len(val_classes)),
        k_shot=k_shot,
        q_query=q_query,
        num_episodes=val_episodes
    )
    
    return train_sampler, val_sampler, train_dataset
