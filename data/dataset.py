"""
Dataset Module - With Strong Augmentation
==========================================
Handles data loading with aggressive augmentation to prevent overfitting.
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Union, Callable
import logging
from collections import defaultdict
import random

logger = logging.getLogger(__name__)


class StrongAugmentation:
    """
    Strong augmentation pipeline to prevent overfitting.
    Creates diverse variations of each image.
    """
    
    def __init__(self, config=None):
        self.config = config
        
        # Geometric params
        self.rotation_degrees = 20
        self.scale_range = (0.85, 1.15)
        self.translate_range = (0.1, 0.1)
        self.flip_prob = 0.5
        
        # Photometric params
        self.brightness_range = (0.7, 1.3)
        self.contrast_range = (0.7, 1.3)
        
        # Noise params
        self.blur_prob = 0.3
        self.noise_prob = 0.2
        self.noise_std = 0.05
        
        # Erasing params
        self.erasing_prob = 0.3
    
    def __call__(self, image: np.ndarray, seed: int = None) -> np.ndarray:
        """Apply augmentation to image."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        img = image.copy().astype(np.float32)
        
        # Normalize to 0-1 if needed
        if img.max() > 1.0:
            img = img / 255.0
        
        h, w = img.shape[:2]
        
        # 1. Geometric transforms
        # Rotation
        angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Scale
        scale = random.uniform(*self.scale_range)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h))
        
        # Crop or pad back to original size
        if scale > 1:
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            img = img[start_h:start_h+h, start_w:start_w+w]
        else:
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            if len(img.shape) == 3:
                img = np.pad(img, ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w), (0, 0)), mode='reflect')
            else:
                img = np.pad(img, ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w)), mode='reflect')
        
        # Ensure correct size
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h))
        
        # Horizontal flip
        if random.random() < self.flip_prob:
            img = np.fliplr(img).copy()
        
        # 2. Photometric transforms
        # Brightness
        brightness = random.uniform(*self.brightness_range)
        img = img * brightness
        
        # Contrast
        contrast = random.uniform(*self.contrast_range)
        mean = img.mean()
        img = (img - mean) * contrast + mean
        
        # 3. Noise
        # Gaussian blur
        if random.random() < self.blur_prob:
            ksize = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        
        # Gaussian noise
        if random.random() < self.noise_prob:
            noise = np.random.normal(0, self.noise_std, img.shape).astype(np.float32)
            img = img + noise
        
        # 4. Random erasing
        if random.random() < self.erasing_prob:
            eh = random.randint(h // 10, h // 4)
            ew = random.randint(w // 10, w // 4)
            ex = random.randint(0, w - ew)
            ey = random.randint(0, h - eh)
            img[ey:ey+eh, ex:ex+ew] = random.random()
        
        # Clip to valid range
        img = np.clip(img, 0, 1)
        
        # Convert back to uint8
        img = (img * 255).astype(np.uint8)
        
        return img


class LightAugmentation:
    """Light augmentation for validation."""
    
    def __call__(self, image: np.ndarray, seed: int = None) -> np.ndarray:
        return image


def load_image(path: Union[str, Path]) -> np.ndarray:
    """Load image from path."""
    path = str(path)
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    return img


class PalmVeinDataset(Dataset):
    """
    Dataset for palm vein images with RGB and IR modalities.
    
    Structure:
    data_dir/
        person_001/
            rgb.jpg
            ir.jpg
        person_002/
            rgb.jpg
            ir.jpg
        ...
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        target_size: Tuple[int, int] = (224, 224),
        augmentation: Optional[Callable] = None,
        mode: str = "train"
    ):
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.augmentation = augmentation
        self.mode = mode
        
        # Load data
        self.samples = []
        self.labels = []
        self.label_to_name = {}
        self.name_to_label = {}
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load and organize dataset."""
        subdirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        for idx, person_dir in enumerate(subdirs):
            person_name = person_dir.name
            self.label_to_name[idx] = person_name
            self.name_to_label[person_name] = idx
            
            # Find RGB and IR images
            rgb_files = list(person_dir.glob("*rgb*")) + list(person_dir.glob("*RGB*"))
            ir_files = list(person_dir.glob("*ir*")) + list(person_dir.glob("*IR*"))
            
            if not rgb_files or not ir_files:
                logger.warning(f"Missing images for {person_name}")
                continue
            
            # Pair up images
            for rgb_path, ir_path in zip(sorted(rgb_files), sorted(ir_files)):
                self.samples.append({
                    'rgb_path': rgb_path,
                    'ir_path': ir_path,
                    'label': idx,
                    'name': person_name
                })
                self.labels.append(idx)
        
        logger.info(f"Loaded {len(self.samples)} samples from {len(self.label_to_name)} identities")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load images
        rgb_image = load_image(sample['rgb_path'])
        ir_image = load_image(sample['ir_path'])
        
        # Resize
        rgb_image = cv2.resize(rgb_image, self.target_size)
        ir_image = cv2.resize(ir_image, self.target_size)
        
        # Convert IR to grayscale if needed
        if len(ir_image.shape) == 3:
            ir_image = cv2.cvtColor(ir_image, cv2.COLOR_BGR2GRAY)
        
        # Apply augmentation with same seed for both
        if self.augmentation is not None and self.mode == "train":
            seed = random.randint(0, 2**32 - 1)
            rgb_image = self.augmentation(rgb_image, seed=seed)
            ir_image = self.augmentation(ir_image, seed=seed)
        
        # Convert to tensor
        rgb_tensor = self._to_tensor(rgb_image, channels=3)
        ir_tensor = self._to_tensor(ir_image, channels=1)
        
        return {
            'rgb': rgb_tensor,
            'ir': ir_tensor,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'name': sample['name']
        }
    
    def _to_tensor(self, image: np.ndarray, channels: int) -> torch.Tensor:
        """Convert image to normalized tensor."""
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
        
        if channels == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif channels == 1 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # HWC to CHW
        image = np.transpose(image, (2, 0, 1))
        
        return torch.from_numpy(image)
    
    def get_num_classes(self) -> int:
        return len(self.label_to_name)
    
    def get_name_by_label(self, label: int) -> str:
        return self.label_to_name.get(label, "Unknown")
    
    def get_label_by_name(self, name: str) -> int:
        return self.name_to_label.get(name, -1)
    
    def save_metadata(self, path: Union[str, Path]):
        """Save dataset metadata."""
        metadata = {
            'label_to_name': {str(k): v for k, v in self.label_to_name.items()},
            'name_to_label': self.name_to_label,
            'num_samples': len(self.samples),
            'num_classes': len(self.label_to_name)
        }
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load_metadata(cls, path: Union[str, Path]) -> dict:
        with open(path, 'r') as f:
            metadata = json.load(f)
        metadata['label_to_name'] = {int(k): v for k, v in metadata['label_to_name'].items()}
        return metadata


def create_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    target_size: Tuple[int, int] = (224, 224),
    val_split: float = 0.2,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, PalmVeinDataset]:
    """
    Create training and validation dataloaders.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Create augmentation
    train_augmentation = StrongAugmentation()
    
    # Create datasets
    train_dataset = PalmVeinDataset(
        data_dir=data_dir,
        target_size=target_size,
        augmentation=train_augmentation,
        mode="train"
    )
    
    val_dataset = PalmVeinDataset(
        data_dir=data_dir,
        target_size=target_size,
        augmentation=None,
        mode="val"
    )
    
    # Split indices
    indices = list(range(len(train_dataset)))
    random.shuffle(indices)
    
    split_idx = int(len(indices) * (1 - val_split))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Create subsets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    
    # Balanced sampler for training
    train_labels = [train_dataset.labels[i] for i in train_indices]
    class_counts = np.bincount(train_labels, minlength=train_dataset.get_num_classes())
    class_counts = np.maximum(class_counts, 1)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    return train_loader, val_loader, train_dataset
