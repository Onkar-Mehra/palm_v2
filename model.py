"""
Prototypical Network for Few-Shot Palm Vein Classification
===========================================================
Learns to create embeddings where same-class samples are close
and different-class samples are far apart.

Key Idea:
- Create a "prototype" (mean embedding) for each class
- Classify by finding nearest prototype
- Works with just 1-2 samples per class!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


class ConvBlock(nn.Module):
    """Basic convolutional block."""
    
    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        
        if pool:
            layers.append(nn.MaxPool2d(2))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y


class EmbeddingNetwork(nn.Module):
    """
    Embedding network for few-shot learning.
    Creates compact, discriminative embeddings.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        embedding_dim: int = 128,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Encoder: 112x112 -> 7x7
        self.encoder = nn.Sequential(
            # Block 1: 112 -> 56
            ConvBlock(in_channels, 64, pool=True),
            SEBlock(64),
            
            # Block 2: 56 -> 28
            ConvBlock(64, 128, pool=True),
            SEBlock(128),
            
            # Block 3: 28 -> 14
            ConvBlock(128, 256, pool=True),
            SEBlock(256),
            
            # Block 4: 14 -> 7
            ConvBlock(256, 512, pool=True),
            SEBlock(512),
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Embedding head
        self.embedding = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        self.embedding_dim = embedding_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        # L2 normalize
        x = F.normalize(x, p=2, dim=1)
        return x


class ResNetEmbedding(nn.Module):
    """
    ResNet-based embedding network.
    Uses pretrained ResNet18 as backbone.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        embedding_dim: int = 128,
        dropout: float = 0.3,
        pretrained: bool = True
    ):
        super().__init__()
        
        # Load pretrained ResNet18
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            if pretrained:
                self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.backbone = resnet18(weights=None)
        except:
            from torchvision.models import resnet18
            self.backbone = resnet18(pretrained=pretrained)
        
        # Modify first conv for different input channels
        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        # Remove FC layer
        self.backbone.fc = nn.Identity()
        
        # Embedding head
        self.embedding = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        self.embedding_dim = embedding_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        # L2 normalize
        x = F.normalize(x, p=2, dim=1)
        return x


class DualStreamEncoder(nn.Module):
    """
    Dual stream encoder for RGB + IR fusion.
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        dropout: float = 0.3,
        backbone: str = "custom",
        pretrained: bool = True
    ):
        super().__init__()
        
        if backbone == "resnet18":
            self.rgb_encoder = ResNetEmbedding(
                in_channels=3,
                embedding_dim=embedding_dim,
                dropout=dropout,
                pretrained=pretrained
            )
            self.ir_encoder = ResNetEmbedding(
                in_channels=1,
                embedding_dim=embedding_dim,
                dropout=dropout,
                pretrained=pretrained
            )
        else:
            self.rgb_encoder = EmbeddingNetwork(
                in_channels=3,
                embedding_dim=embedding_dim,
                dropout=dropout
            )
            self.ir_encoder = EmbeddingNetwork(
                in_channels=1,
                embedding_dim=embedding_dim,
                dropout=dropout
            )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        self.embedding_dim = embedding_dim
    
    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """Get fused embedding."""
        rgb_emb = self.rgb_encoder(rgb)
        ir_emb = self.ir_encoder(ir)
        
        # Concatenate and fuse
        combined = torch.cat([rgb_emb, ir_emb], dim=1)
        fused = self.fusion(combined)
        
        # L2 normalize final embedding
        fused = F.normalize(fused, p=2, dim=1)
        
        return fused
    
    def get_embedding(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """Alias for forward."""
        return self.forward(rgb, ir)


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network for Few-Shot Classification.
    
    Training:
    1. Get embeddings for support and query samples
    2. Compute prototype (mean) for each class from support
    3. Classify query by finding nearest prototype
    
    Inference:
    1. Compute prototypes from enrolled samples
    2. Compare new sample to all prototypes
    3. Return class with nearest prototype
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        dropout: float = 0.3,
        backbone: str = "custom",
        pretrained: bool = True,
        temperature: float = 0.5
    ):
        super().__init__()
        
        self.encoder = DualStreamEncoder(
            embedding_dim=embedding_dim,
            dropout=dropout,
            backbone=backbone,
            pretrained=pretrained
        )
        
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        
        # Storage for class prototypes (for inference)
        self.prototypes: Dict[int, torch.Tensor] = {}
        self.class_names: Dict[int, str] = {}
    
    def forward(
        self,
        support_rgb: torch.Tensor,
        support_ir: torch.Tensor,
        support_labels: torch.Tensor,
        query_rgb: torch.Tensor,
        query_ir: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for episodic training.
        
        Args:
            support_rgb: (N*K, 3, H, W) support RGB images
            support_ir: (N*K, 1, H, W) support IR images
            support_labels: (N*K,) support labels
            query_rgb: (N*Q, 3, H, W) query RGB images
            query_ir: (N*Q, 1, H, W) query IR images
        
        Returns:
            logits: (N*Q, N) classification logits
            prototypes: (N, D) class prototypes
        """
        # Get embeddings
        support_emb = self.encoder(support_rgb, support_ir)  # (N*K, D)
        query_emb = self.encoder(query_rgb, query_ir)  # (N*Q, D)
        
        # Compute prototypes
        unique_labels = torch.unique(support_labels)
        n_classes = len(unique_labels)
        
        prototypes = []
        for label in unique_labels:
            mask = support_labels == label
            class_emb = support_emb[mask]
            prototype = class_emb.mean(dim=0)
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes)  # (N, D)
        
        # Compute distances (negative squared Euclidean)
        # query_emb: (N*Q, D), prototypes: (N, D)
        dists = torch.cdist(query_emb, prototypes, p=2)  # (N*Q, N)
        
        # Convert to logits (negative distance / temperature)
        logits = -dists / self.temperature
        
        return logits, prototypes
    
    def get_embedding(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """Get embedding for a single sample."""
        return self.encoder(rgb, ir)
    
    def enroll(self, class_idx: int, class_name: str, rgb: torch.Tensor, ir: torch.Tensor):
        """Enroll a sample for a class."""
        self.eval()
        with torch.no_grad():
            emb = self.get_embedding(rgb, ir)
        
        if class_idx in self.prototypes:
            # Update prototype with running average
            old_proto = self.prototypes[class_idx]
            self.prototypes[class_idx] = (old_proto + emb.cpu()) / 2
            self.prototypes[class_idx] = F.normalize(self.prototypes[class_idx], p=2, dim=1)
        else:
            self.prototypes[class_idx] = emb.cpu()
            self.class_names[class_idx] = class_name
    
    def classify(
        self,
        rgb: torch.Tensor,
        ir: torch.Tensor,
        top_k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Classify a sample against enrolled prototypes.
        
        Returns:
            top_k_classes: (top_k,) class indices
            top_k_scores: (top_k,) confidence scores
        """
        self.eval()
        with torch.no_grad():
            query_emb = self.get_embedding(rgb, ir)  # (1, D)
        
        if not self.prototypes:
            raise ValueError("No classes enrolled!")
        
        # Stack prototypes
        class_indices = list(self.prototypes.keys())
        proto_stack = torch.stack([self.prototypes[c] for c in class_indices])
        proto_stack = proto_stack.squeeze(1).to(query_emb.device)  # (num_classes, D)
        
        # Compute similarities (dot product since normalized)
        query_emb = query_emb.cpu()
        proto_stack = proto_stack.cpu()
        similarities = torch.mm(query_emb, proto_stack.t()).squeeze(0)  # (num_classes,)
        
        # Get top-k
        top_k = min(top_k, len(class_indices))
        top_scores, top_indices = torch.topk(similarities, top_k)
        top_classes = torch.tensor([class_indices[i] for i in top_indices])
        
        return top_classes, top_scores
    
    def clear_prototypes(self):
        """Clear enrolled prototypes."""
        self.prototypes = {}
        self.class_names = {}


class PrototypicalLoss(nn.Module):
    """
    Loss function for Prototypical Networks.
    Cross-entropy over distances to prototypes.
    """
    
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        logits: torch.Tensor,
        query_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute prototypical loss.
        
        Args:
            logits: (N*Q, N) logits from prototypical network
            query_labels: (N*Q,) ground truth labels (0 to N-1)
        
        Returns:
            loss: scalar loss
            metrics: dict with accuracy
        """
        loss = F.cross_entropy(logits, query_labels)
        
        # Compute accuracy
        preds = logits.argmax(dim=1)
        accuracy = (preds == query_labels).float().mean().item() * 100
        
        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy
        }
        
        return loss, metrics
