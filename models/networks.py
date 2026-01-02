"""
Neural Network Models - Lightweight Anti-Overfitting Version
============================================================
Smaller models with high regularization to prevent overfitting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    
    def __init__(self, channels: int, reduction: int = 16):
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


class LightweightBackbone(nn.Module):
    """
    Lightweight CNN backbone for feature extraction.
    Smaller than ResNet50 to prevent overfitting.
    """
    
    def __init__(self, in_channels: int = 3):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Block 2: 112 -> 56
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            SEBlock(64),
            
            # Block 3: 56 -> 28
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SEBlock(128),
            
            # Block 4: 28 -> 14
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SEBlock(256),
            
            # Block 5: 14 -> 7
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            SEBlock(512),
            
            # Global pooling
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.feature_dim = 512
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return x.view(x.size(0), -1)


class EfficientNetBackbone(nn.Module):
    """
    EfficientNet-B0 backbone - small and efficient.
    """
    
    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        
        try:
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            if pretrained:
                self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            else:
                self.backbone = efficientnet_b0(weights=None)
        except:
            from torchvision.models import efficientnet_b0
            self.backbone = efficientnet_b0(pretrained=pretrained)
        
        # Modify first conv if needed
        if in_channels != 3:
            old_conv = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(
                in_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )
        
        # Remove classifier
        self.backbone.classifier = nn.Identity()
        self.feature_dim = 1280
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        return x.view(x.size(0), -1)


class FeatureExtractor(nn.Module):
    """
    Feature extractor for single modality.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        backbone: str = "lightweight",
        pretrained: bool = True
    ):
        super().__init__()
        
        if backbone == "lightweight":
            self.features = LightweightBackbone(in_channels)
            self.feature_dim = 512
        elif backbone == "efficientnet_b0":
            self.features = EfficientNetBackbone(in_channels, pretrained)
            self.feature_dim = 1280
        else:
            self.features = LightweightBackbone(in_channels)
            self.feature_dim = 512
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class SimpleFusion(nn.Module):
    """
    Simple concatenation fusion - less prone to overfitting.
    """
    
    def __init__(self, feature_dim: int, output_dim: int, dropout: float = 0.5):
        super().__init__()
        
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
    
    def forward(self, rgb_features: torch.Tensor, ir_features: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([rgb_features, ir_features], dim=1)
        return self.fusion(combined)


class EmbeddingHead(nn.Module):
    """
    Embedding head with high dropout for regularization.
    """
    
    def __init__(self, input_dim: int, embedding_dim: int, dropout: float = 0.5):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.BatchNorm1d(input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        # L2 normalize
        x = F.normalize(x, p=2, dim=1)
        return x


class PalmVeinNet(nn.Module):
    """
    Complete palm vein recognition network.
    Lightweight version to prevent overfitting.
    """
    
    def __init__(
        self,
        backbone: str = "lightweight",
        pretrained: bool = True,
        embedding_dim: int = 256,
        dropout: float = 0.5
    ):
        super().__init__()
        
        # Feature extractors
        self.rgb_extractor = FeatureExtractor(
            in_channels=3,
            backbone=backbone,
            pretrained=pretrained
        )
        
        self.ir_extractor = FeatureExtractor(
            in_channels=1,
            backbone=backbone,
            pretrained=pretrained
        )
        
        feature_dim = self.rgb_extractor.feature_dim
        
        # Simple fusion
        self.fusion = SimpleFusion(
            feature_dim=feature_dim,
            output_dim=feature_dim,
            dropout=dropout
        )
        
        # Embedding head
        self.embedding_head = EmbeddingHead(
            input_dim=feature_dim,
            embedding_dim=embedding_dim,
            dropout=dropout
        )
        
        self.embedding_dim = embedding_dim
    
    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            rgb: RGB image (B, 3, H, W)
            ir: IR image (B, 1, H, W)
        
        Returns:
            Normalized embedding (B, embedding_dim)
        """
        rgb_features = self.rgb_extractor(rgb)
        ir_features = self.ir_extractor(ir)
        
        fused = self.fusion(rgb_features, ir_features)
        embedding = self.embedding_head(fused)
        
        return embedding


class Classifier(nn.Module):
    """
    Simple classifier head with label smoothing support.
    """
    
    def __init__(self, embedding_dim: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class PalmVeinClassifier(nn.Module):
    """
    Complete classification model.
    Combines backbone + classifier.
    """
    
    def __init__(
        self,
        num_classes: int,
        backbone: str = "lightweight",
        pretrained: bool = True,
        embedding_dim: int = 256,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.backbone = PalmVeinNet(
            backbone=backbone,
            pretrained=pretrained,
            embedding_dim=embedding_dim,
            dropout=dropout
        )
        
        self.classifier = Classifier(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            dropout=dropout
        )
        
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
    
    def forward(
        self,
        rgb: torch.Tensor,
        ir: torch.Tensor,
        return_embedding: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            Dictionary with 'logits' and optionally 'embeddings'
        """
        embeddings = self.backbone(rgb, ir)
        logits = self.classifier(embeddings)
        
        output = {'logits': logits, 'embeddings': embeddings}
        return output
    
    def get_embedding(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """Get embedding only (for verification)."""
        return self.backbone(rgb, ir)
