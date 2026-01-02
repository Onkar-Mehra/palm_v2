"""
Loss Functions - Simple and Stable
===================================
Using simple CrossEntropy with label smoothing to prevent overfitting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy with Label Smoothing.
    
    Label smoothing prevents the model from becoming too confident,
    which helps prevent overfitting.
    """
    
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(1)
        
        # Create smoothed labels
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        # Compute loss
        log_probs = F.log_softmax(pred, dim=1)
        loss = -torch.sum(true_dist * log_probs, dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Reduces loss for well-classified examples.
    """
    
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SimpleLoss(nn.Module):
    """
    Simple combined loss: CrossEntropy with Label Smoothing.
    No complex losses that can cause training instability.
    """
    
    def __init__(
        self,
        num_classes: int,
        label_smoothing: float = 0.2,
        use_focal: bool = False,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_focal = use_focal
        
        if use_focal:
            self.criterion = FocalLoss(gamma=focal_gamma)
        else:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    
    def forward(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss.
        
        Args:
            logits: Classification logits (B, num_classes)
            embeddings: Feature embeddings (B, embedding_dim) - not used but kept for compatibility
            labels: Ground truth labels (B,)
        
        Returns:
            loss: Total loss
            loss_dict: Dictionary with loss breakdown
        """
        loss = self.criterion(logits, labels)
        
        loss_dict = {
            'total': loss.item(),
            'ce': loss.item()
        }
        
        return loss, loss_dict
