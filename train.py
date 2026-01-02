"""
ANTI-OVERFITTING TRAINING SCRIPT
=================================
Complete training with all safeguards to prevent overfitting.

Safeguards:
1. Early stopping when val loss increases
2. Stop if val_acc stays 0% for N epochs
3. Stop if train-val accuracy gap too large
4. High dropout (0.5)
5. Strong augmentation
6. Label smoothing (0.2)
7. Weight decay (0.01)
8. Lightweight model
9. Detailed logging

Expected Results (1500 people × 2 images):
- Classification: 55-65%
- Verification: 82-88%
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import time
import json
import random

# Set CPU threads BEFORE importing torch
NUM_CPU_CORES = 16
os.environ["OMP_NUM_THREADS"] = str(NUM_CPU_CORES)
os.environ["MKL_NUM_THREADS"] = str(NUM_CPU_CORES)
os.environ["NUMEXPR_NUM_THREADS"] = str(NUM_CPU_CORES)

sys.path.insert(0, str(Path(__file__).parent))

import torch
torch.set_num_threads(NUM_CPU_CORES)

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging

# ============================================
# CONFIGURATION
# ============================================
CONFIG = {
    # Paths
    'data_dir': 'final_folder',
    'save_dir': 'models_v2',
    'log_dir': 'logs_v2',
    
    # Training
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 0.01,
    'label_smoothing': 0.2,
    
    # Model
    'backbone': 'lightweight',  # 'lightweight' or 'efficientnet_b0'
    'embedding_dim': 256,
    'dropout': 0.5,
    
    # Anti-overfitting
    'early_stopping_patience': 15,
    'overfit_check_epochs': 5,      # Stop if val_acc=0 for this many epochs
    'max_train_val_gap': 30.0,      # Stop if train_acc - val_acc > this
    
    # Scheduler
    'warmup_epochs': 5,
    'min_lr': 1e-6,
    
    # Checkpointing
    'save_every': 5,
    
    # Validation
    'val_split': 0.2,
    
    # Device
    'device': 'cpu',
    'num_workers': 4,
    
    # Seed
    'seed': 42
}
# ============================================

# Create directories
Path(CONFIG['save_dir']).mkdir(parents=True, exist_ok=True)
Path(CONFIG['log_dir']).mkdir(parents=True, exist_ok=True)

# Setup logging
log_filename = f"{CONFIG['log_dir']}/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Progress file
progress_file = f"{CONFIG['log_dir']}/progress.txt"


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_progress(epoch, epochs, train_loss, train_acc, val_loss, val_acc, 
                 epoch_time, best_acc, status="TRAINING"):
    """Write progress to file for quick checking."""
    with open(progress_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("PALM VEIN TRAINING PROGRESS (Anti-Overfitting)\n")
        f.write(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Status: {status}\n")
        f.write("=" * 60 + "\n\n")
        
        # Progress bar
        progress = epoch / epochs
        bar_len = 40
        filled = int(bar_len * progress)
        bar = '█' * filled + '░' * (bar_len - filled)
        f.write(f"Progress: [{bar}] {progress*100:.1f}%\n")
        f.write(f"Epoch: {epoch}/{epochs}\n\n")
        
        # Metrics
        f.write("Current Metrics:\n")
        f.write(f"  Train Loss: {train_loss:.4f}\n")
        f.write(f"  Train Acc:  {train_acc:.2f}%\n")
        f.write(f"  Val Loss:   {val_loss:.4f}\n")
        f.write(f"  Val Acc:    {val_acc:.2f}%\n\n")
        
        # Overfitting check
        gap = train_acc - val_acc
        f.write("Overfitting Check:\n")
        f.write(f"  Train-Val Gap: {gap:.2f}%")
        if gap > CONFIG['max_train_val_gap']:
            f.write(" ⚠️ HIGH\n")
        elif gap > 15:
            f.write(" ⚡ MODERATE\n")
        else:
            f.write(" ✓ OK\n")
        f.write("\n")
        
        # Best and timing
        f.write(f"Best Val Accuracy: {best_acc:.2f}%\n")
        f.write(f"Epoch Time: {epoch_time:.1f} minutes\n")
        remaining = (epochs - epoch) * epoch_time / 60
        f.write(f"Estimated Remaining: {remaining:.1f} hours\n")
        f.write("=" * 60 + "\n")


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    Stops training when validation loss doesn't improve.
    """
    
    def __init__(self, patience: int = 15, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.should_stop


class OverfitDetector:
    """
    Detects overfitting and training problems.
    """
    
    def __init__(
        self,
        zero_val_patience: int = 5,
        max_gap: float = 30.0
    ):
        self.zero_val_patience = zero_val_patience
        self.max_gap = max_gap
        self.zero_val_counter = 0
        self.status = "OK"
        self.message = ""
    
    def check(self, train_acc: float, val_acc: float) -> Tuple[bool, str]:
        """
        Check for overfitting or training problems.
        
        Returns:
            should_stop: True if training should stop
            message: Reason for stopping
        """
        # Check for 0% validation accuracy
        if val_acc < 0.1:  # Less than 0.1%
            self.zero_val_counter += 1
            if self.zero_val_counter >= self.zero_val_patience:
                self.status = "FAILED"
                self.message = f"Validation accuracy stayed near 0% for {self.zero_val_patience} epochs. Model not learning."
                return True, self.message
        else:
            self.zero_val_counter = 0
        
        # Check train-val gap
        gap = train_acc - val_acc
        if gap > self.max_gap and train_acc > 20:
            self.status = "OVERFIT"
            self.message = f"Train-Val gap ({gap:.1f}%) exceeds maximum ({self.max_gap}%). Severe overfitting."
            return True, self.message
        
        self.status = "OK"
        self.message = ""
        return False, ""


from data.dataset import PalmVeinDataset, StrongAugmentation, create_dataloaders
from models.networks import PalmVeinClassifier
from models.losses import SimpleLoss


def train():
    """Main training function with all safeguards."""
    
    start_time = datetime.now()
    set_seed(CONFIG['seed'])
    
    # Logging header
    logger.info("=" * 60)
    logger.info("PALM VEIN TRAINING - ANTI-OVERFITTING VERSION")
    logger.info("=" * 60)
    logger.info(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Device: {CONFIG['device']}")
    logger.info(f"CPU Cores: {NUM_CPU_CORES}")
    logger.info("")
    logger.info("Configuration:")
    for key, value in CONFIG.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)
    
    # Save config
    save_path = Path(CONFIG['save_dir'])
    with open(save_path / "config.json", 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    # Create dataloaders
    logger.info("\nLoading dataset...")
    train_loader, val_loader, dataset = create_dataloaders(
        data_dir=CONFIG['data_dir'],
        batch_size=CONFIG['batch_size'],
        target_size=(224, 224),
        val_split=CONFIG['val_split'],
        num_workers=CONFIG['num_workers'],
        seed=CONFIG['seed']
    )
    
    num_classes = dataset.get_num_classes()
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    
    # Save metadata
    dataset.save_metadata(save_path / "dataset_metadata.json")
    
    # Create model
    logger.info("\nCreating model...")
    model = PalmVeinClassifier(
        num_classes=num_classes,
        backbone=CONFIG['backbone'],
        pretrained=True,
        embedding_dim=CONFIG['embedding_dim'],
        dropout=CONFIG['dropout']
    )
    model = model.to(CONFIG['device'])
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function
    criterion = SimpleLoss(
        num_classes=num_classes,
        label_smoothing=CONFIG['label_smoothing'],
        use_focal=False
    )
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < CONFIG['warmup_epochs']:
            return (epoch + 1) / CONFIG['warmup_epochs']
        else:
            progress = (epoch - CONFIG['warmup_epochs']) / (CONFIG['epochs'] - CONFIG['warmup_epochs'])
            return max(CONFIG['min_lr'] / CONFIG['learning_rate'], 
                      0.5 * (1 + np.cos(np.pi * progress)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Early stopping and overfit detection
    early_stopping = EarlyStopping(patience=CONFIG['early_stopping_patience'])
    overfit_detector = OverfitDetector(
        zero_val_patience=CONFIG['overfit_check_epochs'],
        max_gap=CONFIG['max_train_val_gap']
    )
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'lr': [], 'epoch_time': []
    }
    
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_epoch = 0
    
    logger.info("\n" + "=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)
    
    for epoch in range(CONFIG['epochs']):
        epoch_start = time.time()
        
        # ==================== TRAINING ====================
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            rgb = batch['rgb'].to(CONFIG['device'])
            ir = batch['ir'].to(CONFIG['device'])
            labels = batch['label'].to(CONFIG['device'])
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(rgb, ir)
            logits = output['logits']
            embeddings = output['embeddings']
            
            # Compute loss
            loss, loss_dict = criterion(logits, embeddings, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * rgb.size(0)
            _, predicted = logits.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)
            
            # Progress logging
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                current_acc = 100. * train_correct / train_total
                logger.info(f"Epoch {epoch+1}/{CONFIG['epochs']} | "
                           f"Batch {batch_idx+1}/{len(train_loader)} | "
                           f"Loss: {loss.item():.4f} | Acc: {current_acc:.2f}%")
        
        train_loss /= train_total
        train_acc = 100. * train_correct / train_total
        
        # ==================== VALIDATION ====================
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                rgb = batch['rgb'].to(CONFIG['device'])
                ir = batch['ir'].to(CONFIG['device'])
                labels = batch['label'].to(CONFIG['device'])
                
                output = model(rgb, ir)
                logits = output['logits']
                embeddings = output['embeddings']
                
                loss, _ = criterion(logits, embeddings, labels)
                
                val_loss += loss.item() * rgb.size(0)
                _, predicted = logits.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)
        
        val_loss /= val_total
        val_acc = 100. * val_correct / val_total
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Epoch time
        epoch_time = (time.time() - epoch_start) / 60
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        history['epoch_time'].append(epoch_time)
        
        # Calculate gap
        gap = train_acc - val_acc
        
        # Log epoch summary
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"EPOCH {epoch+1}/{CONFIG['epochs']} COMPLETED")
        logger.info("-" * 60)
        logger.info(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        logger.info(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        logger.info(f"  Gap:        {gap:.2f}% | LR: {current_lr:.6f}")
        logger.info(f"  Time:       {epoch_time:.1f} min")
        logger.info(f"  Best:       {best_val_acc:.2f}% (Epoch {best_epoch})")
        
        # Remaining time
        avg_time = np.mean(history['epoch_time'])
        remaining = (CONFIG['epochs'] - epoch - 1) * avg_time / 60
        logger.info(f"  Remaining:  {remaining:.1f} hours")
        logger.info("=" * 60)
        
        # Update progress file
        log_progress(epoch+1, CONFIG['epochs'], train_loss, train_acc, 
                    val_loss, val_acc, epoch_time, best_val_acc, "TRAINING")
        
        # ==================== OVERFITTING CHECKS ====================
        
        # Check 1: Zero validation accuracy
        should_stop, message = overfit_detector.check(train_acc, val_acc)
        if should_stop:
            logger.error("")
            logger.error("!" * 60)
            logger.error("TRAINING STOPPED - PROBLEM DETECTED")
            logger.error(message)
            logger.error("!" * 60)
            log_progress(epoch+1, CONFIG['epochs'], train_loss, train_acc,
                        val_loss, val_acc, epoch_time, best_val_acc, f"STOPPED: {overfit_detector.status}")
            break
        
        # Check 2: Early stopping
        if early_stopping(val_loss):
            logger.warning("")
            logger.warning("=" * 60)
            logger.warning("EARLY STOPPING TRIGGERED")
            logger.warning(f"Validation loss hasn't improved for {CONFIG['early_stopping_patience']} epochs")
            logger.warning("=" * 60)
            log_progress(epoch+1, CONFIG['epochs'], train_loss, train_acc,
                        val_loss, val_acc, epoch_time, best_val_acc, "EARLY STOPPED")
            break
        
        # ==================== SAVE BEST MODEL ====================
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch + 1
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'num_classes': num_classes,
                'config': CONFIG
            }, save_path / "best_model.pth")
            
            logger.info(f"★★★ NEW BEST MODEL SAVED! Val Acc: {val_acc:.2f}% ★★★")
        
        # Save checkpoint
        if (epoch + 1) % CONFIG['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'num_classes': num_classes
            }, save_path / f"checkpoint_epoch_{epoch+1}.pth")
            logger.info(f"Checkpoint saved: checkpoint_epoch_{epoch+1}.pth")
        
        # Save history
        with open(save_path / "history.json", 'w') as f:
            json.dump(history, f, indent=2)
    
    # ==================== TRAINING COMPLETE ====================
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds() / 3600
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'config': CONFIG
    }, save_path / "final_model.pth")
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED!")
    logger.info("=" * 60)
    logger.info(f"Total Time: {total_time:.2f} hours")
    logger.info(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    logger.info(f"Best Epoch: {best_epoch}")
    logger.info(f"Final Train Acc: {train_acc:.2f}%")
    logger.info(f"Final Val Acc: {val_acc:.2f}%")
    logger.info(f"Models saved in: {save_path}")
    logger.info("=" * 60)
    
    # Final progress update
    log_progress(epoch+1, CONFIG['epochs'], train_loss, train_acc,
                val_loss, val_acc, epoch_time, best_val_acc, "COMPLETED")
    
    return best_val_acc


if __name__ == "__main__":
    from typing import Tuple
    
    try:
        best_acc = train()
        logger.info(f"\nFinal best accuracy: {best_acc:.2f}%")
    except KeyboardInterrupt:
        logger.warning("\n*** Training interrupted by user ***")
    except Exception as e:
        logger.error(f"\n*** Training failed: {e} ***")
        import traceback
        logger.error(traceback.format_exc())
        raise
