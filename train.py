"""
FEW-SHOT LEARNING TRAINING SCRIPT
==================================
Trains Prototypical Network for palm vein classification.

Designed for: 1500 classes × 2 images per class

Expected Results:
- Episode Accuracy: 70-85% (30-way 1-shot)
- Top-1 Full Eval: 35-50%
- Top-5 Full Eval: 70-80%
- Top-10 Full Eval: 82-90%
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import time
import json
import random

# Set CPU threads
NUM_CPU_THREADS = 16
os.environ["OMP_NUM_THREADS"] = str(NUM_CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_CPU_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(NUM_CPU_THREADS)

import torch
torch.set_num_threads(NUM_CPU_THREADS)

import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging

# ===========================================
# CONFIGURATION
# ===========================================
CONFIG = {
    # Paths
    'data_dir': 'final_folder',
    'save_dir': 'models_fewshot',
    'log_dir': 'logs_fewshot',
    
    # Few-shot settings
    'n_way': 30,              # Classes per episode
    'k_shot': 1,              # Support samples per class
    'q_query': 1,             # Query samples per class
    
    # Training
    'num_epochs': 200,
    'episodes_per_epoch': 100,
    'val_episodes': 50,
    
    # Model
    'backbone': 'custom',     # 'custom' or 'resnet18'
    'embedding_dim': 128,
    'dropout': 0.3,
    'temperature': 0.5,
    
    # Optimizer
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    'warmup_epochs': 10,
    'min_lr': 1e-6,
    
    # Early stopping
    'patience': 30,
    'min_delta': 0.1,
    
    # Checkpointing
    'save_every': 10,
    
    # System
    'device': 'cpu',
    'seed': 42,
    'val_split': 0.2,
    'image_size': (112, 112),
}
# ===========================================

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
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def log_progress(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    best_acc: float,
    epoch_time: float,
    status: str = "TRAINING"
):
    """Write progress to file."""
    with open(progress_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("FEW-SHOT PALM VEIN TRAINING\n")
        f.write(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Status: {status}\n")
        f.write("=" * 60 + "\n\n")
        
        # Progress bar
        progress = epoch / total_epochs
        bar_len = 40
        filled = int(bar_len * progress)
        bar = '█' * filled + '░' * (bar_len - filled)
        f.write(f"Progress: [{bar}] {progress*100:.1f}%\n")
        f.write(f"Epoch: {epoch}/{total_epochs}\n\n")
        
        # Episode settings
        f.write(f"Episode: {CONFIG['n_way']}-way {CONFIG['k_shot']}-shot\n\n")
        
        # Metrics
        f.write("Current Metrics (Episode Accuracy):\n")
        f.write(f"  Train Loss: {train_loss:.4f}\n")
        f.write(f"  Train Acc:  {train_acc:.2f}%\n")
        f.write(f"  Val Loss:   {val_loss:.4f}\n")
        f.write(f"  Val Acc:    {val_acc:.2f}%\n\n")
        
        f.write(f"Best Val Accuracy: {best_acc:.2f}%\n")
        f.write(f"Epoch Time: {epoch_time:.1f} minutes\n")
        
        remaining = (total_epochs - epoch) * epoch_time / 60
        f.write(f"Estimated Remaining: {remaining:.1f} hours\n")
        f.write("=" * 60 + "\n")
        
        # Expected final results
        f.write("\nExpected Final Results:\n")
        f.write("  Episode Accuracy: 70-85%\n")
        f.write("  Top-1 Classification: 35-50%\n")
        f.write("  Top-5 Classification: 70-80%\n")
        f.write("  Top-10 Classification: 82-90%\n")
        f.write("=" * 60 + "\n")


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int = 30, min_delta: float = 0.1):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.should_stop


# Import modules
from dataset import FewShotDataset, EpisodicSampler, create_data_loaders, Augmentation
from model import PrototypicalNetwork, PrototypicalLoss


def train_epoch(
    model: nn.Module,
    train_sampler: EpisodicSampler,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int
) -> tuple:
    """Train for one epoch (multiple episodes)."""
    model.train()
    
    total_loss = 0.0
    total_acc = 0.0
    num_episodes = 0
    
    for episode_idx, episode in enumerate(train_sampler):
        # Move to device
        support_rgb = episode['support_rgb'].to(device)
        support_ir = episode['support_ir'].to(device)
        support_labels = episode['support_labels'].to(device)
        query_rgb = episode['query_rgb'].to(device)
        query_ir = episode['query_ir'].to(device)
        query_labels = episode['query_labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits, _ = model(support_rgb, support_ir, support_labels, query_rgb, query_ir)
        
        # Compute loss
        loss, metrics = criterion(logits, query_labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += metrics['loss']
        total_acc += metrics['accuracy']
        num_episodes += 1
        
        # Log every 20 episodes
        if (episode_idx + 1) % 20 == 0:
            avg_loss = total_loss / num_episodes
            avg_acc = total_acc / num_episodes
            logger.info(f"Epoch {epoch} | Episode {episode_idx+1}/{len(train_sampler)} | "
                       f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.2f}%")
    
    return total_loss / num_episodes, total_acc / num_episodes


def validate(
    model: nn.Module,
    val_sampler: EpisodicSampler,
    criterion: nn.Module,
    device: str
) -> tuple:
    """Validate on episodes."""
    model.eval()
    
    total_loss = 0.0
    total_acc = 0.0
    num_episodes = 0
    
    with torch.no_grad():
        for episode in val_sampler:
            support_rgb = episode['support_rgb'].to(device)
            support_ir = episode['support_ir'].to(device)
            support_labels = episode['support_labels'].to(device)
            query_rgb = episode['query_rgb'].to(device)
            query_ir = episode['query_ir'].to(device)
            query_labels = episode['query_labels'].to(device)
            
            logits, _ = model(support_rgb, support_ir, support_labels, query_rgb, query_ir)
            loss, metrics = criterion(logits, query_labels)
            
            total_loss += metrics['loss']
            total_acc += metrics['accuracy']
            num_episodes += 1
    
    return total_loss / num_episodes, total_acc / num_episodes


def evaluate_full(
    model: nn.Module,
    dataset: FewShotDataset,
    device: str,
    num_test: int = 200
) -> dict:
    """
    Full evaluation: enroll all classes, test classification.
    """
    model.eval()
    model.clear_prototypes()
    
    # Enroll all classes (use first sample as prototype)
    logger.info("Enrolling all classes...")
    for class_idx in dataset.classes:
        sample = dataset.get_sample(class_idx, sample_idx=0)
        rgb = sample['rgb'].unsqueeze(0).to(device)
        ir = sample['ir'].unsqueeze(0).to(device)
        model.enroll(class_idx, sample['name'], rgb, ir)
    
    logger.info(f"Enrolled {len(model.prototypes)} classes")
    
    # Test on random samples
    top1_correct = 0
    top5_correct = 0
    top10_correct = 0
    total = 0
    
    # Sample random test cases
    test_samples = []
    for class_idx in dataset.classes:
        samples = dataset.class_to_samples[class_idx]
        if len(samples) > 1:
            # Use second sample for testing if available
            test_samples.append((class_idx, 1))
        else:
            # Otherwise use same sample (will give biased results)
            test_samples.append((class_idx, 0))
    
    random.shuffle(test_samples)
    test_samples = test_samples[:num_test]
    
    for class_idx, sample_idx in test_samples:
        sample = dataset.get_sample(class_idx, sample_idx)
        rgb = sample['rgb'].unsqueeze(0).to(device)
        ir = sample['ir'].unsqueeze(0).to(device)
        
        top_classes, top_scores = model.classify(rgb, ir, top_k=10)
        top_classes = top_classes.tolist()
        
        if top_classes[0] == class_idx:
            top1_correct += 1
        if class_idx in top_classes[:5]:
            top5_correct += 1
        if class_idx in top_classes[:10]:
            top10_correct += 1
        
        total += 1
    
    results = {
        'top1_accuracy': 100 * top1_correct / total if total > 0 else 0,
        'top5_accuracy': 100 * top5_correct / total if total > 0 else 0,
        'top10_accuracy': 100 * top10_correct / total if total > 0 else 0,
        'total_tested': total,
        'enrolled_classes': len(model.prototypes)
    }
    
    return results


def train():
    """Main training function."""
    start_time = datetime.now()
    set_seed(CONFIG['seed'])
    
    # Logging header
    logger.info("=" * 60)
    logger.info("FEW-SHOT PALM VEIN CLASSIFICATION")
    logger.info("=" * 60)
    logger.info(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Device: {CONFIG['device']}")
    logger.info("")
    logger.info("Configuration:")
    for key, value in CONFIG.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)
    
    save_path = Path(CONFIG['save_dir'])
    
    # Save config
    with open(save_path / "config.json", 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    # Create data loaders
    logger.info("\nLoading dataset...")
    train_sampler, val_sampler, dataset = create_data_loaders(
        data_dir=CONFIG['data_dir'],
        n_way=CONFIG['n_way'],
        k_shot=CONFIG['k_shot'],
        q_query=CONFIG['q_query'],
        train_episodes=CONFIG['episodes_per_epoch'],
        val_episodes=CONFIG['val_episodes'],
        image_size=CONFIG['image_size'],
        val_split=CONFIG['val_split'],
        seed=CONFIG['seed']
    )
    
    logger.info(f"Total classes: {dataset.get_num_classes()}")
    logger.info(f"Train episodes per epoch: {len(train_sampler)}")
    logger.info(f"Val episodes: {len(val_sampler)}")
    
    # Save metadata
    dataset.save_metadata(save_path / "dataset_metadata.json")
    
    # Create model
    logger.info("\nCreating Prototypical Network...")
    model = PrototypicalNetwork(
        embedding_dim=CONFIG['embedding_dim'],
        dropout=CONFIG['dropout'],
        backbone=CONFIG['backbone'],
        pretrained=True,
        temperature=CONFIG['temperature']
    )
    model = model.to(CONFIG['device'])
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = PrototypicalLoss(temperature=CONFIG['temperature'])
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # Scheduler with warmup
    def lr_lambda(epoch):
        if epoch < CONFIG['warmup_epochs']:
            return (epoch + 1) / CONFIG['warmup_epochs']
        else:
            progress = (epoch - CONFIG['warmup_epochs']) / (CONFIG['num_epochs'] - CONFIG['warmup_epochs'])
            return max(CONFIG['min_lr'] / CONFIG['learning_rate'],
                      0.5 * (1 + np.cos(np.pi * progress)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=CONFIG['patience'],
        min_delta=CONFIG['min_delta']
    )
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'lr': [], 'epoch_time': []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    
    logger.info("\n" + "=" * 60)
    logger.info("STARTING TRAINING")
    logger.info(f"Episode: {CONFIG['n_way']}-way {CONFIG['k_shot']}-shot")
    logger.info("=" * 60)
    
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_sampler, criterion, optimizer, CONFIG['device'], epoch
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_sampler, criterion, CONFIG['device']
        )
        
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
        
        # Log epoch summary
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"EPOCH {epoch}/{CONFIG['num_epochs']} COMPLETED")
        logger.info("-" * 60)
        logger.info(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        logger.info(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        logger.info(f"  LR: {current_lr:.6f} | Time: {epoch_time:.1f} min")
        logger.info(f"  Best: {best_val_acc:.2f}% (Epoch {best_epoch})")
        
        remaining = (CONFIG['num_epochs'] - epoch) * np.mean(history['epoch_time']) / 60
        logger.info(f"  Remaining: {remaining:.1f} hours")
        logger.info("=" * 60)
        
        # Update progress file
        log_progress(
            epoch, CONFIG['num_epochs'],
            train_loss, train_acc, val_loss, val_acc,
            best_val_acc, epoch_time, "TRAINING"
        )
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': CONFIG
            }, save_path / "best_model.pth")
            
            logger.info(f"★★★ NEW BEST MODEL! Val Acc: {val_acc:.2f}% ★★★")
        
        # Save checkpoint
        if epoch % CONFIG['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'config': CONFIG
            }, save_path / f"checkpoint_epoch_{epoch}.pth")
        
        # Save history
        with open(save_path / "history.json", 'w') as f:
            json.dump(history, f, indent=2)
        
        # Early stopping
        if early_stopping(val_acc):
            logger.warning(f"\nEarly stopping at epoch {epoch}")
            break
    
    # Final evaluation
    logger.info("\n" + "=" * 60)
    logger.info("FINAL FULL EVALUATION")
    logger.info("=" * 60)
    
    # Load best model
    checkpoint = torch.load(save_path / "best_model.pth", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Full evaluation
    eval_results = evaluate_full(model, dataset, CONFIG['device'], num_test=300)
    
    logger.info(f"Top-1 Accuracy:  {eval_results['top1_accuracy']:.2f}%")
    logger.info(f"Top-5 Accuracy:  {eval_results['top5_accuracy']:.2f}%")
    logger.info(f"Top-10 Accuracy: {eval_results['top10_accuracy']:.2f}%")
    logger.info(f"Total tested: {eval_results['total_tested']}")
    
    # Save final results
    final_results = {
        'episode_accuracy': best_val_acc,
        'top1_accuracy': eval_results['top1_accuracy'],
        'top5_accuracy': eval_results['top5_accuracy'],
        'top10_accuracy': eval_results['top10_accuracy'],
        'best_epoch': best_epoch,
        'total_epochs': epoch,
        'config': CONFIG
    }
    
    with open(save_path / "final_results.json", 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Final summary
    total_time = (datetime.now() - start_time).total_seconds() / 3600
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETED!")
    logger.info("=" * 60)
    logger.info(f"Total Time: {total_time:.2f} hours")
    logger.info(f"Best Episode Accuracy: {best_val_acc:.2f}%")
    logger.info(f"Best Epoch: {best_epoch}")
    logger.info("-" * 60)
    logger.info("FINAL CLASSIFICATION ACCURACY:")
    logger.info(f"  Top-1:  {eval_results['top1_accuracy']:.2f}%")
    logger.info(f"  Top-5:  {eval_results['top5_accuracy']:.2f}%")
    logger.info(f"  Top-10: {eval_results['top10_accuracy']:.2f}%")
    logger.info("=" * 60)
    
    # Update progress file
    log_progress(
        epoch, CONFIG['num_epochs'],
        train_loss, train_acc, val_loss, val_acc,
        best_val_acc, epoch_time, "COMPLETED"
    )
    
    return final_results


if __name__ == "__main__":
    try:
        results = train()
        print("\n" + "=" * 60)
        print("FINAL RESULTS SUMMARY")
        print("=" * 60)
        print(f"Top-1:  {results['top1_accuracy']:.2f}%")
        print(f"Top-5:  {results['top5_accuracy']:.2f}%")
        print(f"Top-10: {results['top10_accuracy']:.2f}%")
        print("=" * 60)
    except KeyboardInterrupt:
        logger.warning("\n*** Training interrupted ***")
    except Exception as e:
        logger.error(f"\n*** Training failed: {e} ***")
        import traceback
        logger.error(traceback.format_exc())
        raise
