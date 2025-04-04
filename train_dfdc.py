import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss, f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import yaml
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from typing import Dict, List, Tuple
import time
import json

from avff.datasets.dfdc import DFDCDataset
from avff.models import AVFFModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

def setup_device():
    if torch.cuda.is_available():
        # Get the current device
        device = torch.cuda.current_device()
        # Get device properties
        props = torch.cuda.get_device_properties(device)
        logging.info(f"Using GPU: {props.name}")
        logging.info(f"Total GPU memory: {props.total_memory / 1024**2:.0f}MB")
        logging.info(f"CUDA version: {torch.version.cuda}")
        logging.info(f"Number of CUDA cores: {props.multi_processor_count}")
        
        # Set memory allocation strategy
        torch.cuda.empty_cache()
        
        # Enable cuDNN benchmarking for faster training
        torch.backends.cudnn.benchmark = True
        
        # Enable TF32 for better performance on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        return torch.device("cuda")
    else:
        logging.warning("CUDA is not available. Using CPU.")
        return torch.device("cpu")

def train_model(model, train_loader, val_loader, num_epochs, device, output_dir, config, start_epoch=0):
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize metrics storage
    train_metrics = {
        'loss': [], 'f1': [], 'precision': [], 'recall': [], 'accuracy': []
    }
    val_metrics = {
        'loss': [], 'f1': [], 'precision': [], 'recall': [], 'accuracy': []
    }
    
    # Initialize optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config['training']['scheduler']['T_0'],
        T_mult=config['training']['scheduler']['T_mult']
    )
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping']['patience'],
        min_delta=config['training']['early_stopping']['min_delta']
    )
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        logging.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0
        train_batches = 0
        train_preds = []
        train_labels = []
        
        # Initialize progress tracking
        start_time = time.time()
        last_log_time = start_time
        train_progress = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(train_progress):
            # Forward pass
            outputs = model(batch['audio_features'].to(device), batch['video_frames'].to(device))
            labels = batch['label'].to(device)
            
            # Compute loss
            loss = nn.CrossEntropyLoss()(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            if (batch_idx + 1) % config['training']['gradient_accumulation_steps'] == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Update metrics
            train_loss += loss.item()
            train_batches += 1
            
            # Store predictions and labels
            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            # Clear intermediate variables
            del outputs, loss
            torch.cuda.empty_cache()
            
            # Calculate processing speed and memory usage
            current_time = time.time()
            if current_time - last_log_time > 5.0:  # Log every 5 seconds
                elapsed_time = current_time - start_time
                samples_processed = batch_idx * config['training']['batch_size']
                samples_per_second = samples_processed / elapsed_time
                
                gpu_memory_used = torch.cuda.memory_allocated() / 1024**2
                gpu_memory_cached = torch.cuda.memory_reserved() / 1024**2
                
                logging.info(
                    f"Speed: {samples_per_second:.1f} samples/sec | "
                    f"GPU Memory: Used={gpu_memory_used:.0f}MB, Cached={gpu_memory_cached:.0f}MB | "
                    f"Loss: {train_loss/train_batches:.4f}"
                )
                last_log_time = current_time
            
            # Update progress bar
            train_progress.set_postfix({
                'loss': f"{train_loss/train_batches:.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
        
        # Calculate training metrics
        train_metrics['loss'].append(train_loss / train_batches)
        train_metrics['f1'].append(f1_score(train_labels, train_preds))
        train_metrics['precision'].append(precision_score(train_labels, train_preds))
        train_metrics['recall'].append(recall_score(train_labels, train_preds))
        train_metrics['accuracy'].append(accuracy_score(train_labels, train_preds))
        
        # Log training metrics
        logging.info(f"Training metrics - Loss: {train_metrics['loss'][-1]:.4f}, "
                    f"F1: {train_metrics['f1'][-1]:.4f}, "
                    f"Precision: {train_metrics['precision'][-1]:.4f}, "
                    f"Recall: {train_metrics['recall'][-1]:.4f}, "
                    f"Accuracy: {train_metrics['accuracy'][-1]:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_batches = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                outputs = model(batch['audio_features'].to(device), batch['video_frames'].to(device))
                labels = batch['label'].to(device)
                
                loss = nn.CrossEntropyLoss()(outputs, labels)
                val_loss += loss.item()
                val_batches += 1
                
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        val_metrics['loss'].append(val_loss / val_batches)
        val_metrics['f1'].append(f1_score(val_labels, val_preds))
        val_metrics['precision'].append(precision_score(val_labels, val_preds))
        val_metrics['recall'].append(recall_score(val_labels, val_preds))
        val_metrics['accuracy'].append(accuracy_score(val_labels, val_preds))
        
        # Log validation metrics
        logging.info(f"Validation metrics - Loss: {val_metrics['loss'][-1]:.4f}, "
                    f"F1: {val_metrics['f1'][-1]:.4f}, "
                    f"Precision: {val_metrics['precision'][-1]:.4f}, "
                    f"Recall: {val_metrics['recall'][-1]:.4f}, "
                    f"Accuracy: {val_metrics['accuracy'][-1]:.4f}")
        
        # Update learning rate
        scheduler.step()
        
        # Save model checkpoint
        if val_metrics['f1'][-1] > max(val_metrics['f1'][:-1] or [0]):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'config': config
            }, os.path.join(output_dir, 'best_model.pth'))
            logging.info("Saved best model checkpoint")
        
        # Plot metrics
        plot_metrics(train_metrics, val_metrics, output_dir)
        
        # Check for early stopping
        if early_stopping(val_metrics['loss'][-1]):
            logging.info("Early stopping triggered")
            break
        
        # Clear memory at end of epoch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return train_metrics, val_metrics

def plot_metrics(train_metrics: Dict[str, List[float]], val_metrics: Dict[str, List[float]], output_dir: str):
    """Plot and save training metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics', fontsize=16)
    
    # Plot loss
    axes[0, 0].plot(train_metrics['loss'], label='Train')
    axes[0, 0].plot(val_metrics['loss'], label='Validation')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot F1 score
    axes[0, 1].plot(train_metrics['f1'], label='Train')
    axes[0, 1].plot(val_metrics['f1'], label='Validation')
    axes[0, 1].set_title('F1 Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot Precision
    axes[1, 0].plot(train_metrics['precision'], label='Train')
    axes[1, 0].plot(val_metrics['precision'], label='Validation')
    axes[1, 0].set_title('Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot Recall
    axes[1, 1].plot(train_metrics['recall'], label='Train')
    axes[1, 1].plot(val_metrics['recall'], label='Validation')
    axes[1, 1].set_title('Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'))
    plt.close()
    
    # Save metrics to JSON
    metrics_data = {
        'train': train_metrics,
        'val': val_metrics
    }
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_data, f, indent=4)

def evaluate_model(model, test_loader, device):
    """Evaluate model on test data."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            video_data = batch['video'].to(device)
            audio_data = batch['audio'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(video_data, audio_data)
            preds = (outputs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds)
    }
    
    return metrics

def main():
    # Load config
    with open('configs/dfdc_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = setup_device()
    
    # Initialize datasets
    train_dataset = DFDCDataset(
        root_dir=config['data']['train_videos'],
        metadata_path=config['data']['train_labels'],
        split='train'
    )
    
    val_dataset = DFDCDataset(
        root_dir=config['data']['train_videos'],
        metadata_path=config['data']['train_labels'],
        split='val'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    # Initialize model
    model = AVFFModel(
        num_classes=config['model']['num_classes'],
        audio_backbone=config['model']['audio_backbone'],
        visual_backbone=config['model']['visual_backbone'],
        fusion_dim=config['model']['fusion_dim'],
        dropout=config['model']['dropout']
    ).to(device)
    
    # Train model
    train_metrics, val_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs'],
        device=device,
        output_dir=config['training']['output_dir'],
        config=config
    )
    
    # Plot metrics
    plot_metrics(train_metrics, val_metrics, config['training']['output_dir'])
    
    # Save final metrics
    metrics_data = {
        'train': train_metrics,
        'val': val_metrics
    }
    with open(os.path.join(config['training']['output_dir'], 'final_metrics.json'), 'w') as f:
        json.dump(metrics_data, f, indent=4)

if __name__ == '__main__':
    main() 