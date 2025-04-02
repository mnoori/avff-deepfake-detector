import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from models.avff_model import get_model
from utils.data_utils import create_data_loaders

def parse_args():
    parser = argparse.ArgumentParser(description='Train AVFF model for deepfake detection')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                      help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                      help='Path to checkpoint to resume from')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for frames, audio, labels in tqdm(train_loader, desc='Training'):
        frames = frames.to(device)
        audio = audio.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(frames, audio)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = (outputs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    metrics = {
        'loss': total_loss / len(train_loader),
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds)
    }
    
    return metrics

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for frames, audio, labels in tqdm(val_loader, desc='Validation'):
            frames = frames.to(device)
            audio = audio.to(device)
            labels = labels.to(device)
            
            outputs = model(frames, audio)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    metrics = {
        'loss': total_loss / len(val_loader),
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds)
    }
    
    return metrics

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = get_model(num_classes=1)
    model = model.to(device)
    
    # Load checkpoint if specified
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_videos=config['data']['train_videos'],
        train_labels=config['data']['train_labels'],
        val_videos=config['data']['val_videos'],
        val_labels=config['data']['val_labels'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        frame_count=config['data']['frame_count'],
        image_size=tuple(config['data']['image_size']),
        audio_length=config['data']['audio_length'],
        sample_rate=config['data']['sample_rate']
    )
    
    # Setup training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
        verbose=True
    )
    
    # Setup logging
    writer = SummaryWriter(config['training']['log_dir'])
    
    # Training loop
    best_val_f1 = 0
    for epoch in range(config['training']['num_epochs']):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Log metrics
        for metric, value in train_metrics.items():
            writer.add_scalar(f'train/{metric}', value, epoch)
        for metric, value in val_metrics.items():
            writer.add_scalar(f'val/{metric}', value, epoch)
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Save checkpoint if best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics
            }
            torch.save(checkpoint, os.path.join(config['training']['checkpoint_dir'], 'best_model.pth'))
        
        # Print metrics
        print(f'Epoch {epoch}:')
        print(f'Train - Loss: {train_metrics["loss"]:.4f}, F1: {train_metrics["f1"]:.4f}')
        print(f'Val - Loss: {val_metrics["loss"]:.4f}, F1: {val_metrics["f1"]:.4f}')

if __name__ == '__main__':
    main() 