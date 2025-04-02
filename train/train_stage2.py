import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import yaml
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from models.audio_encoder import AudioEncoder
from models.visual_encoder import VisualEncoder
from models.cross_modal import CrossModalFusion
from models.classifier import Classifier
from utils.logger import setup_logger
from utils.data_utils import create_data_loaders

def train_stage2(config_path: str, stage1_checkpoint: str):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logger
    logger = setup_logger(config['training']['log_dir'])
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize models
    audio_encoder = AudioEncoder().to(device)
    visual_encoder = VisualEncoder().to(device)
    cross_modal = CrossModalFusion().to(device)
    classifier = Classifier().to(device)
    
    # Load stage1 checkpoint
    checkpoint = torch.load(stage1_checkpoint)
    audio_encoder.load_state_dict(checkpoint['audio_encoder'])
    visual_encoder.load_state_dict(checkpoint['visual_encoder'])
    cross_modal.load_state_dict(checkpoint['cross_modal'])
    
    # Freeze pre-trained models
    audio_encoder.eval()
    visual_encoder.eval()
    cross_modal.eval()
    
    # Setup optimizer (only for classifier)
    optimizer = optim.Adam(classifier.parameters(), 
                          lr=config['training']['learning_rate'])
    
    # Setup loss
    criterion = nn.BCEWithLogitsLoss()
    
    # Setup tensorboard
    writer = SummaryWriter(config['training']['log_dir'])
    
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
    
    # Training loop
    best_val_f1 = 0
    patience_counter = 0
    
    for epoch in range(config['training']['num_epochs']):
        # Training phase
        classifier.train()
        
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for batch_idx, (frames, audio, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}')):
            frames = frames.to(device)
            audio = audio.to(device)
            labels = labels.to(device)
            
            # Forward pass
            with torch.no_grad():
                audio_features = audio_encoder(audio)
                visual_features = visual_encoder(frames)
                fused_features = cross_modal(visual_features, audio_features)
            
            logits = classifier(fused_features)
            loss = criterion(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate metrics
            train_loss += loss.item()
            train_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            # Log batch metrics
            if batch_idx % 10 == 0:
                writer.add_scalar('Batch/Loss', loss.item(), epoch * len(train_loader) + batch_idx)
        
        # Compute training metrics
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        train_prec = precision_score(train_labels, train_preds)
        train_rec = recall_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds)
        
        # Validation phase
        classifier.eval()
        
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for frames, audio, labels in val_loader:
                frames = frames.to(device)
                audio = audio.to(device)
                labels = labels.to(device)
                
                # Forward pass
                audio_features = audio_encoder(audio)
                visual_features = visual_encoder(frames)
                fused_features = cross_modal(visual_features, audio_features)
                logits = classifier(fused_features)
                
                loss = criterion(logits, labels)
                
                # Accumulate metrics
                val_loss += loss.item()
                val_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Compute validation metrics
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_prec = precision_score(val_labels, val_preds)
        val_rec = recall_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)
        
        # Log epoch metrics
        writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
        writer.add_scalar('Epoch/Train_Accuracy', train_acc, epoch)
        writer.add_scalar('Epoch/Train_Precision', train_prec, epoch)
        writer.add_scalar('Epoch/Train_Recall', train_rec, epoch)
        writer.add_scalar('Epoch/Train_F1', train_f1, epoch)
        writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
        writer.add_scalar('Epoch/Val_Accuracy', val_acc, epoch)
        writer.add_scalar('Epoch/Val_Precision', val_prec, epoch)
        writer.add_scalar('Epoch/Val_Recall', val_rec, epoch)
        writer.add_scalar('Epoch/Val_F1', val_f1, epoch)
        
        # Log to console
        logger.info(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
        logger.info(f'Train Metrics - Acc: {train_acc:.4f}, Prec: {train_prec:.4f}, Rec: {train_rec:.4f}, F1: {train_f1:.4f}')
        logger.info(f'Val Metrics - Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}')
        
        # Early stopping
        if val_f1 > best_val_f1 + config['training']['early_stopping_min_delta']:
            best_val_f1 = val_f1
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'classifier': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_f1': val_f1
            }
            torch.save(checkpoint, os.path.join(config['training']['checkpoint_dir'], 'stage2_best.pth'))
        else:
            patience_counter += 1
            if patience_counter >= config['training']['early_stopping_patience']:
                logger.info(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    writer.close()
    return best_val_f1

if __name__ == '__main__':
    train_stage2('configs/config.yaml', 'checkpoints/stage1_model.pth') 