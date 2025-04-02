import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import yaml
from tqdm import tqdm
import numpy as np

from models.audio_encoder import AudioEncoder
from models.visual_encoder import VisualEncoder
from models.cross_modal import CrossModalFusion
from models.decoder import Decoder
from utils.logger import setup_logger
from utils.data_utils import create_data_loaders

def train_stage1(config_path: str):
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
    decoder = Decoder().to(device)
    
    # Setup optimizer
    optimizer = optim.Adam([
        {'params': audio_encoder.parameters()},
        {'params': visual_encoder.parameters()},
        {'params': cross_modal.parameters()},
        {'params': decoder.parameters()}
    ], lr=config['training']['learning_rate'])
    
    # Setup losses
    contrastive_loss = nn.CrossEntropyLoss()
    reconstruction_loss = nn.MSELoss()
    
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
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['training']['num_epochs']):
        # Training phase
        audio_encoder.train()
        visual_encoder.train()
        cross_modal.train()
        decoder.train()
        
        train_loss = 0
        train_recon_loss = 0
        train_contrast_loss = 0
        
        for batch_idx, (frames, audio, _) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}')):
            frames = frames.to(device)
            audio = audio.to(device)
            
            # Forward pass
            audio_features = audio_encoder(audio)
            visual_features = visual_encoder(frames)
            fused_features = cross_modal(visual_features, audio_features)
            reconstructed = decoder(fused_features)
            
            # Compute losses
            recon_loss = reconstruction_loss(reconstructed, audio)
            contrast_loss = contrastive_loss(fused_features, audio_features)
            loss = recon_loss + contrast_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate losses
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_contrast_loss += contrast_loss.item()
            
            # Log batch metrics
            if batch_idx % 10 == 0:
                writer.add_scalar('Batch/Loss', loss.item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Batch/Reconstruction_Loss', recon_loss.item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Batch/Contrastive_Loss', contrast_loss.item(), epoch * len(train_loader) + batch_idx)
        
        # Compute average training losses
        train_loss /= len(train_loader)
        train_recon_loss /= len(train_loader)
        train_contrast_loss /= len(train_loader)
        
        # Validation phase
        audio_encoder.eval()
        visual_encoder.eval()
        cross_modal.eval()
        decoder.eval()
        
        val_loss = 0
        val_recon_loss = 0
        val_contrast_loss = 0
        
        with torch.no_grad():
            for frames, audio, _ in val_loader:
                frames = frames.to(device)
                audio = audio.to(device)
                
                # Forward pass
                audio_features = audio_encoder(audio)
                visual_features = visual_encoder(frames)
                fused_features = cross_modal(visual_features, audio_features)
                reconstructed = decoder(fused_features)
                
                # Compute losses
                recon_loss = reconstruction_loss(reconstructed, audio)
                contrast_loss = contrastive_loss(fused_features, audio_features)
                loss = recon_loss + contrast_loss
                
                # Accumulate losses
                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_contrast_loss += contrast_loss.item()
        
        # Compute average validation losses
        val_loss /= len(val_loader)
        val_recon_loss /= len(val_loader)
        val_contrast_loss /= len(val_loader)
        
        # Log epoch metrics
        writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
        writer.add_scalar('Epoch/Train_Reconstruction_Loss', train_recon_loss, epoch)
        writer.add_scalar('Epoch/Train_Contrastive_Loss', train_contrast_loss, epoch)
        writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
        writer.add_scalar('Epoch/Val_Reconstruction_Loss', val_recon_loss, epoch)
        writer.add_scalar('Epoch/Val_Contrastive_Loss', val_contrast_loss, epoch)
        
        # Log to console
        logger.info(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss - config['training']['early_stopping_min_delta']:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'audio_encoder': audio_encoder.state_dict(),
                'visual_encoder': visual_encoder.state_dict(),
                'cross_modal': cross_modal.state_dict(),
                'decoder': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_loss
            }
            torch.save(checkpoint, os.path.join(config['training']['checkpoint_dir'], 'stage1_best.pth'))
        else:
            patience_counter += 1
            if patience_counter >= config['training']['early_stopping_patience']:
                logger.info(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    writer.close()
    return best_val_loss

if __name__ == '__main__':
    train_stage1('configs/config.yaml') 