import os
import torch
import logging
import yaml
from train_dfdc import train_model, setup_device
from avff.datasets.dfdc import DFDCDataset
from avff.models import AVFFModel
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    
    # Load checkpoint
    checkpoint_path = os.path.join(config['training']['output_dir'], 'best_model.pth')
    checkpoint = torch.load(checkpoint_path)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Continue training
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=20,  # Train for 10 more epochs
        device=device,
        output_dir=config['training']['output_dir'],
        config=config,
        start_epoch=10  # Start from epoch 10
    )

if __name__ == '__main__':
    main() 