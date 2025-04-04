import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import yaml

from avff.datasets.dfdc import DFDCDataset
from avff.models import AVFFModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def train_model(model, train_loader, val_loader, num_epochs, device, output_dir, config):
    os.makedirs(output_dir, exist_ok=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    # Enable automatic mixed precision for faster training
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # Enable CUDA graphs if supported
    use_cuda_graph = config['training'].get('cuda_graph', False) and device.type == 'cuda'
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        logging.info(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        train_progress = tqdm(train_loader, desc='Training')
        for batch_idx, batch in enumerate(train_progress):
            video_data = batch['video'].to(device)  # [batch_size, num_frames, channels, height, width]
            audio_data = batch['audio'].to(device)  # [batch_size, sequence_length]
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            # Use automatic mixed precision
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(video_data, audio_data)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(video_data, audio_data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            train_progress.set_postfix({'batch_loss': f"{loss.item():.4f}"})
            
            if batch_idx % 10 == 0:
                logging.info(f"Batch {batch_idx}: Loss = {loss.item():.4f}")
                
            # Clear cache periodically
            if device.type == 'cuda' and batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        logging.info(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc='Validation')
            for batch in val_progress:
                video_data = batch['video'].to(device)
                audio_data = batch['audio'].to(device)
                labels = batch['label'].to(device)
                
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(video_data, audio_data)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(video_data, audio_data)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_batches += 1
                
                probs = torch.softmax(outputs, dim=1)
                all_preds.extend(probs[:, 1].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                val_progress.set_postfix({'batch_loss': f"{loss.item():.4f}"})
        
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        # Calculate log loss
        val_log_loss = log_loss(all_labels, all_preds)
        logging.info(f"Validation - Avg Loss: {avg_val_loss:.4f}, Log Loss: {val_log_loss:.4f}")
        
        # Save best model
        if val_log_loss < best_val_loss:
            best_val_loss = val_log_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            logging.info(f"Saved new best model with validation log loss: {val_log_loss:.4f}")
        
        # Plot learning curves
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Learning Curves')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'learning_curves.png'))
        plt.close()

def main():
    # Load configuration
    with open('configs/dfdc_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device with proper GPU setup
    device = setup_device()
    
    # Create datasets
    train_dataset = DFDCDataset(
        root_dir=config['dataset']['train_root'],
        metadata_path=config['dataset']['train_metadata'],
        split='train'
    )
    val_dataset = DFDCDataset(
        root_dir=config['dataset']['train_root'],
        metadata_path=config['dataset']['train_metadata'],
        split='val'
    )
    
    logging.info(f"Number of training samples: {len(train_dataset)}")
    logging.info(f"Number of validation samples: {len(val_dataset)}")
    
    # Create data loaders with pin_memory for faster GPU transfer
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Initialize model
    model = AVFFModel(
        num_classes=config['model']['num_classes'],
        audio_backbone=config['model']['audio_backbone'],
        visual_backbone=config['model']['visual_backbone']
    ).to(device)
    logging.info("Model initialized")
    
    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs'],
        device=device,
        output_dir=config['training']['output_dir'],
        config=config
    )

if __name__ == "__main__":
    main() 