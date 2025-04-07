import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from avff.models import AVFFModel
from avff.datasets.dfdc import DFDCDataset
from torch.utils.tensorboard import SummaryWriter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_metrics(train_metrics, val_metrics, output_dir):
    metrics = {
        'train': train_metrics,
        'val': val_metrics
    }
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

def plot_metrics(train_metrics, val_metrics, output_dir):
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training and validation loss
    ax1.plot(train_metrics['loss'], label='Training')
    ax1.plot(val_metrics['loss'], label='Validation')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot F1 score
    ax2.plot(train_metrics['f1'], label='Training')
    ax2.plot(val_metrics['f1'], label='Validation')
    ax2.set_title('F1 Score')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.legend()
    ax2.grid(True)
    
    # Plot Precision
    ax3.plot(train_metrics['precision'], label='Training')
    ax3.plot(val_metrics['precision'], label='Validation')
    ax3.set_title('Precision')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Precision')
    ax3.legend()
    ax3.grid(True)
    
    # Plot Recall
    ax4.plot(train_metrics['recall'], label='Training')
    ax4.plot(val_metrics['recall'], label='Validation')
    ax4.set_title('Recall')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Recall')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'))
    plt.close()

def main():
    # Load config
    with open('configs/dfdc_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize model with correct architecture
    model = AVFFModel(
        num_classes=config['model']['num_classes'],
        audio_backbone=config['model']['audio_backbone'],
        visual_backbone=config['model']['visual_backbone'],
        fusion_dim=config['model']['fusion_dim'],
        dropout=config['model']['dropout']
    ).to(device)
    
    # Initialize optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
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
    
    # Initialize metrics
    train_metrics = {
        'loss': [], 'f1': [], 'precision': [], 'recall': [], 'accuracy': []
    }
    val_metrics = {
        'loss': [], 'f1': [], 'precision': [], 'recall': [], 'accuracy': []
    }
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir='logs/dfdc')
    
    # Training loop
    num_epochs = 50
    criterion = nn.CrossEntropyLoss()
    best_val_f1 = 0.0
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch['video'].to(device), batch['audio'].to(device))
            labels = batch['label'].to(device)
            
            # Compute loss
            loss = criterion(outputs['logits'], labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Store metrics
            train_loss += loss.item()
            train_preds.extend(outputs['logits'].argmax(dim=1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        # Calculate training metrics
        train_metrics['loss'].append(train_loss / len(train_loader))
        train_metrics['f1'].append(f1_score(train_labels, train_preds))
        train_metrics['precision'].append(precision_score(train_labels, train_preds))
        train_metrics['recall'].append(recall_score(train_labels, train_preds))
        train_metrics['accuracy'].append(accuracy_score(train_labels, train_preds))
        
        # Log training metrics to TensorBoard
        writer.add_scalar('Loss/train', train_metrics['loss'][-1], epoch)
        writer.add_scalar('F1/train', train_metrics['f1'][-1], epoch)
        writer.add_scalar('Precision/train', train_metrics['precision'][-1], epoch)
        writer.add_scalar('Recall/train', train_metrics['recall'][-1], epoch)
        writer.add_scalar('Accuracy/train', train_metrics['accuracy'][-1], epoch)
        
        # Log training metrics
        logger.info(f"Training metrics - Loss: {train_metrics['loss'][-1]:.4f}, "
                   f"F1: {train_metrics['f1'][-1]:.4f}, "
                   f"Precision: {train_metrics['precision'][-1]:.4f}, "
                   f"Recall: {train_metrics['recall'][-1]:.4f}, "
                   f"Accuracy: {train_metrics['accuracy'][-1]:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                outputs = model(batch['video'].to(device), batch['audio'].to(device))
                labels = batch['label'].to(device)
                
                loss = criterion(outputs['logits'], labels)
                val_loss += loss.item()
                
                val_preds.extend(outputs['logits'].argmax(dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        val_metrics['loss'].append(val_loss / len(val_loader))
        val_metrics['f1'].append(f1_score(val_labels, val_preds))
        val_metrics['precision'].append(precision_score(val_labels, val_preds))
        val_metrics['recall'].append(recall_score(val_labels, val_preds))
        val_metrics['accuracy'].append(accuracy_score(val_labels, val_preds))
        
        # Log validation metrics to TensorBoard
        writer.add_scalar('Loss/val', val_metrics['loss'][-1], epoch)
        writer.add_scalar('F1/val', val_metrics['f1'][-1], epoch)
        writer.add_scalar('Precision/val', val_metrics['precision'][-1], epoch)
        writer.add_scalar('Recall/val', val_metrics['recall'][-1], epoch)
        writer.add_scalar('Accuracy/val', val_metrics['accuracy'][-1], epoch)
        
        # Log validation metrics
        logger.info(f"Validation metrics - Loss: {val_metrics['loss'][-1]:.4f}, "
                   f"F1: {val_metrics['f1'][-1]:.4f}, "
                   f"Precision: {val_metrics['precision'][-1]:.4f}, "
                   f"Recall: {val_metrics['recall'][-1]:.4f}, "
                   f"Accuracy: {val_metrics['accuracy'][-1]:.4f}")
        
        # Update learning rate
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('Learning Rate', current_lr, epoch)
        scheduler.step()
        
        # Save metrics and plots
        save_metrics(train_metrics, val_metrics, 'outputs/dfdc')
        plot_metrics(train_metrics, val_metrics, 'outputs/dfdc')
        
        # Save checkpoint if it's the best model
        if val_metrics['f1'][-1] > best_val_f1:
            best_val_f1 = val_metrics['f1'][-1]
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'config': config
            }, 'checkpoints/dfdc/best_model.pth')
            logger.info("Saved new best model checkpoint")
        
        # Save latest model
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'config': config
        }, 'checkpoints/dfdc/latest_model.pth')
    
    writer.close()

if __name__ == '__main__':
    main() 