import torch
import torch.nn as nn
import argparse
import yaml
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from models.avff_model import get_model
from utils.data_utils import create_data_loaders

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate AVFF model for deepfake detection')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                      help='Path to config file')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='Directory to save evaluation results')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def plot_confusion_matrix(cm, output_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(output_path)
    plt.close()

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for frames, audio, labels in tqdm(test_loader, desc='Evaluating'):
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
        'loss': total_loss / len(test_loader),
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds)
    }
    
    cm = confusion_matrix(all_labels, all_preds)
    
    return metrics, cm, all_preds, all_labels

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = get_model(num_classes=1)
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create data loader
    _, test_loader = create_data_loaders(
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
    
    # Evaluate
    criterion = nn.BCEWithLogitsLoss()
    metrics, cm, preds, labels = evaluate(model, test_loader, criterion, device)
    
    # Save results
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        for metric, value in metrics.items():
            f.write(f'{metric}: {value:.4f}\n')
    
    plot_confusion_matrix(cm, os.path.join(args.output_dir, 'confusion_matrix.png'))
    
    # Print results
    print('\nEvaluation Results:')
    for metric, value in metrics.items():
        print(f'{metric}: {value:.4f}')

if __name__ == '__main__':
    main() 