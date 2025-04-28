import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from avff.models import AVFFModel
from avff.datasets.dfdc import DFDCDataset
from avff.utils import visualize_results
from sklearn.metrics import log_loss, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def evaluate_model(model, test_loader, device, output_dir):
    """
    Evaluate the AVFF model on DFDC test set
    
    Args:
        model: AVFFModel instance
        test_loader: DataLoader for test data
        device: Device to evaluate on
        output_dir: Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    all_filenames = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            video_frames = batch['video_frames'].to(device)
            audio_features = batch['audio_features'].to(device)
            labels = batch['label'].to(device)
            filenames = batch['filename']
            
            # Forward pass
            outputs = model(audio_features, video_frames)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_filenames.extend(filenames)
            
            # Visualize first few samples
            if len(all_preds) <= 5:
                visualize_results(
                    video_path=os.path.join('test_videos', filenames[0]),
                    audio_features=audio_features[0],
                    visual_features=video_frames[0],
                    prediction=outputs[0],
                    save_path=os.path.join(output_dir, f'sample_{len(all_preds)}.png')
                )
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    test_log_loss = log_loss(all_labels, all_preds)
    test_auc = roc_auc_score(all_labels, all_preds)
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds > 0.5)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Generate ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()
    
    # Save predictions
    results = {
        'filename': all_filenames,
        'prediction': all_preds,
        'label': all_labels
    }
    np.save(os.path.join(output_dir, 'predictions.npy'), results)
    
    # Print metrics
    print(f'Test Log Loss: {test_log_loss:.4f}')
    print(f'Test AUC: {test_auc:.4f}')
    
    return test_log_loss, test_auc

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test dataset
    test_dataset = DFDCDataset(
        root_dir='test_videos',
        metadata_path='metadata.json',
        split='test'
    )
    
    # Create test loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4
    )
    
    # Load model
    model = AVFFModel(num_classes=2).to(device)
    checkpoint = torch.load('checkpoints/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate model
    evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir='results'
    )

if __name__ == '__main__':
    main() 