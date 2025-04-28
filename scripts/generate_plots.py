import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Set style
plt.style.use('default')
sns.set_theme()

def load_tensorboard_data(log_dir):
    """Load data from TensorBoard logs."""
    ea = EventAccumulator(log_dir)
    ea.Reload()
    
    print("\nDetailed TensorBoard Data:")
    print("=" * 50)
    
    # Get all steps from training loss
    events = ea.Scalars('Epoch/train_loss')
    steps = [event.step for event in events]
    
    # Initialize data dictionary with steps as index
    data = {
        'train_loss': [],
        'val_loss': [],
        'train_f1': [],
        'val_f1': [],
        'train_precision': [],
        'val_precision': [],
        'train_recall': [],
        'val_recall': []
    }
    
    # Create a mapping of step to value for each metric
    metrics = {
        'train_loss': 'Epoch/train_loss',
        'val_loss': 'Epoch/val_loss',
        'train_f1': 'Epoch/train_f1',
        'val_f1': 'Epoch/val_f1',
        'train_precision': 'Epoch/train_precision',
        'val_precision': 'Epoch/val_precision',
        'train_recall': 'Epoch/train_recall',
        'val_recall': 'Epoch/val_recall'
    }
    
    # Load and print data for each metric
    print("\nEpoch-by-Epoch Values:")
    print("-" * 50)
    for metric, tag in metrics.items():
        events = ea.Scalars(tag)
        step_to_value = {event.step: event.value for event in events}
        values = [step_to_value.get(step, np.nan) for step in steps]
        data[metric] = values
        
        print(f"\n{metric}:")
        for step, value in zip(steps, values):
            print(f"  Epoch {step}: {value:.3f}")
    
    # Create DataFrame with steps as index
    df = pd.DataFrame(data, index=steps)
    df.index.name = 'step'
    
    return df

def load_learning_rate_data(log_dir):
    """Load learning rate data separately."""
    ea = EventAccumulator(log_dir)
    ea.Reload()
    
    events = ea.Scalars('Batch/learning_rate')
    steps = [event.step for event in events]
    values = [event.value for event in events]
    
    print("\nLearning Rate Values:")
    print("-" * 50)
    for step, value in zip(steps[:10], values[:10]):  # Print first 10 values
        print(f"  Step {step}: {value:.6f}")
    print("  ...")
    for step, value in zip(steps[-10:], values[-10:]):  # Print last 10 values
        print(f"  Step {step}: {value:.6f}")
    
    return steps, values

def plot_loss_curves(df, save_path):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(df) + 1)
    plt.plot(epochs, df['train_loss'], label='Training Loss', marker='o')
    plt.plot(epochs, df['val_loss'], label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.ylim(0.2, 1.0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_f1_scores(df, save_path):
    """Plot F1 score evolution."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(df) + 1)
    plt.plot(epochs, df['train_f1'], label='Training F1', marker='o')
    plt.plot(epochs, df['val_f1'], label='Validation F1', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Evolution')
    plt.legend()
    plt.grid(True)
    plt.ylim(0.8, 1.0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_precision_recall(df, save_path):
    """Plot precision-recall curves."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(df) + 1)
    plt.plot(epochs, df['train_precision'], label='Training Precision', marker='o')
    plt.plot(epochs, df['train_recall'], label='Training Recall', marker='o')
    plt.plot(epochs, df['val_precision'], label='Validation Precision', marker='o')
    plt.plot(epochs, df['val_recall'], label='Validation Recall', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Precision and Recall Evolution')
    plt.legend()
    plt.grid(True)
    plt.ylim(0.8, 1.0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_learning_rate(steps, values, save_path):
    """Plot learning rate schedule."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(steps) + 1), values, marker='.')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    plt.yscale('log')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Create assets directory if it doesn't exist
    os.makedirs('docs/assets', exist_ok=True)
    
    # Load TensorBoard data
    df = load_tensorboard_data('outputs/dfdc/tensorboard')
    lr_steps, lr_values = load_learning_rate_data('outputs/dfdc/tensorboard')
    
    # Generate plots
    plot_loss_curves(df, 'docs/assets/training_loss.png')
    plot_f1_scores(df, 'docs/assets/f1_score.png')
    plot_precision_recall(df, 'docs/assets/precision_recall.png')
    plot_learning_rate(lr_steps, lr_values, 'docs/assets/learning_rate.png')
    
    print("\nPlots generated successfully!")

if __name__ == '__main__':
    main() 