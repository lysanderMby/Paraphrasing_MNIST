'''
A basic archiecture.

Training a classifier architecture designed to produce intepretable intermediate states.
These intermediate states are all, until the final logits, of the same shape of the original MNIST inputs.
This harms classification performance (concerningly pointing to a negative alignment tax).

Classification can be performed while the intermediate states are randomly selected to have the paraphraser model applied to them.
Randomly applying this model reduces the potential for steganography in the internal thoughts of the model, although it seems unlikely that this would be an issue at this scale and level of simplicity.

This script shows the impact of randomly applying this paraphrasing. Other scripts show the impact of always or never applying it.
'''

# Package imports

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.models import resnet18
from tqdm import tqdm
import os
import sys
from pathlib import Path
from typing import List, Tuple
import json
import re
import time
from datetime import datetime, timedelta

# Add parent directory to Python path to enable imports from sibling directories
project_root = Path(__file__).resolve().parent.parent  # Going to the true parent
sys.path.append(str(project_root))

# Import from other modules
from initial_classifier.training_initial_classifier import train_classifier, PretrainedClassifier
from paraphraser.training_paraphraser import train_paraphraser, ImageParaphraser

class InterpretableClassifier(nn.Module):
    """Classifier with customizable interpretable intermediate representations"""
    def __init__(self, layer_configs: List[Tuple[int, int]], num_classes: int = 10):
        """
        Args:
            layer_configs: List of tuples (in_channels, out_channels) for each layer
            num_classes: Number of output classes
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.residual_projections = nn.ModuleList()
        
        for in_channels, out_channels in layer_configs:
            # Main convolutional block
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*2, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels*2),
                nn.ReLU(),
                nn.Conv2d(out_channels*2, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
            self.layers.append(layer)
            
            # Residual projection if needed
            if in_channels != out_channels:
                proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            else:
                proj = nn.Identity()
            self.residual_projections.append(proj)
        
        # Calculate total flattened size for final layer
        final_channels = layer_configs[-1][1]
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),  # Adaptive pooling to fixed size
            nn.Flatten(),
            nn.Linear(final_channels * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def get_param_count(self):
        """Return total number of parameters"""
        return sum(p.numel() for p in self.parameters())
        
    def forward(self, x, return_intermediates=False):
        intermediates = []
        current = x
        
        for layer, proj in zip(self.layers, self.residual_projections):
            # Apply main layer
            layer_out = layer(current)
            # Add residual connection
            residual = proj(current)
            current = layer_out + residual
            intermediates.append(current)
        
        logits = self.classifier(current)
        
        if return_intermediates:
            return logits, intermediates
        return logits

def paraphrase_multichannel(paraphraser, state):
    """
    Paraphrase a multi-channel state by processing each channel independently
    
    Args:
        paraphraser: Single-channel paraphraser model
        state: Multi-channel state tensor [B, C, H, W] (B=batch size, C=channels, H=height, W=width)
        
    Returns:
        Paraphrased multi-channel state tensor [B, C, H, W]
    """
    batch_size, num_channels, height, width = state.shape
    
    # Process each channel independently
    paraphrased_channels = []
    for c in range(num_channels):
        # Extract single channel [B, 1, H, W]
        channel = state[:, c:c+1, :, :]
        
        # Paraphrase this channel
        paraphrased_channel = paraphraser(channel)
        
        paraphrased_channels.append(paraphrased_channel)
    
    # Concatenate channels back together
    return torch.cat(paraphrased_channels, dim=1)

    
def train_interpretable_classifier(
    model, paraphraser, train_loader, test_loader,
    paraphrase_prob=1.0, num_epochs=5, device='cuda',
    exp_dir=None, reg_weight=0.01
):
    """
    Train the interpretable classifier with support for probability schedules.
    
    Args:
        model: The model to train
        paraphraser: The paraphraser model
        train_loader: Training data loader
        test_loader: Test data loader
        paraphrase_prob: float or list. If float, used as constant probability.
                        If list, must have length num_epochs
        num_epochs: Number of training epochs
        device: Device to train on
        exp_dir: Directory to save results
        reg_weight: Weight for regularization loss
        
    Returns:
        dict: Training metrics
    """
    def _validate_prob_schedule(prob_schedule, num_epochs):
        """Validate and normalize probability schedule"""
        if isinstance(prob_schedule, (int, float)):
            return [float(prob_schedule)] * num_epochs
        
        if isinstance(prob_schedule, list):
            if len(prob_schedule) == num_epochs:
                return [float(p) for p in prob_schedule]
            elif len(prob_schedule) == 1:
                return [float(prob_schedule[0])] * num_epochs
            else:
                raise ValueError(
                    f"Probability schedule length ({len(prob_schedule)}) "
                    f"must match num_epochs ({num_epochs}) or be length 1"
                )
        
        raise ValueError(
            f"Invalid probability schedule type: {type(prob_schedule)}"
        )

    # Validate and normalize probability schedule
    if isinstance(paraphrase_prob, list) and len(paraphrase_prob) > 0 and isinstance(paraphrase_prob[0], list):
        # Multiple probability schedules provided
        prob_schedules = [_validate_prob_schedule(schedule, num_epochs) 
                         for schedule in paraphrase_prob]
        num_runs = len(prob_schedules) # should always be 1
    else:
        # Single probability or schedule provided
        prob_schedules = [_validate_prob_schedule(paraphrase_prob, num_epochs)]
        num_runs = 1

    all_metrics = []
    
    for run_idx, prob_schedule in enumerate(prob_schedules): # should only even be one passed
        if len(prob_schedule) < 100:
            print(f"Probability schedule: {prob_schedule}")
        
        model = model.to(device)
        paraphraser = paraphraser.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2
        )
        
        metrics_log = {
            'paraphrase_schedule': prob_schedule,
            'train_acc': [],
            'train_loss': [],
            'test_acc': [],
            'layer_metrics': {}
        }
        
        best_acc = 0.0
        
        start_time = time.time()
        print(f"\nStarting training at {datetime.now().strftime('%H:%M:%S')}")
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            model.train()
            
            # Get current paraphrase probability
            current_prob = prob_schedule[epoch] if isinstance(prob_schedule, list) else prob_schedule
            print(f"\nEpoch {epoch + 1}/{num_epochs}, "
                  f"Paraphrase probability: {current_prob:.3f}")
            print("")  # Clean line for progress bar
            
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            # Training loop with progress bar
            for batch_idx, (images, labels) in tqdm(enumerate(train_loader), 
                                                  desc=f"Training", 
                                                  total=len(train_loader),
                                                  position=0,
                                                  leave=True):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                
                logits, intermediates = model(images, return_intermediates=True)
                
                # Apply current epoch's paraphrase probability
                paraphrased = []
                for state in intermediates:
                    if torch.rand(1).item() < current_prob:
                        with torch.no_grad():
                            para_state = paraphrase_multichannel(paraphraser, state)
                        paraphrased.append(para_state)
                    else:
                        paraphrased.append(state)
                
                class_loss = criterion(logits, labels)
                reg_loss = sum(F.mse_loss(inter, para) 
                             for inter, para in zip(intermediates, paraphrased))
                
                total_loss = class_loss + reg_weight * reg_loss
                total_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += total_loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_acc = 100 * correct / total
            train_loss = epoch_loss / len(train_loader)
            
            # Test without progress bar
            model.eval()
            test_loss = 0
            correct = 0
            print("\nEvaluating...", end=' ')
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(images)
                    test_loss += criterion(outputs, labels).item()
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()
            
            test_acc = 100 * correct / len(test_loader.dataset)
            
            scheduler.step(test_acc)
            
            metrics_log['train_acc'].append(train_acc)
            metrics_log['train_loss'].append(train_loss)
            metrics_log['test_acc'].append(test_acc)
            
            if exp_dir:
                run_dir = exp_dir / f"run_{run_idx}"
                run_dir.mkdir(exist_ok=True)
                
                # Save training progress plots
                save_training_visualizations(metrics_log, run_dir, epoch)
                
                # Save intermediate states visualization
                intermediates_dir = run_dir / 'intermediate_states' / f'epoch_{epoch+1}'
                images, _ = next(iter(train_loader))
                images = images.to(device)
                
                with torch.no_grad():
                    logits, intermediates = model(images, return_intermediates=True)
                    # Generate paraphrased versions and track which were paraphrased
                    paraphrased_states = []
                    layer_diffs = [{'was_paraphrased': False}]  # Initial state for input
                    
                    for state in intermediates:
                        was_paraphrased = torch.rand(1).item() < current_prob
                        if was_paraphrased:
                            para_state = paraphrase_multichannel(paraphraser, state)
                            paraphrased_states.append(para_state)
                        else:
                            paraphrased_states.append(state)
                        layer_diffs.append({'was_paraphrased': was_paraphrased})
                    
                    # Create visualization subdirectories
                    vis_base_dir = run_dir / 'visualizations'
                    intermediates_dir = vis_base_dir / 'intermediate_states' / f'epoch_{epoch+1}'
                    paraphraser_dir = vis_base_dir / 'paraphraser_samples'
                    
                    visualize_intermediates(
                        images, images, intermediates, paraphrased_states, 
                        layer_diffs, epoch, batch_idx, num_samples=3,
                        save_path=intermediates_dir
                    )
                
                # Save paraphraser visualization once per epoch
                if epoch == 0:  # Only save paraphraser visualization once
                    visualize_paraphrasing(
                        paraphraser, train_loader, device, 
                        num_samples=10, save_path=paraphraser_dir
                    )
            
            if test_acc > best_acc and exp_dir:
                best_acc = test_acc
                if num_runs > 1:
                    save_path = exp_dir / f'run_{run_idx}'
                    save_path.mkdir(exist_ok=True)
                    model_path = save_path / 'best_model.pth'
                    metrics_path = save_path / 'best_metrics.pth'
                else:
                    model_path = exp_dir / 'best_model.pth'
                    metrics_path = exp_dir / 'best_metrics.pth'
                    
                torch.save(model.state_dict(), model_path)
                torch.save(metrics_log, metrics_path)
            
            print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, '
                  f'Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%')
            
            # Calculate and display timing information
            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            
            print(f"\nEpoch {epoch + 1} completed in {timedelta(seconds=int(epoch_time))}")
            print(f"Total training time so far: {timedelta(seconds=int(total_time))}")
            print(f"Average time per epoch: {timedelta(seconds=int(total_time/(epoch+1)))}")
            print(f"Test Accuracy: {test_acc:.2f}%")
            print("-" * 50)
        
        # Final timing information
        total_training_time = time.time() - start_time
        print(f"\nTraining completed at {datetime.now().strftime('%H:%M:%S')}")
        print(f"Total training time: {timedelta(seconds=int(total_training_time))}")
        print(f"Average time per epoch: {timedelta(seconds=int(total_training_time/num_epochs))}")
        
        all_metrics.append(metrics_log)
    
    return all_metrics if num_runs > 1 else all_metrics[0]


# visualise the paraphraser images with their original counterparts
def visualize_paraphrasing(paraphraser, train_loader, device, num_samples=10, save_path: Path = None):
    """
    Helper function to visualize paraphrased images.
    Optionally saves the visualization instead of displaying it.
    """
    # Get a batch of images
    images, _ = next(iter(train_loader))
    images = images[:num_samples].to(device)
    
    # Generate paraphrased images
    with torch.no_grad():
        paraphrased = paraphraser(images)
    
    # Move tensors to CPU for visualization
    original_images = images.cpu()
    paraphrased_images = paraphrased.cpu()
    
    # Plot original vs paraphrased
    plt.figure(figsize=(20, 4))
    for i in range(num_samples):
        # Original
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(original_images[i].squeeze(), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title('Original')
        
        # Paraphrased
        plt.subplot(2, num_samples, num_samples + i + 1)
        plt.imshow(paraphrased_images[i].squeeze(), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title('Paraphrased')
    
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / f'paraphraser_samples.png', bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

def visualize_intermediates(original_input, paraphrased_input, layer_originals, 
                          layer_paraphrased, layer_diffs, epoch, batch_idx, 
                          num_samples=1, save_path: Path = None):
    """
    Visualize random sample of images, their intermediate representations, and paraphrased versions.
    Optionally saves the visualization instead of displaying it.
    """
    batch_size = original_input.size(0)
    indices = torch.randperm(batch_size)[:num_samples]
    
    for idx in indices:
        num_cols = len(layer_originals) + 1
        plt.figure(figsize=(3 * num_cols, 6))
        
        # Original input
        plt.subplot(2, num_cols, 1)
        plt.imshow(original_input[idx].cpu().squeeze(), cmap='gray')
        plt.title('Original Input')
        plt.axis('off')
        
        # Input paraphrasing status and visualization
        was_paraphrased = layer_diffs[0]['was_paraphrased']
        status = "Paraphrased" if was_paraphrased else "Original"
        plt.subplot(2, num_cols, num_cols + 1)
        
        # Show either paraphrased input or original depending on whether paraphrasing occurred
        display_input = paraphrased_input if was_paraphrased else original_input
        plt.imshow(display_input[idx].detach().cpu().squeeze(), cmap='gray')
        plt.title(f'Input ({status})')
        plt.axis('off')
        
        # Layer outputs and their paraphrased versions
        for i, (orig, para) in enumerate(zip(layer_originals, layer_paraphrased)):
            # Original layer output
            plt.subplot(2, num_cols, i + 2)
            # Take mean across channels if multi-channel
            if orig[idx].shape[0] > 1:
                img_to_show = orig[idx].detach().cpu().mean(dim=0)  # Average across channels
            else: # if not multi-channel. 
                img_to_show = orig[idx].detach().cpu().squeeze()
            plt.imshow(img_to_show, cmap='gray')
            plt.title(f'Layer {i+1}')
            plt.axis('off')
            
            # Paraphrasing status for this layer
            was_paraphrased = layer_diffs[i+1]['was_paraphrased']
            status = "Paraphrased" if was_paraphrased else "Original"
            
            # Paraphrased or original version
            plt.subplot(2, num_cols, num_cols + i + 2)
            # Take mean across channels if multi-channel
            if para[idx].shape[0] > 1:
                img_to_show = para[idx].detach().cpu().mean(dim=0)  # Average across channels
            else:
                img_to_show = para[idx].detach().cpu().squeeze()
            plt.imshow(img_to_show, cmap='gray')
            plt.title(f'L{i+1} ({status})')
            plt.axis('off')
        
        plt.suptitle(f'Epoch {epoch+1}, Step {batch_idx}, Sample {idx.item()}')
        plt.tight_layout()
        
        if save_path:
            # Create directory if it doesn't exist
            save_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path / f'intermediates_epoch{epoch+1}_step{batch_idx}_sample{idx.item()}.png',
                       bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def plot_training_progress(metrics_log):
    """Plot comprehensive training metrics"""
    num_layers = len(metrics_log['layer_metrics'])
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 10))
    
    # Plot classification accuracy
    plt.subplot(2, 2, 1)
    plt.plot(metrics_log['train_acc'], label='Train')
    plt.plot(metrics_log['test_acc'], label='Test')
    plt.title('Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # Plot MSE for each layer
    plt.subplot(2, 2, 2)
    for i in range(num_layers):
        plt.plot(metrics_log['layer_metrics'][i]['mse'], 
                label=f'Layer {i+1}')
    plt.title('Layer-wise MSE')
    plt.xlabel('Update Step')
    plt.ylabel('MSE')
    plt.legend()
    
    # Plot Cosine Similarity for each layer
    plt.subplot(2, 2, 3)
    for i in range(num_layers):
        plt.plot(metrics_log['layer_metrics'][i]['cosine_sim'], 
                label=f'Layer {i+1}')
    plt.title('Layer-wise Cosine Similarity')
    plt.xlabel('Update Step')
    plt.ylabel('Similarity')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def compare_training_results(metrics_logs):
    """Compare and visualize results from different paraphrasing probabilities"""
    plt.figure(figsize=(15, 10))
    
    # Plot test accuracy comparison
    plt.subplot(2, 2, 1)
    for metrics in metrics_logs:
        prob = metrics['paraphrase_prob']
        plt.plot(metrics['test_acc'], label=f'Prob={prob:.1f}')
    plt.title('Test Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # Plot average MSE per layer
    plt.subplot(2, 2, 2)
    for metrics in metrics_logs:
        prob = metrics['paraphrase_prob']
        layer_mses = []
        for layer in range(len(metrics['layer_metrics'])):
            avg_mse = np.mean(metrics['layer_metrics'][layer]['mse'])
            layer_mses.append(avg_mse)
        plt.plot(layer_mses, marker='o', label=f'Prob={prob:.1f}')
    plt.title('Average MSE by Layer')
    plt.xlabel('Layer')
    plt.ylabel('MSE')
    plt.legend()
    
    # Plot average cosine similarity per layer
    plt.subplot(2, 2, 3)
    for metrics in metrics_logs:
        prob = metrics['paraphrase_prob']
        layer_cos = []
        for layer in range(len(metrics['layer_metrics'])):
            avg_cos = np.mean(metrics['layer_metrics'][layer]['cosine_sim'])
            layer_cos.append(avg_cos)
        plt.plot(layer_cos, marker='o', label=f'Prob={prob:.1f}')
    plt.title('Average Cosine Similarity by Layer')
    plt.xlabel('Layer')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def save_training_visualizations(metrics_log, exp_dir: Path, epoch: int):
    """Save training visualizations to the experiment directory"""
    # Clear any existing plots
    plt.clf()
    plt.close('all')
    
    # Create new figure with specific size and DPI
    fig = plt.figure(figsize=(20, 10), dpi=100)
    
    # Plot accuracies
    plt.subplot(2, 2, 1)
    plt.plot(metrics_log['train_acc'], label='Train', linewidth=2)
    plt.plot(metrics_log['test_acc'], label='Test', linewidth=2)
    plt.title('Classification Accuracy', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Plot losses
    plt.subplot(2, 2, 2)
    plt.plot(metrics_log['train_loss'], label='Train Loss', linewidth=2)
    plt.title('Training Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Ensure proper spacing between subplots
    plt.tight_layout(pad=3.0)
    
    # Save plot with high quality
    save_path = exp_dir / f"training_progress_epoch_{epoch+1}.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=100, format='png')
    
    # Clean up
    plt.close(fig)
    plt.close('all')

def create_versioned_directory(base_path: Path, folder_name: str) -> Path:
    """
    Creates a versioned directory. If the directory already exists,
    adds a version number (v1, v2, etc.) to create a unique path.
    """
    # First try without version number
    target_path = base_path / folder_name
    if not target_path.exists():
        target_path.mkdir(parents=True, exist_ok=False)
        return target_path
        
    # Find existing versions
    version_pattern = re.compile(rf"^{re.escape(folder_name)}_v(\d+)$")
    existing_versions = []
    
    for path in base_path.iterdir():
        if path.is_dir():
            match = version_pattern.match(path.name)
            if match:
                existing_versions.append(int(match.group(1)))
    
    # If no versions exist yet, start with v1
    if not existing_versions:
        new_version = 1
    else:
        new_version = max(existing_versions) + 1
        
    # Create new versioned directory
    versioned_path = base_path / f"{folder_name}_v{new_version}"
    versioned_path.mkdir(parents=True, exist_ok=False)
    
    return versioned_path


def main():
    # Setup device and paths

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Setup paths
    data_dir = project_root / 'data'
    models_dir = project_root / 'models'
    #models_dir.mkdir(exist_ok=True) # if this directory is not present, an error should be created clearly
    
    # Define model architecture configurations
    layer_configs = [
        # (in_channels, out_channels)
        (1, 32),    # First layer expands channels
        (32, 64),   # Further increase feature complexity
        (64, 32),   # Gradually reduce channels
        (32, 16)    # Final interpretable representation
    ]

    # Setup data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(root=str(data_dir), train=True,
                                             download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root=str(data_dir), train=False,
                                            download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, 
                            pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, 
                            pin_memory=True, num_workers=4)
        
    # Load prerequisite models
    classifier = PretrainedClassifier()
    paraphraser = ImageParaphraser()
    
    # Load initial classifier
    classifier_path = models_dir / 'initial_classifier.pth'
    if not classifier_path.exists():
        raise FileNotFoundError(
            "Initial classifier not found! Please run initial_classifier/train_classifier.py first."
        )
    print("Loading pretrained classifier...")
    classifier.load_state_dict(torch.load(classifier_path, map_location=device, weights_only=True))
    classifier.to(device)
    classifier.eval()
    
    # Load paraphraser
    paraphraser_path = models_dir / 'paraphraser.pth'
    if not paraphraser_path.exists():
        raise FileNotFoundError(
            "Paraphraser not found! Please run paraphrasing/training_paraphraser.py first."
        )
    print("Loading pretrained paraphraser...")
    paraphraser.load_state_dict(torch.load(paraphraser_path, map_location=device, weights_only=True))
    paraphraser.to(device)
    paraphraser.eval()
    
    # Define training schedules
    num_epochs = 10  # Define number of epochs. Currently constant across experiments
    
    # Define different probability schedules to test
    paraphrase_schedules = [
        # Constant probabilities
        #0.0,    # No paraphrasing
        #0.5,    # 50% paraphrasing
        #1.0,    # 100% paraphrasing
        
        # Linear increase schedules
        [(i+1) * 0.2 / (num_epochs - 1) for i in range(num_epochs)],  # 0.0 to 0.2
        [i * 0.5 / (num_epochs - 1) for i in range(num_epochs)],  # 0.0 to 0.5
        
        # Custom schedules
        [0.0] * 3 + [0.2] * 7,  # No paraphrasing for 3 epochs, then 20%
        #[0.1] * 5 + [0.3] * 5,  # 10% for 5 epochs, then 30%
    ]
    
    print("\nTraining interpretable classifiers with different paraphrasing schedules...")
    
    # Create experiment root directory with layer config name
    exp_name = f"layers_{len(layer_configs)}_channels_" + "_".join(str(cfg[1]) for cfg in layer_configs)
    exp_root = create_versioned_directory(models_dir, exp_name)
    
    # Save layer configuration
    with open(exp_root / "architecture_config.txt", "w") as f:
        f.write("Layer configurations:\n")
        for i, (in_ch, out_ch) in enumerate(layer_configs):
            f.write(f"Layer {i}: {in_ch} -> {out_ch} channels\n")
    
    for schedule_idx, schedule in enumerate(paraphrase_schedules):
        print(f"Running schedule {schedule_idx+1} / {len(paraphrase_schedules)}") # allows tracking of progress
        # Create schedule-specific directory
        if isinstance(schedule, (int, float)):
            schedule_name = f"constant_{schedule:.2f}"
            schedule_desc = f"{schedule*100:.0f}% constant paraphrasing"
        else:
            schedule_name = f"schedule_{schedule_idx}"
            schedule_desc = f"Custom schedule {schedule_idx}: {schedule}"
        
        schedule_dir = exp_root / schedule_name
        schedule_dir.mkdir(exist_ok=True)
        
        print(f"\nTraining classifier with {schedule_desc}")
        model = InterpretableClassifier(layer_configs)
        print(f"\nTotal interpretable classifier parameter number is {model.get_param_count()}")
        
        # Save schedule configuration
        with open(schedule_dir / "schedule_config.txt", "w") as f:
            f.write(f"Schedule description: {schedule_desc}\n")
            f.write(f"Schedule values: {schedule}\n")
            f.write(f"Number of epochs: {num_epochs}\n")
        
        metrics = train_interpretable_classifier(
            model=model,
            paraphraser=paraphraser,
            train_loader=train_loader,
            test_loader=test_loader,
            paraphrase_prob=schedule,
            device=device,
            num_epochs=num_epochs,
            exp_dir=schedule_dir
        )
        
        # Save final model and metrics
        torch.save(model.state_dict(), schedule_dir / 'final_model.pth')
        torch.save(metrics, schedule_dir / 'training_metrics.pth')
    
    print("\nTraining complete! Models and metrics saved in:", exp_root)

if __name__ == "__main__":
    main()