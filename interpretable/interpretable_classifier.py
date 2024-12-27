'''
Training a classifier architecture designed to produce intepretable intermediate states/
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

# Add parent directory to Python path to enable imports from sibling directories
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# Import from other modules
from initial_classifier.training_initial_classifier import train_classifier, PretrainedClassifier
from paraphraser.training_paraphraser import train_paraphraser, ImageParaphraser

class InterpretableClassifier(nn.Module):
    """Classifier that outputs interpretable intermediate representations"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.layers = nn.ModuleList([
            # Layer 1: Edge detection and basic feature extraction
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 1, kernel_size=3, padding=1),
                nn.Sigmoid()
            ),
            # Layer 2: Pattern recognition
            nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 1, kernel_size=3, padding=1),
                nn.Sigmoid()
            ),
            # Layer 3: Higher-level feature extraction
            nn.Sequential(
                nn.Conv2d(1, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 1, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
        ])
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, return_intermediates=False):
        intermediates = []
        current = x
        
        for layer in self.layers:
            current = layer(current)
            intermediates.append(current)
        
        logits = self.classifier(current)
        
        if return_intermediates:
            return logits, intermediates
        return logits
    
def train_interpretable_classifier(model, paraphraser, train_loader, test_loader, 
                                 paraphrase_prob=1.0, num_epochs=5, device='cuda', save_dir=None):
    """
    Train the interpretable classifier. Probabilistically paraphrases input and intermediate model states.
    """
    model = model.to(device)
    paraphraser = paraphraser.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    metrics_log = {
        'paraphrase_prob': paraphrase_prob,
        'train_acc': [],
        'train_loss': [],
        'test_acc': [],
        'batch_metrics': {
            'loss': [],
            'acc': []
        },
        'layer_metrics': {i: {'mse': [], 'cosine_sim': [], 'was_paraphrased': []} 
                         for i in range(len(model.layers) + 1)}  # +1 for input layer
    }
    
    best_acc = 0.0
    
    print(f"\nTraining with paraphrase probability: {paraphrase_prob * 100:.1f}%")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0
        layer_diffs = [[] for _ in range(len(model.layers) + 1)]  # +1 for input layer

        for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Handle input paraphrasing
            was_input_paraphrased = np.random.random() < paraphrase_prob
            if was_input_paraphrased:
                with torch.no_grad():
                    current = paraphraser(images)
                # Calculate metrics between original and paraphrased input
                mse = F.mse_loss(images, current).item()
                cosine_sim = F.cosine_similarity(images.view(images.size(0), -1),
                                               current.view(current.size(0), -1),
                                               dim=1).mean().item()
            else:
                current = images
                mse = 0.0  # No paraphrasing occurred
                cosine_sim = 1.0  # Identity mapping
            
            # Store input layer metrics
            layer_diffs[0].append({
                'mse': mse, 
                'cosine_sim': cosine_sim,
                'was_paraphrased': was_input_paraphrased
            })
            
            # Forward pass tracking intermediates
            intermediates = []
            paraphrased_intermediates = []
            layer_originals = []  # Store original outputs for visualization
            
            # Process through each layer with probabilistic paraphrasing
            for i, layer in enumerate(model.layers, 1):  # Start from 1 since 0 is input
                # Get layer output
                current = layer(current)
                layer_originals.append(current.clone())  # Store original layer output
                
                # Probabilistically apply paraphrasing
                was_paraphrased = np.random.random() < paraphrase_prob
                if was_paraphrased:
                    with torch.no_grad():
                        paraphrased = paraphraser(current)
                    # Calculate metrics only if paraphrasing occurred
                    mse = F.mse_loss(current, paraphrased).item()
                    cosine_sim = F.cosine_similarity(current.view(current.size(0), -1),
                                                   paraphrased.view(paraphrased.size(0), -1),
                                                   dim=1).mean().item()
                else:
                    paraphrased = current
                    mse = 0.0  # No paraphrasing occurred
                    cosine_sim = 1.0  # Identity mapping
                
                intermediates.append(current)
                paraphrased_intermediates.append(paraphrased)
                
                layer_diffs[i].append({
                    'mse': mse, 
                    'cosine_sim': cosine_sim,
                    'was_paraphrased': was_paraphrased
                })
                
                current = paraphrased
            
            # Final classification
            logits = model.classifier(current)
            
            # Calculate loss
            class_loss = criterion(logits, labels)
            reg_loss = sum(F.mse_loss(inter, para) 
                          for inter, para in zip(intermediates, paraphrased_intermediates))
            
            total_loss = class_loss + 0.1 * reg_loss
            total_loss.backward()
            optimizer.step()

            # Update metrics
            batch_loss = total_loss.item()
            epoch_loss += batch_loss
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            batch_count += 1

            # Store batch-level metrics
            metrics_log['batch_metrics']['loss'].append(batch_loss)
            metrics_log['batch_metrics']['acc'].append(100 * (predicted == labels).sum().item() / labels.size(0))

            if batch_idx % 100 == 0:  # Reduced frequency for more meaningful plots
                visualize_intermediates(
                    original_input=images,
                    paraphrased_input=current if was_input_paraphrased else None,
                    layer_originals=layer_originals,
                    layer_paraphrased=paraphrased_intermediates,
                    layer_diffs=layer_diffs,
                    epoch=epoch,
                    batch_idx=batch_idx
                )
                
                # Log layer statistics
                for i in range(len(model.layers) + 1):
                    layer_metrics = layer_diffs[i][-100:]  # Use last 100 batches
                    metrics_log['layer_metrics'][i]['mse'].append(
                        np.mean([m['mse'] for m in layer_metrics]))
                    metrics_log['layer_metrics'][i]['cosine_sim'].append(
                        np.mean([m['cosine_sim'] for m in layer_metrics]))
                    metrics_log['layer_metrics'][i]['was_paraphrased'].append(
                        np.mean([m['was_paraphrased'] for m in layer_metrics]))
                    
                print(f'\nLayer Statistics (Paraphrase Prob: {paraphrase_prob * 100:.1f}%):')
                for i in range(len(model.layers) + 1):
                    layer_name = "Input" if i == 0 else f"Layer {i}"
                    paraphrase_rate = np.mean([m['was_paraphrased'] for m in layer_diffs[i][-100:]])
                    print(f'{layer_name}:')
                    print(f'  MSE: {metrics_log["layer_metrics"][i]["mse"][-1]:.4f}')
                    print(f'  Cosine Similarity: {metrics_log["layer_metrics"][i]["cosine_sim"][-1]:.4f}')
                    print(f'  Actual Paraphrase Rate: {paraphrase_rate * 100:.1f}%')

        # Epoch-level metrics
        train_acc = 100 * correct / total
        train_loss = epoch_loss / batch_count
        test_acc = evaluate_model(model, test_loader, device)
        
        metrics_log['train_acc'].append(train_acc)
        metrics_log['train_loss'].append(train_loss)
        metrics_log['test_acc'].append(test_acc)
        
        print(f'\nEpoch {epoch+1} (Paraphrase Prob: {paraphrase_prob * 100:.1f}%):')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        
        # Plot training progress
        plot_training_progress(metrics_log)
        
        # Save best model if specified
        if save_dir and test_acc > best_acc:
            best_acc = test_acc
            save_path = Path(save_dir) / f'interpretable_classifier_p{paraphrase_prob:.2f}.pth'
            torch.save(model.state_dict(), save_path)
    
    return metrics_log

# visualise the paraphraser iimages with their original counterparts
def visualize_paraphrasing(paraphraser, train_loader, device, num_samples=10):
    """Helper function to visualize paraphrased images"""
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
                          layer_paraphrased, layer_diffs, epoch, batch_idx, num_samples=1):
    """
    Visualize random sample of images, their intermediate representations, and paraphrased versions.
    Clearly indicates which versions were actually paraphrased vs copied.
    """
    # Set num_samples to 1 for brevity of output. Including a large number of samples gives better visibility over training
    batch_size = original_input.size(0)
    indices = torch.randperm(batch_size)[:num_samples]
    
    for idx in indices:
        num_cols = len(layer_originals) + 1  # +1 for input
        plt.figure(figsize=(3 * num_cols, 6))
        
        # Original input
        plt.subplot(2, num_cols, 1)
        plt.imshow(original_input[idx].cpu().squeeze(), cmap='gray')
        plt.title('Original Input')
        plt.axis('off')
        
        # Input paraphrasing status and visualization
        was_paraphrased = layer_diffs[0][-1]['was_paraphrased']
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
            plt.imshow(orig[idx].detach().cpu().squeeze(), cmap='gray')
            plt.title(f'Layer {i+1}')
            plt.axis('off')
            
            # Paraphrasing status for this layer
            was_paraphrased = layer_diffs[i+1][-1]['was_paraphrased']
            status = "Paraphrased" if was_paraphrased else "Original"
            
            # Paraphrased or original version
            plt.subplot(2, num_cols, num_cols + i + 2)
            plt.imshow(para[idx].detach().cpu().squeeze(), cmap='gray')
            plt.title(f'L{i+1} ({status})')
            plt.axis('off')
        
        plt.suptitle(f'Epoch {epoch+1}, Step {batch_idx}, Sample {idx.item()}')
        plt.tight_layout()
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

def main():
    # Setup device and paths
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    root_dir = Path(__file__).resolve().parent
    data_dir = root_dir / 'data'
    models_dir = root_dir / 'models'
    models_dir.mkdir(exist_ok=True)
    
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
    
    # Load or train prerequisite models
    classifier = PretrainedClassifier()
    paraphraser = ImageParaphraser()
    
    # Load/train classifier
    classifier_path = models_dir / 'initial_classifier.pth'
    if classifier_path.exists():
        print("Loading pretrained classifier...")
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    else:
        print("Training initial classifier...")
        classifier.to(device)
        train_classifier(classifier, train_loader, test_loader, device)
        torch.save(classifier.state_dict(), classifier_path)
    
    classifier.to(device)
    classifier.eval()
    
    # Load/train paraphraser
    paraphraser_path = models_dir / 'paraphraser.pth'
    if paraphraser_path.exists():
        print("Loading pretrained paraphraser...")
        paraphraser.load_state_dict(torch.load(paraphraser_path, map_location=device))
    else:
        print("Training paraphraser...")
        paraphraser.to(device)
        train_paraphraser(paraphraser, classifier, train_loader, device)
        torch.save(paraphraser.state_dict(), paraphraser_path)
    
    paraphraser.to(device)
    paraphraser.eval()
    
    # Train interpretable classifiers with different paraphrasing probabilities
    paraphrase_probs = [0.0, 0.5, 1.0]  # 0%, 50%, 100%
    all_metrics = []
    
    for prob in paraphrase_probs:
        print(f"\nTraining classifier with {prob * 100:.1f}% paraphrasing probability")
        model = InterpretableClassifier()
        metrics = train_interpretable_classifier(
            model, 
            paraphraser,
            train_loader, 
            test_loader, 
            paraphrase_prob=prob,
            device=device,
            save_dir=str(models_dir)
        )
        all_metrics.append(metrics)
        
        # Save final model and metrics
        torch.save(model.state_dict(), 
                  models_dir / f'interpretable_classifier_p{prob:.2f}.pth')
        torch.save(metrics, 
                  models_dir / f'training_metrics_p{prob:.2f}.pth')
    
    # Compare results
    compare_training_results(all_metrics)
    
    print("\nTraining complete! Models and metrics saved in:", models_dir)

if __name__ == "__main__":
    main()