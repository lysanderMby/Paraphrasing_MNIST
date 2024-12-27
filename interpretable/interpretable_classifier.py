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
                                 num_epochs=5, device='cuda', save_dir=None):
    """Train the interpretable classifier with paraphrasing at each layer"""
    model = model.to(device)
    paraphraser = paraphraser.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    metrics_log = {
        'train_acc': [],
        'test_acc': [],
        'layer_metrics': {i: {'mse': [], 'cosine_sim': []} for i in range(len(model.layers))}
    }
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        layer_diffs = [[] for _ in range(len(model.layers))]

        for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Forward pass tracking intermediates
            current = images
            intermediates = []
            paraphrased_intermediates = []
            
            # Process through each layer with paraphrasing
            for i, layer in enumerate(model.layers):
                # Get layer output
                current = layer(current)
                intermediates.append(current)
                
                # Store original layer output
                orig_output = current.clone()
                
                # Apply paraphrasing
                with torch.no_grad():
                    paraphrased = paraphraser(current)
                paraphrased_intermediates.append(paraphrased)
                
                # Calculate metrics between original and paraphrased versions
                orig_norm = (orig_output - orig_output.min()) / (orig_output.max() - orig_output.min() + 1e-8)
                para_norm = (paraphrased - paraphrased.min()) / (paraphrased.max() - paraphrased.min() + 1e-8)
                
                mse = F.mse_loss(orig_norm, para_norm).item()
                cosine_sim = F.cosine_similarity(orig_output.view(orig_output.size(0), -1),
                                               paraphrased.view(paraphrased.size(0), -1),
                                               dim=1).mean().item()
                
                layer_diffs[i].append({'mse': mse, 'cosine_sim': cosine_sim})
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

            running_loss += total_loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Visualize intermediates periodically
            if batch_idx % 500 == 0:
                visualize_intermediates(images, intermediates, 
                                     paraphrased_intermediates, paraphraser,
                                     epoch, batch_idx)
                
                # Log layer statistics
                for i in range(len(model.layers)):
                    layer_metrics = layer_diffs[i][-500:]
                    avg_mse = np.mean([m['mse'] for m in layer_metrics])
                    avg_cos = np.mean([m['cosine_sim'] for m in layer_metrics])
                    metrics_log['layer_metrics'][i]['mse'].append(avg_mse)
                    metrics_log['layer_metrics'][i]['cosine_sim'].append(avg_cos)
                    
                print(f'\nLayer Statistics:')
                for i in range(len(model.layers)):
                    print(f'Layer {i+1}:')
                    print(f'  MSE: {metrics_log["layer_metrics"][i]["mse"][-1]:.4f}')
                    print(f'  Cosine Similarity: {metrics_log["layer_metrics"][i]["cosine_sim"][-1]:.4f}')

        # Evaluate
        train_acc = 100 * correct / total
        test_acc = evaluate_model(model, test_loader, device)
        
        metrics_log['train_acc'].append(train_acc)
        metrics_log['test_acc'].append(test_acc)
        
        print(f'\nEpoch {epoch+1}:')
        print(f'Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        
        # Plot training progress
        plot_training_progress(metrics_log)
        
        # Save best model if specified
        if save_dir and test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 
                      os.path.join(save_dir, 'best_interpretable_classifier.pth'))
    
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

def visualize_intermediates(images, intermediates, paraphrased, paraphraser, epoch, step, num_samples=4):
    """Visualize random sample of images, their intermediate representations, and paraphrased versions"""
    batch_size = images.size(0)
    # Randomly select indices
    indices = torch.randperm(batch_size)[:num_samples]
    
    for idx in indices:
        num_cols = len(intermediates) + 1  # +1 for original image
        plt.figure(figsize=(3 * num_cols, 6))
        
        # Original image and its paraphrased version
        plt.subplot(2, num_cols, 1)
        plt.imshow(images[idx].cpu().squeeze(), cmap='gray')
        plt.title('Original')
        plt.axis('off')
        
        # Paraphrased original (using first paraphraser application)
        plt.subplot(2, num_cols, num_cols + 1)
        with torch.no_grad():
            paraphrased_input = paraphraser(images[idx:idx+1].to(images.device))
        plt.imshow(paraphrased_input[0].cpu().squeeze(), cmap='gray')
        plt.title('Paraphrased Input')
        plt.axis('off')
        
        # Intermediate representations and their paraphrased versions
        for i, (inter, para) in enumerate(zip(intermediates, paraphrased)):
            # Original intermediate
            plt.subplot(2, num_cols, i + 2)
            plt.imshow(inter[idx].detach().cpu().squeeze(), cmap='gray')
            plt.title(f'Layer {i+1}')
            plt.axis('off')
            
            # Paraphrased version
            plt.subplot(2, num_cols, num_cols + i + 2)
            plt.imshow(para[idx].detach().cpu().squeeze(), cmap='gray')
            plt.title(f'Paraphrased L{i+1}')
            plt.axis('off')
        
        plt.suptitle(f'Epoch {epoch+1}, Step {step}, Sample {idx.item()}')
        plt.tight_layout()
        plt.show()
        
    # Plot activation statistics
    plt.figure(figsize=(15, 5))
    for i, (inter, para) in enumerate(zip(intermediates, paraphrased)):
        plt.subplot(1, len(intermediates), i + 1)
        
        # Plot activation distributions
        inter_vals = inter.detach().cpu().numpy().flatten()
        para_vals = para.detach().cpu().numpy().flatten()
        
        plt.hist(inter_vals, bins=50, alpha=0.5, label='Original', density=True)
        plt.hist(para_vals, bins=50, alpha=0.5, label='Paraphrased', density=True)
        
        plt.title(f'Layer {i+1} Activations')
        plt.xlabel('Activation Value')
        plt.ylabel('Density')
        plt.legend()
    
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

def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Setup paths
    root_dir = Path(__file__).resolve().parent
    data_dir = root_dir / 'data'
    models_dir = root_dir / 'models'
    models_dir.mkdir(exist_ok=True)
    
    # Model paths
    classifier_path = models_dir / 'initial_classifier.pth'
    paraphraser_path = models_dir / 'paraphraser.pth'
    interpretable_path = models_dir / 'interpretable_classifier.pth'
    
    # Data loading
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
    
    # Initialize models
    classifier = PretrainedClassifier()
    paraphraser = ImageParaphraser()
    interpretable_classifier = InterpretableClassifier()
    
    # Load or train initial classifier
    if classifier_path.exists():
        print("Loading pretrained classifier...")
        state_dict = torch.load(classifier_path, map_location=device)
        classifier.load_state_dict(state_dict)
    else:
        print("Training initial classifier...")
        classifier.to(device)
        train_classifier(classifier, train_loader, test_loader, device)
        torch.save(classifier.state_dict(), classifier_path)
    
    classifier.to(device)
    classifier.eval()
    
    # Load or train paraphraser
    if paraphraser_path.exists():
        print("Loading pretrained paraphraser...")
        state_dict = torch.load(paraphraser_path, map_location=device)
        paraphraser.load_state_dict(state_dict)
    else:
        print("Training paraphraser...")
        paraphraser.to(device)
        train_paraphraser(paraphraser, classifier, train_loader, device)
        torch.save(paraphraser.state_dict(), paraphraser_path)
    
    paraphraser.to(device)
    paraphraser.eval()
    
    # Train interpretable classifier
    print("Training interpretable classifier...")
    interpretable_classifier.to(device)
    metrics_log = train_interpretable_classifier(
        interpretable_classifier, 
        paraphraser,
        train_loader, 
        test_loader, 
        device=device,
        save_dir=str(models_dir)
    )
    
    # Save final model and metrics
    torch.save(interpretable_classifier.state_dict(), interpretable_path)
    torch.save(metrics_log, models_dir / 'training_metrics.pth')
    
    print("\nTraining complete! Models saved in:", models_dir)

if __name__ == "__main__":
    main()
