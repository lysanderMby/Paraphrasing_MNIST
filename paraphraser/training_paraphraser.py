# Package Imports

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.models import resnet18 # basis for the paraphraser model
from tqdm import tqdm
import os
import os
import sys
from pathlib import Path

# Finding parent directory path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

# Imports from directory
from initial_classifier.train_classifier import train_classifier, PretrainedClassifier

class ImageParaphraser(nn.Module):
    # Resnet-based classifier which maximally transforms images while maintaining their classification
    # Uses a pretrained classifier, and strongly punishes changes which result in a change in classification (logits)
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            # Input: 28x28x1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # 28x28x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),  # 14x14x64 - Use average pooling instead of max pooling

            # ResNet-style blocks
            self._make_res_block(64, 128),  # 14x14x128
            nn.AvgPool2d(2, 2),  # 7x7x128
            self._make_res_block(128, 256),  # 7x7x256
        )

        # Style embedding
        self.style_linear = nn.Linear(32, 256)

        # Decoder with anti-aliasing
        self.decoder = nn.Sequential(
            # 7x7x256 -> 14x14x128
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # Use bilinear upsampling
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # Anti-aliasing convolution
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 14x14x128 -> 28x28x64
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Smooth final output
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def _make_res_block(self, in_channels, out_channels):
        """Create a ResNet-style block with smoothing"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # Add a small amount of average pooling to suppress high frequencies
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # Generate random style vector
        style = torch.randn(batch_size, 32).to(x.device)
        style = self.style_linear(style)
        style = style.view(batch_size, -1, 1, 1)

        # Encode
        x = self.encoder(x)

        # Add style information (reduced magnitude to prevent artifacts)
        x = x + 0.05 * style.expand(-1, -1, x.size(2), x.size(3))

        # Decode with smoothing
        x = self.decoder(x)

        return x
    
# Paraphraser loss function

class ParaphrasingLoss(nn.Module):
    """Simplified loss function focused on classification preservation with moderate transformation"""
    def __init__(self, classifier, classification_weight=5.0, transform_weight=0.5):
        super().__init__()
        self.classifier = classifier
        self.classification_weight = classification_weight
        self.transform_weight = transform_weight

    def forward(self, original, paraphrased, true_labels):
        # Classification preservation using original classifier
        with torch.no_grad():
            original_logits = self.classifier(original)
        paraphrased_logits = self.classifier(paraphrased)

        classification_loss = F.cross_entropy(paraphrased_logits, true_labels)

        # Simple L2 difference to encourage moderate transformation
        transform_loss = -F.mse_loss(original, paraphrased)  # Negative because we want some difference

        # Total loss
        total_loss = (self.classification_weight * classification_loss +
                     self.transform_weight * transform_loss)

        # For logging: compute classification accuracy
        with torch.no_grad():
            _, predicted = torch.max(paraphrased_logits, 1)
            correct = (predicted == true_labels).sum().item()
            total = true_labels.size(0)
            accuracy = 100 * correct / total

            # Get confusion matrix data
            confusion = torch.zeros(10, 10, device=original.device)
            for t, p in zip(true_labels, predicted):
                confusion[t.item(), p.item()] += 1

        return total_loss, classification_loss, accuracy, confusion
    
# Train paraphraser

def train_paraphraser(paraphraser, classifier, train_loader, num_epochs=2):
    """Train the paraphraser with comprehensive logging"""
    optimizer = optim.Adam(paraphraser.parameters(), lr=0.0001)
    criterion = ParaphrasingLoss(classifier)

    # For logging
    epoch_accuracies = []
    epoch_losses = []
    confusion_matrices = []

    print("\nStarting paraphraser training:")
    print("--------------------------------")

    for epoch in range(num_epochs):
        paraphraser.train()
        running_loss = 0.0
        running_class_loss = 0.0
        running_accuracy = 0.0
        epoch_confusion = torch.zeros(10, 10)
        batch_count = 0

        for images, labels in tqdm(train_loader):
            optimizer.zero_grad()

            # Generate paraphrased images
            paraphrased = paraphraser(images)

            # Compute losses and accuracy
            loss, class_loss, accuracy, confusion = criterion(images, paraphrased, labels)
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            running_loss += loss.item()
            running_class_loss += class_loss.item()
            running_accuracy += accuracy
            epoch_confusion += confusion.cpu()
            batch_count += 1

            # Print progress every 100 batches
            if batch_count % 100 == 0:
                print(f"Batch {batch_count}: Loss = {loss.item():.4f}, Accuracy = {accuracy:.1f}%")

        # Compute epoch averages
        avg_loss = running_loss / batch_count
        avg_class_loss = running_class_loss / batch_count
        avg_accuracy = running_accuracy / batch_count

        # Store metrics
        epoch_accuracies.append(avg_accuracy)
        epoch_losses.append(avg_loss)
        confusion_matrices.append(epoch_confusion)

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Classification Loss: {avg_class_loss:.4f}")
        print(f"Average Classification Accuracy: {avg_accuracy:.1f}%")

        # Display confusion matrix
        print("\nConfusion Matrix (True vs Predicted):")
        confusion_matrix = epoch_confusion.numpy()
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        normalized_confusion = 100 * confusion_matrix / row_sums # percentage scores

        # Print normalized confusion matrix
        print("\nNormalized Confusion Matrix (%):")
        print("    ", end="")
        for i in range(10):
            print(f"{i:5d}", end="")
        print("\n" + "-" * 60)

        for i in range(10):
            print(f"{i:2d} |", end="")
            for j in range(10):
                print(f"{normalized_confusion[i,j]:5.1f}", end="") # i is true, j is predicted
            print(f" | {confusion_matrix[i].sum():.0f}")
        print("-" * 60) # goated line

        fig = plt.imshow(normalized_confusion, cmap='viridis', interpolation ='nearest', origin ='lower')
        plt.colorbar(fig)
        plt.title('Confusion on Paraphrased Images')
        plt.xlabel('True Class') # check this way around. Might be transpossed
        plt.ylabel('Predicted Class')
        plt.show()

        # Visualize examples if it's the first or last epoch
        #if epoch in [0, num_epochs-1]:
        # No - visualise all examples
        visualize_paraphrasing(paraphraser, train_loader, num_samples=10)

    # Plot training progress
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epoch_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(epoch_accuracies)
    plt.title('Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')

    plt.tight_layout()
    plt.show()

    return epoch_accuracies, epoch_losses, confusion_matrices

# visualise the paraphraser iimages with their original counterparts
def visualize_paraphrasing(paraphraser, dataloader, num_samples=5):
    paraphraser.eval()
    max_num = len(dataloader)
    rand_index = np.random.randint(0, max_num)
    #images, labels = next(iter(dataloader))
    images, labels = dataloader[rand_index] # select random samples to visualise

    with torch.no_grad():
        paraphrased = paraphraser(images)

    plt.figure(figsize=(15, 6))
    for i in range(num_samples):
        # Original image
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f'Original\nLabel: {labels[i]}')
        plt.axis('off')

        # Paraphrased image
        plt.subplot(2, num_samples, i + num_samples + 1)
        plt.imshow(paraphrased[i].squeeze(), cmap='gray')
        plt.title('Paraphrased')
        plt.axis('off')

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

def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Set up paths
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / 'data'
    classifier_dir = root_dir / 'initial_classifier'
    classifier_path = classifier_dir / 'initial_classifier.pth'
    paraphraser_path = Path(__file__).parent / 'paraphraser.pth'
    
    # Data loading with GPU support
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
    
    # Initialize classifier and try to load pretrained weights
    classifier = PretrainedClassifier()
    
    if classifier_path.exists():
        print("Loading pretrained classifier...")
        state_dict = torch.load(classifier_path, map_location=device)
        classifier.load_state_dict(state_dict)
        classifier.to(device)
    else:
        print("Training classifier from scratch...")
        # Import necessary components from initial_classifier
        classifier.to(device) # empty model class
        train_classifier(classifier, train_loader, test_loader, device)
        #os.makedirs(classifier_dir, exist_ok=True) # directory must exist for training to take place
        torch.save(classifier.state_dict(), classifier_path)
    
    # Set classifier to eval mode
    classifier.eval()
    
    # Initialize and train paraphraser
    paraphraser = ImageParaphraser()
    print("Training paraphraser...")
    paraphraser.to(device)
    train_paraphraser(paraphraser, classifier, train_loader, device)
    #os.makedirs(paraphraser_path.parent, exist_ok=True)
    torch.save(paraphraser.state_dict(), paraphraser_path)

if __name__ == "__main__":
    main()
