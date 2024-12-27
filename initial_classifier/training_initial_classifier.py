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
from torchvision.models import resnet18
import os.path
from tqdm import tqdm
import os
from pathlib import Path

# Initial classifier model class
class PretrainedClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 28x28
        # First two convolution layers do not change the 
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 28x28 -> 28x28
        self.pool = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14

        # Size progression:
        # After first pool: 14x14x32
        # After conv2 + pool: 7x7x64
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # First conv + pool
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14

        # Second conv + pool
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7

        # Flatten and fully connected layers
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Train initial classifier on the original MNIST dataset
def train_classifier(model, train_loader, test_loader, device, num_epochs=10):
    """Train the initial classifier on original data"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model = model.to(device)  # Ensure model is on correct device
    criterion = criterion.to(device)  # Move loss function to device if needed

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader):
            # Move input data to device
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Evaluate on test set
        test_acc = evaluate_model(model, test_loader, device)
        print(f'Epoch {epoch+1}: Train Acc: {100*correct/total:.2f}%, Test Acc: {test_acc:.2f}%')

def evaluate_model(model, test_loader, device):
    # Evaluate model on the test set provided in test loaded
    # Moving this information to a separate device has questionable performance improvements at these scales
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
    
    # Mount drive and setup save directory
    save_dir = ''

    # Set data directory path (one level up)
    root_dir = Path(__file__).resolve().parent
    data_dir = root_dir / 'data'
    models_dir = root_dir / 'models'
    #data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    
    # Data loading with GPU support
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(root=data_dir, train=True,
                                             download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False,
                                            download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, 
                            pin_memory=True, num_workers=4)  # Added num_workers for efficiency. May not be appropriate for your hardware
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, 
                            pin_memory=True, num_workers=4)
    
    # Initialise classifier
    classifier = PretrainedClassifier()  # Move to device after loading
    
    # Train model
    classifier_path = os.path.join(models_dir, 'initial_classifier.pth')
    
    train_classifier(classifier, train_loader, test_loader, device)
    torch.save(classifier.state_dict(), classifier_path)

if __name__ == "__main__":
    main()
