import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Import from project modules
from initial_classifier.training_initial_classifier import PretrainedClassifier
from paraphraser.training_paraphraser import ImageParaphraser
from interpretable.interpretable_classifier import train_interpretable_classifier, InterpretableClassifier

def main():
    """Main entry point for training the interpretable classifier"""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Setup paths
    root_dir = Path(__file__).resolve().parent
    data_dir = root_dir / 'data'
    models_dir = root_dir / 'models'
    models_dir.mkdir(exist_ok=True)
    
    # Setup data loading
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
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier.to(device)
    classifier.eval()
    
    # Load paraphraser
    paraphraser_path = models_dir / 'paraphraser.pth'
    if not paraphraser_path.exists():
        raise FileNotFoundError(
            "Paraphraser not found! Please run paraphrasing/training_paraphraser.py first."
        )
    print("Loading pretrained paraphraser...")
    paraphraser.load_state_dict(torch.load(paraphraser_path, map_location=device))
    paraphraser.to(device)
    paraphraser.eval()
    
    # Train interpretable classifiers with different paraphrasing probabilities
    paraphrase_probs = [0.0, 0.5, 1.0]  # 0%, 50%, 100%
    print("\nTraining interpretable classifiers with different paraphrasing probabilities...")
    print(f"Probabilities to test: {[f'{p*100:.0f}%' for p in paraphrase_probs]}")
    
    for prob in paraphrase_probs:
        print(f"\nTraining classifier with {prob*100:.0f}% paraphrasing probability")
        model = InterpretableClassifier()
        
        metrics = train_interpretable_classifier(
            model=model,
            paraphraser=paraphraser,
            train_loader=train_loader,
            test_loader=test_loader,
            paraphrase_prob=prob,
            device=device,
            save_dir=str(models_dir)
        )
        
        # Save final model and metrics
        torch.save(model.state_dict(), 
                  models_dir / f'interpretable_classifier_p{prob:.2f}.pth')
        torch.save(metrics, 
                  models_dir / f'training_metrics_p{prob:.2f}.pth')
    
    print("\nTraining complete! Models and metrics saved in:", models_dir)

if __name__ == "__main__":
    main()