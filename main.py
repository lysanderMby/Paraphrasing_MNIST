'''
Used to quickly start the project.
This file will load if present or train from scratch a basic classifier and paraphrasing model on MNIST. 

Interpretable classifiers are trained using a given probabilities of applying the paraphrasing model found in paraphrase_schedules.
'''

import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import time
from datetime import datetime, timedelta

# Import from project modules
from initial_classifier.training_initial_classifier import PretrainedClassifier
from paraphraser.training_paraphraser import ImageParaphraser
from interpretable.interpretable_classifier import train_interpretable_classifier, InterpretableClassifier

TEST_MODE = True  # Flag to control test mode

def get_reduced_dataset(dataset, reduction_factor=100):
    """
    Reduce dataset size by taking every nth sample
    """
    indices = range(0, len(dataset), reduction_factor)
    return torch.utils.data.Subset(dataset, indices)

def main():
    """Finds or trains a classifier, paraphraser, and finally interpretable classifier.
    This is done with appropriate paraphrasing probabilities as found in the paraphrase_schedules list."""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Setup paths
    root_dir = Path(__file__).resolve().parent
    data_dir = root_dir / 'data'
    models_dir = root_dir / 'models'
    #models_dir.mkdir(exist_ok=True) # if this directory is not present, an error should be created clearly
    
    # Define model architecture configurations based on test mode
    if TEST_MODE:
        print("\nRunning in TEST MODE - using simplified configuration")
        layer_configs = [
            (1, 32),
            (32, 1)
        ]
        batch_size = 32
        num_epochs = 3
        num_workers = 0  # Reduce workers for testing
    else: # Main configurations to be run when not doing activate bug fixes
        layer_configs = [
            (1, 32),    # First layer expands channels
            (32, 48),   # Increase feature complexity
            (48, 64),   # Increase feature complexity
            (64, 64),   # Intermediate layers
            (64, 64),
            (64, 64),
            (64, 32),   # Gradually reduce channels
            (32, 16)    # Final interpretable representation
        ]
        batch_size = 64
        num_epochs = 3
        num_workers = 4
    
    # Setup data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(root=str(data_dir), train=True,
                                             download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root=str(data_dir), train=False,
                                            download=True, transform=transform)
    
    # Reduce dataset size if in test mode
    if TEST_MODE:
        train_dataset = get_reduced_dataset(train_dataset)
        test_dataset = get_reduced_dataset(test_dataset)
        print(f"Reduced dataset sizes - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            pin_memory=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            pin_memory=True, num_workers=num_workers)
    
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
    
    # Modify paraphrase schedules based on test mode
    if TEST_MODE:
        paraphrase_schedules = [
            0.2,
            [i * 0.5 / (num_epochs - 1) for i in range(num_epochs)],
            [0, 0.5, 1]
        ]
    else:
        paraphrase_schedules = [
            # Constant probabilities
            0.01,
            0.1,
            0.5,
            # Linear increase schedules
            [i * 0.5 / (num_epochs - 1) for i in range(num_epochs)],  # 0.0 to 0.5
            # Custom schedules
            [0.01] * 2 + [0.02] * 3 + [0.1] * 5,  # 10% for 5 epochs, then 30%
        ]
    
    print("\nTraining interpretable classifiers with different paraphrasing schedules...")
    print(f"Total schedules to train: {len(paraphrase_schedules)}")
    print("-" * 50)
    
    # Create experiment root directory with layer config name
    exp_name = f"layers_{len(layer_configs)}_channels_" + "_".join(str(cfg[1]) for cfg in layer_configs)
    exp_root = models_dir / exp_name
    exp_root.mkdir(exist_ok=True)
    
    # Save layer configuration
    with open(exp_root / "architecture_config.txt", "w") as f:
        f.write("Layer configurations:\n")
        for i, (in_ch, out_ch) in enumerate(layer_configs):
            f.write(f"Layer {i}: {in_ch} -> {out_ch} channels\n")

    # Print layer configurations
    for i, (in_ch, out_ch) in enumerate(layer_configs):
        print(f"Layer {i}: {in_ch} -> {out_ch} channels")
    
    total_start_time = time.time()
    print(f"Starting full training run at {datetime.now().strftime('%H:%M:%S')}")
    
    for schedule_idx, schedule in enumerate(paraphrase_schedules):
        schedule_start_time = time.time()
        
        print(f"\nRunning schedule {schedule_idx + 1} / {len(paraphrase_schedules)}")
        # Create schedule-specific directory
        if isinstance(schedule, (int, float)):
            schedule_name = f"constant_{schedule:.2f}"
            schedule_desc = f"{schedule*100:.0f}% constant paraphrasing"
        else:
            schedule_name = f"schedule_{schedule_idx}"
            schedule_desc = f"Custom schedule {schedule_idx}: {schedule}"
        
        print(f"Schedule description: {schedule_desc}")
        
        schedule_dir = exp_root / schedule_name
        schedule_dir.mkdir(exist_ok=True)
        
        model = InterpretableClassifier(layer_configs) # reinitialise the model
        print(f"Total interpretable classifier parameter number is {model.get_param_count()}")
        
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
        
        # Schedule timing information
        schedule_time = time.time() - schedule_start_time
        total_time = time.time() - total_start_time
        remaining_schedules = len(paraphrase_schedules) - (schedule_idx + 1)
        avg_schedule_time = total_time / (schedule_idx + 1)
        estimated_remaining = avg_schedule_time * remaining_schedules
        
        print("\n" + "=" * 50)
        print(f"Schedule {schedule_idx + 1} completed in {timedelta(seconds=int(schedule_time))}")
        print(f"Total training time so far: {timedelta(seconds=int(total_time))}")
        if remaining_schedules > 0:
            print(f"Estimated time remaining: {timedelta(seconds=int(estimated_remaining))}")
            print(f"""Estimated completion time: 
                  {(datetime.now() + timedelta(seconds=int(estimated_remaining))).strftime('%H:%M:%S')} 
                  on {(datetime.now() + timedelta(seconds=int(estimated_remaining))).strftime('%d/%m/%Y')}""")
        print("=" * 50)
    
    # Final timing information
    total_training_time = time.time() - total_start_time
    print(f"\nAll training completed at {datetime.now().strftime('%H:%M:%S')}")
    print(f"Total training time: {timedelta(seconds=int(total_training_time))}")
    print(f"Average time per schedule: {timedelta(seconds=int(total_training_time/len(paraphrase_schedules)))}")
    print(f"\nModels and metrics saved in: {exp_root}")

if __name__ == "__main__":
    main()