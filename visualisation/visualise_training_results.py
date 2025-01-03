import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import numpy as np
import traceback

def plot_schedule_comparison(exp_dir: Path):
    """Compare results across different paraphrasing schedules"""
    schedules = [d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith(('constant_', 'schedule_'))]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    plot_count = 0  # Track if we actually plot anything
    
    for schedule_dir in schedules:
        try:
            # Load metrics directly from schedule directory
            metrics_path = schedule_dir / 'training_metrics.pth'
            print(f"  Loading metrics from: {metrics_path}")
            
            metrics = torch.load(metrics_path, weights_only=True)
            print(f"  Loaded metrics keys: {metrics.keys()}")
            print(f"  Train acc shape: {len(metrics['train_acc'])}")
            print(f"  Test acc shape: {len(metrics['test_acc'])}")
            print(f"  Loss shape: {len(metrics['train_loss'])}")
            
            schedule_name = schedule_dir.name
            
            # Get schedule description from config
            config_path = schedule_dir / 'schedule_config.txt'
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    desc = f.readline().split(': ')[1].strip()
            else:
                desc = schedule_name
            
            # Plot training accuracy
            ax1.plot(metrics['train_acc'], label=desc)
            ax1.set_title('Training Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy (%)')
            
            # Plot test accuracy
            ax2.plot(metrics['test_acc'], label=desc)
            ax2.set_title('Test Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            
            # Plot training loss
            ax3.plot(metrics['train_loss'], label=desc)
            ax3.set_title('Training Loss')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss')
            
            # Plot paraphrase schedule if available
            if 'paraphrase_schedule' in metrics:
                ax4.plot(metrics['paraphrase_schedule'], label=desc)
                ax4.set_title('Paraphrase Probability Schedule')
                ax4.set_xlabel('Epoch')
                ax4.set_ylabel('Probability')
            
            plot_count += 1
            
        except Exception as e:
            print(f"  WARNING: Failed to process {schedule_dir}: {str(e)}")
            print(f"  Traceback: {traceback.format_exc()}")
            continue
    
    if plot_count == 0:
        print("  ERROR: No data was successfully plotted!")
        return
    
    print(f"  Successfully plotted data for {plot_count} schedules")
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(exp_dir / 'schedule_comparison.png', bbox_inches='tight')
    plt.close()

def plot_best_model_analysis(exp_dir: Path):
    """Analyze and visualize the best performing model"""
    schedules = [d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith(('constant_', 'schedule_'))]
    
    best_acc = 0
    best_schedule = None
    best_metrics = None
    best_run = None  # Not needed since we don't have run directories
    
    print("\n  Best Model Analysis:")
    print("  -------------------")
    
    for schedule_dir in schedules:
        try:
            metrics_path = schedule_dir / 'training_metrics.pth'
            print(f"  Checking metrics from: {metrics_path}")
            
            metrics = torch.load(metrics_path, weights_only=True)
            print(f"  Loaded metrics keys: {metrics.keys()}")
            
            max_acc = max(metrics['test_acc'])
            print(f"  Max accuracy: {max_acc:.2f}%")
            
            if max_acc > best_acc:
                best_acc = max_acc
                best_schedule = schedule_dir
                best_metrics = metrics
                print(f"  New best model found in {schedule_dir}")
        except Exception as e:
            print(f"  WARNING: Failed to process {schedule_dir}: {str(e)}")
            print(f"  Traceback: {traceback.format_exc()}")
            continue
    
    if best_metrics is None:
        print("  WARNING: No valid metrics found in any schedule directory")
        return
    
    print(f"\n  Creating visualization for best model:")
    print(f"  Best schedule: {best_schedule.name}")
    print(f"  Peak accuracy: {best_acc:.2f}%")
    
    # Create visualization for best model
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot accuracy convergence
    axes[0,0].plot(best_metrics['train_acc'], label='Train')
    axes[0,0].plot(best_metrics['test_acc'], label='Test')
    axes[0,0].set_title(f'Best Model Accuracy\nPeak: {best_acc:.2f}%')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Accuracy (%)')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Plot loss curve
    axes[0,1].plot(best_metrics['train_loss'])
    axes[0,1].set_title('Training Loss')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Loss')
    axes[0,1].grid(True)
    
    # Plot accuracy improvement rate
    acc_diffs = np.diff(best_metrics['test_acc'])
    axes[1,0].plot(acc_diffs)
    axes[1,0].set_title('Test Accuracy Improvement Rate')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Accuracy Change (%)')
    axes[1,0].grid(True)
    
    # Add summary statistics
    summary_text = (
        f"Best Schedule: {best_schedule.name}\n"
        f"Peak Accuracy: {best_acc:.2f}%\n"
        f"Final Accuracy: {best_metrics['test_acc'][-1]:.2f}%\n"
        f"Convergence Epoch: {np.argmax(best_metrics['test_acc'])+1}"
    )
    axes[1,1].text(0.5, 0.5, summary_text, 
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=axes[1,1].transAxes,
                   bbox=dict(facecolor='white', alpha=0.8))
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig(exp_dir / 'best_model_analysis.png')
    plt.close()
    
    print("  ✓ Best model analysis plot saved")

def main():
    # Get absolute path and print current working directory
    print(f"Current working directory: {Path.cwd()}")
    
    # Try multiple possible locations for models directory
    possible_paths = [
        Path('models'),
        Path('../models'), # current directory is models
        Path.cwd() / 'models',
        Path(__file__).parent.parent / 'models'
    ]
    
    models_dir = None
    for path in possible_paths:
        print(f"Checking for models directory at: {path}")
        if path.exists() and path.is_dir():
            models_dir = path
            print(f"Found models directory at: {path}")
            break
    
    if models_dir is None:
        print("ERROR: Could not find models directory!")
        return
    
    # Find all experiment directories
    exp_dirs = [d for d in models_dir.iterdir() 
               if d.is_dir() and d.name.startswith('layers_')]
    
    if not exp_dirs:
        print(f"WARNING: No experiment directories found in {models_dir}")
        return
    
    print(f"\nFound {len(exp_dirs)} experiment directories:")
    for d in exp_dirs:
        print(f"  - {d}")
    
    for exp_dir in exp_dirs:
        print(f"\nProcessing {exp_dir.name}:")
        
        # Check for required files
        schedules = [d for d in exp_dir.iterdir() 
                    if d.is_dir() and d.name.startswith(('constant_', 'schedule_'))]
        if not schedules:
            print(f"  WARNING: No schedule directories found in {exp_dir}")
            continue
            
        print(f"  Found {len(schedules)} schedule directories")
        
        try:
            plot_schedule_comparison(exp_dir)
            print(f"  ✓ Generated schedule comparison plot")
        except Exception as e:
            print(f"  ✗ Failed to generate schedule comparison plot: {str(e)}")
            
        try:
            plot_best_model_analysis(exp_dir)
            print(f"  ✓ Generated best model analysis plot")
        except Exception as e:
            print(f"  ✗ Failed to generate best model analysis plot: {str(e)}")
        
        # Verify files were created
        expected_files = ['schedule_comparison.png', 'best_model_analysis.png']
        for file in expected_files:
            path = exp_dir / file
            if path.exists():
                print(f"  ✓ Successfully saved {file}")
            else:
                print(f"  ✗ Failed to save {file}")

if __name__ == "__main__":
    main()