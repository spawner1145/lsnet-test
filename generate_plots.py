import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_log_file(log_path):
    epochs = []
    train_lr = []
    train_loss = []
    test_loss = []
    test_acc1 = []
    test_acc5 = []

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    if 'test_loss' in data:
                        epochs.append(data['epoch'])
                        train_lr.append(data['train_lr'])
                        train_loss.append(data['train_loss'])
                        test_loss.append(data['test_loss'])
                        test_acc1.append(data['test_acc1'])
                        test_acc5.append(data['test_acc5'])
                except json.JSONDecodeError:
                    continue

    return {
        'epochs': np.array(epochs),
        'train_lr': np.array(train_lr),
        'train_loss': np.array(train_loss),
        'test_loss': np.array(test_loss),
        'test_acc1': np.array(test_acc1),
        'test_acc5': np.array(test_acc5)
    }

def create_plots(data, output_dir):
    epochs = data['epochs']
    steps = range(len(epochs))

    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('LSNet Classification Training Results', fontsize=16, fontweight='bold')

    ax1.plot(steps, data['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(steps, data['test_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, data['test_acc1'], 'g-', label='Top-1 Accuracy', linewidth=2)
    ax2.plot(steps, data['test_acc5'], 'orange', label='Top-5 Accuracy', linewidth=2)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.plot(steps, data['train_lr'], 'purple', linewidth=2)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)

    window_size = 10
    if len(epochs) > window_size:
        smooth_test_loss = np.convolve(data['test_loss'], np.ones(window_size)/window_size, mode='valid')
        smooth_test_acc1 = np.convolve(data['test_acc1'], np.ones(window_size)/window_size, mode='valid')
        smooth_steps = range(window_size-1, len(epochs))

        ax4_twin = ax4.twinx()
        line1 = ax4.plot(smooth_steps, smooth_test_loss, 'r-', label='Validation Loss', linewidth=2)
        line2 = ax4_twin.plot(smooth_steps, smooth_test_acc1, 'g-', label='Top-1 Accuracy', linewidth=2)

        ax4.set_xlabel('Step')
        ax4.set_ylabel('Loss', color='r')
        ax4_twin.set_ylabel('Accuracy (%)', color='g')
        ax4.set_title(f'Loss vs Accuracy (Smoothed, window={window_size})')

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='center right')
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(output_dir / 'training_results.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'training_results.pdf', bbox_inches='tight')
    plt.close()

    create_detailed_plots(data, output_dir)

def create_detailed_plots(data, output_dir):
    epochs = data['epochs']
    steps = range(len(epochs))

    plt.figure(figsize=(12, 8))
    plt.plot(steps, data['train_loss'], 'b-', linewidth=1.5, alpha=0.8)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'train_loss_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(steps, data['test_loss'], 'r-', linewidth=1.5, alpha=0.8)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title('Validation Loss Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'val_loss_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(steps, data['test_acc1'], 'g-', linewidth=1.5, alpha=0.8)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Top-1 Accuracy (%)', fontsize=12)
    plt.title('Top-1 Validation Accuracy', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'top1_acc_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(steps, data['test_acc5'], 'orange', linewidth=1.5, alpha=0.8)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Top-5 Accuracy (%)', fontsize=12)
    plt.title('Top-5 Validation Accuracy', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'top5_acc_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_statistics(data):
    print("Training Statistics")

    epochs = data['epochs']
    train_loss = data['train_loss']
    test_loss = data['test_loss']
    test_acc1 = data['test_acc1']
    test_acc5 = data['test_acc5']

    print(f"Total training epochs: {len(epochs)}")
    print(f"Final training loss: {train_loss[-1]:.4f}")
    print(f"Final validation loss: {test_loss[-1]:.4f}")
    print(f"Final Top-1 accuracy: {test_acc1[-1]:.2f}%")
    print(f"Final Top-5 accuracy: {test_acc5[-1]:.2f}%")

    best_acc1_epoch = np.argmax(test_acc1)
    best_acc5_epoch = np.argmax(test_acc5)
    min_train_loss_epoch = np.argmin(train_loss)
    min_test_loss_epoch = np.argmin(test_loss)

    print(f"\nBest Top-1 accuracy: {test_acc1[best_acc1_epoch]:.2f}% (epoch {best_acc1_epoch})")
    print(f"Best Top-5 accuracy: {test_acc5[best_acc5_epoch]:.2f}% (epoch {best_acc5_epoch})")
    print(f"Lowest training loss: {train_loss[min_train_loss_epoch]:.4f} (epoch {min_train_loss_epoch})")
    print(f"Lowest validation loss: {test_loss[min_test_loss_epoch]:.4f} (epoch {min_test_loss_epoch})")

    final_50_epochs = slice(-50, None)
    print(f"\nLast 50 epochs average:")
    print(f"  Training loss: {np.mean(train_loss[final_50_epochs]):.4f} ± {np.std(train_loss[final_50_epochs]):.4f}")
    print(f"  Validation loss: {np.mean(test_loss[final_50_epochs]):.4f} ± {np.std(test_loss[final_50_epochs]):.4f}")
    print(f"  Top-1 accuracy: {np.mean(test_acc1[final_50_epochs]):.2f}% ± {np.std(test_acc1[final_50_epochs]):.2f}%")
    print(f"  Top-5 accuracy: {np.mean(test_acc5[final_50_epochs]):.2f}% ± {np.std(test_acc5[final_50_epochs]):.2f}%")

def main():
    log_path = Path("log.txt")
    output_dir = Path("output/plots")
    output_dir.mkdir(exist_ok=True)

    print("Parsing training log...")
    data = parse_log_file(log_path)

    print("Generating plots...")
    create_plots(data, output_dir)

    print("Printing statistics...")
    print_statistics(data)

    print(f"\nPlots saved to: {output_dir}")
    print("Generated files:")
    for file in output_dir.glob("*.png"):
        print(f"  - {file.name}")
    for file in output_dir.glob("*.pdf"):
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()
