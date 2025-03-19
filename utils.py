import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import config as ENV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def evaluate_model(model, test_loader, class_names, output_dir="output"):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(ENV.DEVICE), labels.to(ENV.DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute and Display Accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\n🔥 Test Accuracy: {accuracy * 100:.2f}%")

    # Classification Report
    print("\n📌 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "confusion_matrix.png")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    # Save the plot
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"📂 Confusion matrix saved to: {output_path}")


def plot_training_metrics(train_losses, val_losses, train_accuracies, val_accuracies, output_dir="output"):
    """
    Plots training & validation loss and accuracy, then saves the figures.
    
    Args:
        train_losses (list): List of training loss values per epoch.
        val_losses (list): List of validation loss values per epoch.
        train_accuracies (list): List of training accuracy values per epoch.
        val_accuracies (list): List of validation accuracy values per epoch.
        output_dir (str): Directory where the plots will be saved.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure
    plt.figure(figsize=(10, 5))
    
    # Plot Training and Validation Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training & Validation Loss')
    
    # Plot Training and Validation Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training & Validation Accuracy')
    
    # Save the plot
    plot_path = os.path.join(output_dir, f"{ENV.MODEL_NAME}.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    
    print(f"📂 Training metrics plot saved to: {plot_path}")

