import torch
import seaborn as sns
import matplotlib.pyplot as plt
import os
from config import DEVICE, MODEL_NAME
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def evaluate_model(model, test_loader, class_names, output_dir=f"output/test/{MODEL_NAME}"):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Lists to store predictions and true labels
    all_preds = []
    all_labels = []

    # Evaluate model without computing gradients
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Forward pass
            outputs = model(images)
            
            # Get predicted class indices
            predicted = torch.argmax(outputs, dim=1)
            
            # Collect predictions and true labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute and display accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\n Test Accuracy: {accuracy * 100:.2f}%")

    # Classification Report
    print("\n Classification Report:")
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)

    # Save classification report to a text file
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    
    # Save the plot
    output_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f" Confusion matrix saved")


def plot_training_metrics(train_losses, val_losses, train_accuracies, val_accuracies, output_dir=f"output/test/{MODEL_NAME}"):
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
    plot_path = os.path.join(output_dir, f"training metric.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    
    print(f" Training metrics plot saved")

