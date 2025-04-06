import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from config import TEST_DATA_PATH, BATCH_SIZE, DEVICE, MODEL_PATH, KFOLD_SPLITS, MODEL_NAME
from models.convnext_cbam import ConvNeXt_CBAM
from load_data_kfold import test_loader, num_classes, class_names


# Store predictions and metrics from all folds
all_preds = []
all_targets = []
metrics_per_fold = []

print(f"Evaluating {KFOLD_SPLITS} fold models on test set")

for fold in range(KFOLD_SPLITS):
    print(f"\nEvaluating Fold {fold + 1}")
    
    fold_model_path = MODEL_PATH.replace(".pth", f"_fold{fold + 1}.pth")
    checkpoint = torch.load(fold_model_path, map_location=DEVICE)

    model = ConvNeXt_CBAM(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    fold_preds = []
    fold_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            fold_preds.extend(preds.cpu().numpy())
            fold_targets.extend(labels.cpu().numpy())

    all_preds.append(fold_preds)
    all_targets.append(fold_targets)
    
    # Compute metrics per fold
    acc = accuracy_score(fold_targets, fold_preds)
    precision = precision_score(fold_targets, fold_preds, average='macro')
    recall = recall_score(fold_targets, fold_preds, average='macro')
    f1 = f1_score(fold_targets, fold_preds, average='macro')
    
    print(f" Fold {fold + 1} Metrics:")
    print(f"   - Accuracy: {acc:.4f}")
    print(f"   - Precision: {precision:.4f}")
    print(f"   - Recall: {recall:.4f}")
    print(f"   - F1 Score: {f1:.4f}")
    
    metrics_per_fold.append({
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })

# Combine all predictions (Majority Voting across folds)
all_preds = np.array(all_preds)  # shape: (folds, samples)
all_targets = np.array(all_targets[0])  # All same ground truth

# Majority Voting across folds
majority_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=all_preds)

#  Final Metrics
print(f"\nClassification Report for {MODEL_NAME} (Majority Vote from {KFOLD_SPLITS} folds):\n")
print(classification_report(all_targets, majority_preds, target_names=class_names, digits=4))

# Confusion Matrix
conf_matrix = confusion_matrix(all_targets, majority_preds)
print("Confusion Matrix:\n", conf_matrix)

#  Overall Average Metrics
avg_metrics = {
    'accuracy': np.mean([m['accuracy'] for m in metrics_per_fold]),
    'precision': np.mean([m['precision'] for m in metrics_per_fold]),
    'recall': np.mean([m['recall'] for m in metrics_per_fold]),
    'f1': np.mean([m['f1'] for m in metrics_per_fold])
}

print("\n Average Metrics Across All Folds:")
print(f"   - Accuracy: {avg_metrics['accuracy']:.4f}")
print(f"   - Precision: {avg_metrics['precision']:.4f}")
print(f"   - Recall: {avg_metrics['recall']:.4f}")
print(f"   - F1 Score: {avg_metrics['f1']:.4f}")