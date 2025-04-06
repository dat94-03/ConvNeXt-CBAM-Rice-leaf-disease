import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
from torchvision import datasets
from models.convnext_cbam import ConvNeXt_CBAM
from utils_kfold import plot_training_metrics
from config import DEVICE, MODEL_NAME, MODEL_PATH, LEARNING_RATE, EPOCHS, KFOLD_SPLITS, TRAIN_DATA_PATH
from load_data_kfold import get_kfold_loaders, num_classes 
from sklearn.utils.class_weight import compute_class_weight

torch.cuda.empty_cache()
print(f" Starting training with {KFOLD_SPLITS}-Fold Cross-Validatieon for model: {MODEL_NAME}")

# Get K-fold data loaders
folds = get_kfold_loaders(KFOLD_SPLITS)

# Initialize tracking for all folds
all_fold_train_losses = []
all_fold_val_losses = []
all_fold_train_accs = []
all_fold_val_accs = []

# Train on each fold
for fold, (train_loader, val_loader) in enumerate(folds):
    print(f"\n Fold {fold + 1}/{KFOLD_SPLITS}")
    # Compute class weights from entire dataset
    targets = [sample[1] for sample in datasets.ImageFolder(root=TRAIN_DATA_PATH)]
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(targets), y=targets)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    model = ConvNeXt_CBAM(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    scaler = torch.amp.GradScaler(device='cuda') if DEVICE.type == "cuda" else None

    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    early_stopping_counter = 0
    patience = 10

    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f" Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            early_stopping_counter = 0
            print(f" Best model updated (val acc: {val_acc:.4f})")
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print(" Early stopping.")
            break

    # Save best model for current fold
    fold_model_path = MODEL_PATH.replace(".pth", f"_fold{fold+1}.pth")
    model.load_state_dict(best_model_wts)
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, fold_model_path)
    print(f" Fold {fold+1} complete. Saved: {fold_model_path}")

    all_fold_train_losses.append(train_losses)
    all_fold_val_losses.append(val_losses)
    all_fold_train_accs.append(train_accuracies)
    all_fold_val_accs.append(val_accuracies)

# Plot results across all folds
plot_training_metrics(all_fold_train_losses, all_fold_val_losses, all_fold_train_accs, all_fold_val_accs, kfold=KFOLD_SPLITS)
print(" All folds completed.")
