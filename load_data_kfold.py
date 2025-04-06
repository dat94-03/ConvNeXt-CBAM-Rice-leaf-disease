import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
from config import TRAIN_DATA_PATH, TEST_DATA_PATH, BATCH_SIZE, KFOLD_SPLITS

# torch.manual_seed(42)#set fix random for testing

# Data augmentations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load datasets
full_train_dataset = datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=train_transform)
test_dataset = datasets.ImageFolder(root=TEST_DATA_PATH, transform=val_test_transform)

num_classes = len(full_train_dataset.classes)
class_names = full_train_dataset.classes

# K-Fold loader with StratifiedKFold
def get_kfold_loaders(k):
    targets = full_train_dataset.targets
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    folds = []
    fold_distributions = defaultdict(lambda: np.zeros(num_classes))

    for fold, (train_idx, val_idx) in enumerate(skf.split(full_train_dataset.samples, targets)):
        train_subset = Subset(full_train_dataset, train_idx)
        val_subset = Subset(full_train_dataset, val_idx)

        # Apply val/test transform to val set
        # val_subset.dataset.transform = val_test_transform
        val_dataset = datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=val_test_transform)
        val_subset = Subset(val_dataset, val_idx)


        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
        folds.append((train_loader, val_loader))

        # Count class distribution in validation set
        val_labels = [targets[i] for i in val_idx]
        val_counter = Counter(val_labels)
        for class_idx in range(num_classes):
            fold_distributions[fold][class_idx] = val_counter.get(class_idx, 0)

    save_fold_class_distribution(fold_distributions)
    return folds

# Save per-fold class distribution plot
def save_fold_class_distribution(distributions):
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    x = np.arange(num_classes)
    width = 0.15

    for fold, counts in distributions.items():
        plt.bar(x + width * fold, counts, width=width, label=f"Fold {fold + 1}")

    plt.xticks(x + width * (KFOLD_SPLITS / 2), class_names, rotation=45)
    plt.ylabel("Samples per Class")
    plt.title("Class Distribution in Validation Set per Fold")
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "fold_class_distribution.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"✅ Saved fold class distribution plot to: {plot_path}")

# Standard test loader
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
