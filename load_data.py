import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH, BATCH_SIZE

# Set random seed for reproducibility
torch.manual_seed(42)

# Augmentation for training set
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
])

# With validation and test set only resize and convert images to PyTorch tensors
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load datasets from folder with subfolder represent classes
train_dataset = datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=train_transform)
val_dataset = datasets.ImageFolder(root=VAL_DATA_PATH, transform=val_test_transform)
test_dataset = datasets.ImageFolder(root=TEST_DATA_PATH, transform=val_test_transform)

# Create DataLoaders for batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define number of class and classname for init model output
num_classes = len(train_dataset.classes)
class_names = train_dataset.classes
