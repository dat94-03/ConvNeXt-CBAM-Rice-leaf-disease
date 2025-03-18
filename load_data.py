import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import config as ENV

torch.manual_seed(42)#set fix random for testing

#Transforms with augmentation
train_transform = transforms.Compose([
    # transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),#scale image from 0-255 to 0-1.0
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load dataset
full_train_dataset = datasets.ImageFolder(root=ENV.TRAIN_DATA_PATH, transform=train_transform)
test_dataset = datasets.ImageFolder(root=ENV.TEST_DATA_PATH, transform=test_transform)


train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=ENV.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=ENV.BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

num_classes = len(full_train_dataset.classes)
