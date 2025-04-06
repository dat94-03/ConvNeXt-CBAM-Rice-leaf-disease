from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH, BATCH_SIZE

# torch.manual_seed(42)#set fix random for testing

#augmentation data
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
])
#with validation and test set no apply augmentation, only resize and normalized
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load dataset and apply augmentation
train_dataset = datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=train_transform)
val_dataset = datasets.ImageFolder(root=VAL_DATA_PATH, transform=val_test_transform)
test_dataset = datasets.ImageFolder(root=TEST_DATA_PATH, transform=val_test_transform)

#Only shuffle train dataset
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

#define number of class for init model output
num_classes = len(train_dataset.classes)
class_names = train_dataset.classes
