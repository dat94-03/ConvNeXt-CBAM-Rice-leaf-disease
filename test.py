import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from config import TEST_DATA_PATH, BATCH_SIZE, DEVICE, MODEL_PATH,MODEL_NAME
from models.convnext_cbam import ConvNeXt_CBAM
from utils import evaluate_model

# Load test dataset
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_dataset = datasets.ImageFolder(root=TEST_DATA_PATH, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load model
num_classes = len(test_dataset.classes)
model = ConvNeXt_CBAM(num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Optional: Only if continuing training
model.to(DEVICE)

# Evaluate the model
print(f"Evaluate model:{MODEL_NAME}")
evaluate_model(model, test_loader, test_dataset.classes)