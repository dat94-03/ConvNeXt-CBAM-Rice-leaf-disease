import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from config import TEST_DATA_PATH, BATCH_SIZE, DEVICE, MODEL_PATH,MODEL_NAME
from models.convnext_cbam import ConvNeXt_CBAM
from utils import evaluate_model
from load_data import test_loader, num_classes, class_names


model = ConvNeXt_CBAM(num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Optional: Only if continuing training
model.to(DEVICE)

# Evaluate the model
print(f"Evaluate model:{MODEL_NAME}")
evaluate_model(model, test_loader, class_names)