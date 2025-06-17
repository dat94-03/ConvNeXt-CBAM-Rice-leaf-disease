import torch
from config import DEVICE, MODEL_SAVE_PATH, MODEL_NAME
from models.convnext_cbam import ConvNeXt_CBAM
from utils import evaluate_model
from load_data import test_loader, num_classes, class_names

# Initialize the model with the number of classes
model = ConvNeXt_CBAM(num_classes)

# Load the saved model state dictionary 'model_state_dict' saved in train.py
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE)['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Optional: Only if continuing training

model.to(DEVICE)

# Evaluate the model
print(f"Evaluate model:{MODEL_NAME}")
evaluate_model(model, test_loader, class_names)