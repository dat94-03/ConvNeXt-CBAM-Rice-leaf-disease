import torch
import matplotlib.pyplot as plt
import numpy as np
import os

from torchvision import transforms
from config import MODEL_SAVE_PATH, DEVICE,MODEL_NAME, BATCH_SIZE
from models.convnext_cbam import ConvNeXt_CBAM
from load_data import num_classes, test_loader, class_names

# Load the model
model = ConvNeXt_CBAM(num_classes).to(DEVICE)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE)['model_state_dict'])
model.eval()


# Lists to store misclassified images and labels
misclassified_images, misclassified_labels, predicted_labels = [], [], []

# Function to convert tensor to PIL image
to_pil = transforms.ToPILImage()

# Evaluate the model to find misclassified images
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        
        # Identify misclassified images
        misclassified_mask = preds != labels
        for i in range(len(images)):
            if misclassified_mask[i]:
                misclassified_images.append(images[i].cpu())
                misclassified_labels.append(class_names[labels[i].cpu().item()])
                predicted_labels.append(class_names[preds[i].cpu().item()])


# Define the output directory and filename
output_dir = f"output/test/{MODEL_NAME}"
output_file_path = os.path.join(output_dir, "misclassified_images.png")

# Set up plot param
num_images = len(misclassified_images)
cols = 5
rows = int(np.ceil(num_images / cols))
fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
axes = axes.flatten()


# Save misclassified images to a file instead of displaying them

for i in range(num_images):
    image = to_pil(misclassified_images[i])
    axes[i].imshow(image)
    axes[i].axis("off")
    axes[i].set_title(f"True: {misclassified_labels[i]}\nPred: {predicted_labels[i]}")

# Hide any unused subplots
for i in range(num_images, len(axes)):
    axes[i].axis("off")

# Save the figure to a file
plt.savefig(output_file_path, bbox_inches="tight", dpi=300)
plt.close()
print(f"Saved misclassified images")
