import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, datasets, models
from config import MODEL_PATH, DEVICE, TEST_DATA_PATH,MODEL_NAME
from models.convnext_cbam import ConvNeXt_CBAM
from load_data import num_classes


# Load the model (modify according to your architecture)
model = ConvNeXt_CBAM(num_classes).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)['model_state_dict'])
model.eval()

# Load dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

test_dataset = datasets.ImageFolder(TEST_DATA_PATH, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
class_names = test_dataset.classes

# Find misclassified images
misclassified_images, misclassified_labels, predicted_labels = [], [], []
to_pil = transforms.ToPILImage()

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        # Identify misclassified images
        misclassified_mask = preds != labels
        for i in range(len(images)):
            if misclassified_mask[i]:
                misclassified_images.append(images[i].cpu())
                misclassified_labels.append(class_names[labels[i].cpu().item()])
                predicted_labels.append(class_names[preds[i].cpu().item()])

# Plot misclassified images
num_images = len(misclassified_images)
cols = 5
rows = int(np.ceil(num_images / cols))
fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
axes = axes.flatten()

# Save misclassified images to a file instead of displaying them
output_filename = f"output/{MODEL_NAME}/misclassified_images.png"
fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
axes = axes.flatten()

for i in range(num_images):
    image = to_pil(misclassified_images[i])
    axes[i].imshow(image)
    axes[i].axis("off")
    axes[i].set_title(f"True: {misclassified_labels[i]}\nPred: {predicted_labels[i]}")

# Hide any unused subplots
for i in range(num_images, len(axes)):
    axes[i].axis("off")

# Save the figure to a file
plt.savefig(output_filename, bbox_inches="tight", dpi=300)
plt.close()

print(f"Saved misclassified images to {output_filename}")

