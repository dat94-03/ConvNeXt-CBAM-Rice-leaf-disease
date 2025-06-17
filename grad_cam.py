import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from config import DEVICE, MODEL_SAVE_PATH, MODEL_NAME
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
import random
from models.convnext_cbam import ConvNeXt_CBAM
from load_data import test_dataset, num_classes

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        """Register forward and backward hooks on the target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, class_idx):
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        self.model.zero_grad()

        # Backward pass for the specified class
        class_score = output[0, class_idx]
        class_score.backward()

        # Compute CAM
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze().detach().cpu().numpy()
        cam = np.maximum(cam, 0) # ReLU

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # Avoid divide-by-zero
        return cam

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0), image

def overlay_heatmap(original_img, cam, alpha=0.5):
    cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlayed_img = cv2.addWeighted(np.array(original_img.resize((224, 224))), 1 - alpha, heatmap, alpha, 0)
    return Image.fromarray(overlayed_img)

def create_grid(images, grid_size=(4, 4), image_size=(224, 224), background_color=(255, 255, 255)):
    grid_w, grid_h = grid_size[0] * image_size[0], grid_size[1] * image_size[1]
    grid_image = Image.new("RGB", (grid_w, grid_h), background_color)

    for i, img in enumerate(images):
        x = (i % grid_size[0]) * image_size[0]
        y = (i // grid_size[0]) * image_size[1]
        grid_image.paste(img, (x, y))

    return grid_image

if __name__ == "__main__":
    # Load model
    model = ConvNeXt_CBAM(num_classes).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE)['model_state_dict'])
    model.eval()
    
    # Initialize GradCAM
    target_layer = model.model.stages[3]  # Final ConvNeXt stage (1024 channels)
    grad_cam = GradCAM(model, target_layer)

    # Create output directory
    output_dir = f"output/test/{MODEL_NAME}/grad_cam"
    os.makedirs(output_dir, exist_ok=True)

    # Group samples by class
    class_to_indices = defaultdict(list)
    for idx, (path, label) in enumerate(test_dataset.samples):
        class_to_indices[label].append(idx)

    # Select equal samples from each class (total 64)
    num_classes = len(class_to_indices)
    samples_per_class = 64 // num_classes
    fixed_indices = []
    
    for label, indices in class_to_indices.items():
        selected = indices[:min(samples_per_class, len(indices))] 
        fixed_indices.extend(selected)

    while len(fixed_indices) < 64:
        for label, indices in class_to_indices.items():
            if len(fixed_indices) >= 64:
                break
            extra = list(set(indices) - set(fixed_indices))
            if extra:
                fixed_indices.append(random.choice(extra))


    for batch in range(8):  # 8 batches of 8 image pairs = 16 images/grid
        images = []
        for i in range(batch * 8, (batch + 1) * 8):
            image_path = test_dataset.samples[fixed_indices[i]][0]
            input_tensor, original_image = preprocess_image(image_path)
            input_tensor = input_tensor.to(DEVICE)

            output = model(input_tensor)
            predicted_class = torch.argmax(output, 1).item()

            cam = grad_cam.generate_cam(input_tensor, predicted_class)
            overlayed_image = overlay_heatmap(original_image, cam)

            # Append original and overlayed side by side
            images.append(original_image.resize((224, 224)))
            images.append(overlayed_image)

        grid = create_grid(images, grid_size=(4, 4))
        grid.save(f"output/test/{MODEL_NAME}/grad_cam/grid_batch_{batch + 1}.jpg")

    print(f"Saved gradCAM images")
