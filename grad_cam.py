import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from config import TEST_DATA_PATH, BATCH_SIZE, DEVICE, MODEL_PATH
import matplotlib.pyplot as plt
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, class_idx):
        output = self.model(input_tensor)
        self.model.zero_grad()
        
        class_score = output[0, class_idx]
        class_score.backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze().detach().cpu().numpy()
        cam = np.maximum(cam, 0)

        cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Fixed size for consistency
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
    from models.convnext_cbam import ConvNeXt_CBAM
    from load_data import test_dataset, num_classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNeXt_CBAM(num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)['model_state_dict'])
    model.eval()

    target_layer = model.model.stages[3]
    grad_cam = GradCAM(model, target_layer)

    os.makedirs("output/grad_cam", exist_ok=True)
    random_indices = np.random.choice(len(test_dataset), 40, replace=False)

    for batch in range(5):  # 2 batches of 16 images
        images = []
        for i in range(batch * 8, (batch + 1) * 8):  # 8 original + 8 overlayed
            image_path = test_dataset.samples[random_indices[i]][0]
            input_tensor, original_image = preprocess_image(image_path)
            input_tensor = input_tensor.to(device)

            output = model(input_tensor)
            predicted_class = torch.argmax(output, 1).item()

            cam = grad_cam.generate_cam(input_tensor, predicted_class)
            overlayed_image = overlay_heatmap(original_image, cam)

            # Append original and overlayed side by side
            images.append(original_image.resize((224, 224)))
            images.append(overlayed_image)

        grid = create_grid(images, grid_size=(4, 4))
        grid.save(f"output/grad_cam/grid_batch_{batch+1}.jpg")

    print("Saved 5 merged images in the output/grad_cam folder!")
