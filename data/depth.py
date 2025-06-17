from transformers import pipeline
from PIL import Image
import numpy as np
import os
import shutil

# === CONFIGURATION ===
input_root_dir = "./Citrus_splited"
output_root_dir = input_root_dir + "_depth_AUG"
threshold = 0.3  # Background depth threshold
batch_size = 1  # Set to 1 to handle variable-sized images
k = 20  # Mask smoothing factor
subsets = ['train', 'validation', 'test']

# Create output root directory if it doesn't exist
os.makedirs(output_root_dir, exist_ok=True)

# Load depth estimation pipeline
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf", batch_size=batch_size, use_fast=True)

# === PROCESS DATA ===
for subset in subsets:
    input_subset_dir = os.path.join(input_root_dir, subset)
    output_subset_dir = os.path.join(output_root_dir, subset)
    os.makedirs(output_subset_dir, exist_ok=True)
    
    for class_dir in os.listdir(input_subset_dir):
        input_class_path = os.path.join(input_subset_dir, class_dir)
        output_class_path = os.path.join(output_subset_dir, class_dir)
        os.makedirs(output_class_path, exist_ok=True)
        
        # Get list of image files path
        # Get image files
        image_files = [f for f in os.listdir(input_class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for img_file in image_files:
            src = os.path.join(input_class_path, img_file)
            dst = os.path.join(output_class_path, img_file)

            # Copy original image
            shutil.copy2(src, dst)

            # Apply depth-based suppression only for training set
            if subset != 'train':
                continue

            # === DEPTH SUPPRESSION ===
            img = Image.open(src).convert("RGB")
            original_size = img.size
            depth_map = pipe(img)["depth"]

            # Resize and normalize depth
            depth_array = np.array(depth_map.resize(original_size, Image.Resampling.LANCZOS)).astype(np.float32)
            d_min, d_max = depth_array.min(), depth_array.max()
            depth_norm = (depth_array - d_min) / (d_max - d_min) if d_max - d_min > 1e-8 else np.zeros_like(depth_array)

            # Smooth binary mask (closer = 1, farther = 0)
            smooth_mask = 1.0 - (1 / (1 + np.exp(k * (depth_norm - threshold))))
            mask_rgb = np.stack([smooth_mask] * 3, axis=-1)

            # Apply mask to image
            img_np = np.array(img).astype(np.float32) / 255.0
            masked_img = (img_np * mask_rgb * 255).astype(np.uint8)

            # Save depth-suppressed image
            suppressed_name = f"{os.path.splitext(img_file)[0]}_depth_suppressed{os.path.splitext(img_file)[1]}"
            Image.fromarray(masked_img).save(os.path.join(output_class_path, suppressed_name))
            print(f"Saved: {suppressed_name} (size={original_size})")
