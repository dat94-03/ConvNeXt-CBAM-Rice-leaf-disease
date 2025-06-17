import os
import shutil
import random
from pathlib import Path

def create_split_dirs(base_dir, class_names):
    for split in ['train', 'validation', 'test']:
        for class_name in class_names:
            Path(base_dir, split, class_name).mkdir(parents=True, exist_ok=True)

def split_dataset(input_dir, output_dir, train_ratio=0.7, val_ratio=0.2):
    
    class_dirs = [d for d in os.listdir(input_dir) if Path(input_dir, d).is_dir()]
    create_split_dirs(output_dir, class_dirs)

    for class_name in class_dirs:
        image_files = [f for f in os.listdir(Path(input_dir, class_name))
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(image_files)

        n = len(image_files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            'train': image_files[:n_train],
            'validation': image_files[n_train:n_train + n_val],
            'test': image_files[n_train + n_val:]
        }

        for split, files in splits.items():
            for fname in files:
                src = Path(input_dir, class_name, fname)
                dst = Path(output_dir, split, class_name, fname)
                shutil.copy2(src, dst)
            print(f"[{split}] {class_name}: {len(files)} images copied.")

def print_summary(output_dir):
    print("\nDataset split complete:")
    for split in ['train', 'validation', 'test']:
        split_dir = Path(output_dir, split)
        total = sum(len(list(Path(split_dir, cls).glob("*.jp*g")))
                    for cls in os.listdir(split_dir))
        print(f"{split.capitalize():<10}: {total} images")

if __name__ == "__main__":
    input_dir = "./Citrus"
    output_dir = f"{input_dir}_splited"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    split_dataset(input_dir, output_dir)
    print_summary(output_dir)
