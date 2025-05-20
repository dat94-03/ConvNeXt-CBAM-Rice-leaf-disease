import torch
import torch.nn as nn
import torch.optim as optim
import copy
import random
import time

from load_data import train_loader, val_loader, num_classes
from models.convnext_cbam import ConvNeXt_CBAM
from utils import plot_training_metrics
from config import DEVICE, MODEL_NAME, MODEL_PATH, LEARNING_RATE, EPOCHS, PATIENCE, SHOTS, WAYS, QUERIES

print(f"✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔[NEW FEW-SHOT LOG: {MODEL_NAME}]✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔")
torch.cuda.empty_cache()

# =========================
# Helper Functions
# =========================

def build_class_to_images(data_loader):
    """ Build mapping: class_id -> list of images """
    class_to_images = {}
    for images, labels in data_loader:
        for img, label in zip(images, labels):
            label = label.item()
            if label not in class_to_images:
                class_to_images[label] = []
            class_to_images[label].append(img)
    return class_to_images

def create_episode(class_to_images, shots, ways, queries=1):
    """ Create an episode """
    eligible_classes = [cls for cls, imgs in class_to_images.items() if len(imgs) >= (shots + queries)]
    if len(eligible_classes) < ways:
        raise ValueError(f"Not enough classes with enough samples. Have {len(eligible_classes)}, need {ways}.")

    selected_classes = random.sample(eligible_classes, ways)

    support_images = []
    support_labels = []
    query_images = []
    query_labels = []

    for idx, cls in enumerate(selected_classes):
        imgs = class_to_images[cls]
        sampled = random.sample(imgs, shots + queries)
        support_samples = sampled[:shots]
        query_samples = sampled[shots:]

        support_images.extend(support_samples)
        support_labels.extend([idx] * shots)
        query_images.extend(query_samples)
        query_labels.extend([idx] * queries)

    support_images = torch.stack(support_images)
    support_labels = torch.tensor(support_labels)
    query_images = torch.stack(query_images)
    query_labels = torch.tensor(query_labels)

    return support_images, support_labels, query_images, query_labels

# =========================
# Initialize Model
# =========================

device = DEVICE
model = ConvNeXt_CBAM(num_classes=WAYS)  # Model now outputs 'WAYS' classes
model.to(device)

# Loss, Optimizer, Scheduler
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# Early Stopping
patience = PATIENCE
best_val_loss = float("inf")
early_stopping_counter = 0
best_model_wts = copy.deepcopy(model.state_dict())

# Episode-based Few-shot Setup
class_to_images = build_class_to_images(train_loader)

# Metrics
train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

# =========================
# Few-shot Training Loop
# =========================

start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    # Train over multiple episodes in one epoch
    num_episodes_per_epoch = 100

    for _ in range(num_episodes_per_epoch):
        support_images, support_labels, query_images, query_labels = create_episode(
            class_to_images, shots=SHOTS, ways=WAYS, queries=QUERIES
        )

        support_images, support_labels = support_images.to(device), support_labels.to(device)
        query_images, query_labels = query_images.to(device), query_labels.to(device)

        # Combine support and query sets (Optional: some meta-learning needs this, but here we train on query only)
        model.zero_grad()
        outputs = model(query_images)
        loss = criterion(outputs, query_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct_train += (preds == query_labels).sum().item()
        total_train += query_labels.size(0)

    train_loss = running_loss / num_episodes_per_epoch
    train_acc = correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validation (optional few-shot episodes on validation set)
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        val_class_to_images = build_class_to_images(val_loader)
        num_val_episodes = 20

        for _ in range(num_val_episodes):
            try:
                support_images, support_labels, query_images, query_labels = create_episode(
                    val_class_to_images, shots=SHOTS, ways=WAYS, queries=QUERIES
                )
            except:
                continue  # Skip if not enough classes
            
            support_images, support_labels = support_images.to(device), support_labels.to(device)
            query_images, query_labels = query_images.to(device), query_labels.to(device)

            outputs = model(query_images)
            loss = criterion(outputs, query_labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == query_labels).sum().item()
            total += query_labels.size(0)

    if total > 0:
        val_loss /= num_val_episodes
        val_acc = correct / total
    else:
        val_loss = 0.0
        val_acc = 0.0

    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    epoch_end = time.time()
    epoch_duration = epoch_end - epoch_start

    print(f"Epoch {epoch+1}/{EPOCHS} | Time: {epoch_duration:.2f}s | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    # Check for best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        print(f"Saving best model at Epoch {epoch+1}")
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1

    scheduler.step()

    if early_stopping_counter >= patience:
        print("Early stopping triggered.")
        break

# =========================
# Save Best Model
# =========================

model.load_state_dict(best_model_wts)
total_time = time.time() - start_time
print(f"Training complete in {total_time:.2f} seconds.")

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# Plot training metrics
plot_training_metrics(train_losses, val_losses, train_accuracies, val_accuracies)
