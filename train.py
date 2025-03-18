import torch
import torch.optim as optim
import torch.nn as nn
import copy
import time
from torch.utils.data import DataLoader
from load_data import train_loader, val_loader, num_classes
from models.convnext_cbam import ConvNeXt_CBAM
from utils import plot_training_metrics
import config as ENV

torch.cuda.empty_cache()

# Initialize Model
device = ENV.DEVICE
model = ConvNeXt_CBAM(num_classes)
model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=ENV.LEARNING_RATE, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# 🔥 Gradient Clipping Function
def clip_gradients(model, clip_value=1.0):
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

# 🔥 Enable Mixed Precision Training (Latest Syntax)
scaler = torch.amp.GradScaler(device='cuda') if device.type == "cuda" else None

# Early Stopping Parameters
patience = 10
best_val_loss = float("inf")
early_stopping_counter = 0
best_model_wts = copy.deepcopy(model.state_dict())

# Training Loop
num_epochs = ENV.EPOCHS
train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

start_time = time.time()  # Start timing

for epoch in range(num_epochs):
    epoch_start = time.time()  # Track time for each epoch
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    train_acc = correct_train / total_train
    train_accuracies.append(train_acc)
    
    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    val_acc = correct / total
    val_accuracies.append(val_acc)

    epoch_end = time.time()  # End timing for this epoch
    epoch_duration = epoch_end - epoch_start

    print(f"🕒 Epoch {epoch+1}/{num_epochs} | Time: {epoch_duration:.2f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Check for best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        print(f"💾 Saving best val acc: {val_acc:.4f}")
        early_stopping_counter = 0  # Reset counter
    else:
        early_stopping_counter += 1

    # Reduce learning rate if validation loss plateaus
    scheduler.step()

    # Early stopping
    if early_stopping_counter >= patience:
        print("⏹️ Early stopping triggered.")
        break

# Load Best Model and Save It
model.load_state_dict(best_model_wts)
total_time = time.time() - start_time  # Total training time
print(f"✅ Training complete in {total_time:.2f} seconds. Best model loaded.")

torch.save({'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()}, ENV.MODEL_PATH)
print(f"📁 Model saved to {ENV.MODEL_PATH}")

# Plot training metrics
plot_training_metrics(train_losses, val_losses, train_accuracies, val_accuracies)
