import torch
import torch.optim as optim
import torch.nn as nn
import copy
import time

from load_data import train_loader, val_loader, num_classes
from models.convnext_cbam import ConvNeXt_CBAM
from utils import plot_training_metrics
from config import DEVICE, MODEL_NAME, MODEL_SAVE_PATH, LEARNING_RATE, EPOCHS, PATIENCE

# For clear start logs
print(f"✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔[NEW LOG: {MODEL_NAME}]✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔✔")

# Clear CUDA cache to free GPU memory
torch.cuda.empty_cache()

# Initialize the model
model = ConvNeXt_CBAM(num_classes)
model.to(DEVICE)

# Define the loss function with label smoothing to prevent model too confident
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Define the AdamW optimizer with weight decay
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

# Define the learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)


# Early stopping parameters
best_val_loss = float("inf")
early_stopping_counter = 0
best_model_wts = copy.deepcopy(model.state_dict())

# Lists to store training and validation metrics for plot
train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

# For start training time
start_time = time.time()

# ===============
# Training Loop =
# ===============
for epoch in range(EPOCHS):
    # Track time for each epoch
    epoch_start = time.time()  
    
    model.train() # Set model to training mode
    
    # Init running loss and accuracy counters
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for images, labels in train_loader:
        # Move data to the device
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # Zero out parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass and use AdamW optimize
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        # Compute training accuracy
        predicted = torch.argmax(outputs, dim=1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    # Calculate average training loss and accuracy for the epoch
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    train_acc = correct_train / total_train
    train_accuracies.append(train_acc)

    # Set model to evaluation mode
    model.eval()
    
    val_loss = 0.0
    correct = 0
    total = 0
    
    # Disable gradient computation for validation
    with torch.no_grad():
        for images, labels in val_loader:
            # Move data to the device
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Compute validation accuracy
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
    # Calculate average validation loss and accuracy for the epoch
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    val_acc = correct / total
    val_accuracies.append(val_acc)
    
    # End timing for this epoch
    epoch_end = time.time()
    epoch_duration = epoch_end - epoch_start

    # Print epoch summary log
    print(f"Epoch {epoch+1}/{EPOCHS} | Time: {epoch_duration:.2f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} ")
    
    # Check for best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        print(f"Saving best model: {val_acc:.4f}")
        early_stopping_counter = 0  # Reset early stopping counter
    else:
        early_stopping_counter += 1

    # Step the learning rate scheduler
    scheduler.step()

    # Early stopping
    if early_stopping_counter >= PATIENCE:
        print("Early stopping triggered.")
        break

# Load the best model weights
model.load_state_dict(best_model_wts)

# Calculate total training time
total_time = time.time() - start_time
print(f"Training complete in {total_time:.2f} seconds.")

# Save the best model and optimizer states
torch.save({'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()}, MODEL_SAVE_PATH)
print(f"Model saved")

# Plot training metrics
plot_training_metrics(train_losses, val_losses, train_accuracies, val_accuracies)
