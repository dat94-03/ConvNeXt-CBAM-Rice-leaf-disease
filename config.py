import torch

# Paths
TRAIN_DATA_PATH = "./data/dataset1/train"
TEST_DATA_PATH = "./data/dataset1/val"
MODEL_NAME='convNext+1cbam+dataset1'
MODEL_PATH = f"./models/{MODEL_NAME}.pth"
# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
EPOCHS = 100

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



