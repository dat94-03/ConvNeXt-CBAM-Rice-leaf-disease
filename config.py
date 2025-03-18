import torch

# Paths
TRAIN_DATA_PATH = "./data/dataset1/train"
TEST_DATA_PATH = "./data/dataset1/val"
MODEL_PATH = "./models/convnext_cbam.pth"

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
EPOCHS = 100

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#pkill -f train.py nohup python3 -u train.py >> output.log 2>&1 && nohup python3 -u test.py >> output.log 2>&1 & tail -f output.log


