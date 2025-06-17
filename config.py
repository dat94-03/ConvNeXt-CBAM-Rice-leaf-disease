import torch

# Dataset Configuration
DATA_SET = 'Potato_splited'
DATA_PATH=f"./data/{DATA_SET}"
TRAIN_DATA_PATH = f"{DATA_PATH}/train"
VAL_DATA_PATH = f"{DATA_PATH}/validation"
TEST_DATA_PATH = f"{DATA_PATH}/test"

MODEL_NAME=f'ConvNeXt+CBAM+{DATA_SET}'
MODEL_SAVE_PATH = f"./output/models/{MODEL_NAME}.pth" # Path to save/load the model

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
EPOCHS = 50
PATIENCE =10
IMAGE_SIZE =224

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


