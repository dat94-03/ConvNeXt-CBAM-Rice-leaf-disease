#🌿 Plant Leaf Disease Classification

This project focuses on classifying potato leaf diseases using a deep learning model based on **ConvNeXt_CBAM**, which integrates Convolutional Block Attention Modules (CBAM) into a pre-trained ConvNeXt base architecture. It supports standard training and evaluation, with options for visualizing model performance through confusion matrices, training metrics, and misclassified images. 


## 🚀 Getting Started

### 1. Install Dependencies

Ensure all required packages are installed. You can set up a Python environment and install dependencies using:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare the Dataset

Place your dataset in the `data/` folder. The expected structure is:

```
data/Potato_splited/
├── train/
│   ├── early_blight/
│   ├── late_blight/
│   └── healthy/
├── validation/
│   ├── early_blight/
│   ├── late_blight/
│   └── healthy/
└── test/
    ├── early_blight/
    ├── late_blight/
    └── healthy/
```

### 3. Update Configuration

Edit `config.py` to set the correct paths and hyperparameters. Example:

```python
# config.py
DATA_PATH = "./data/Potato_splited/"
TEST_DATA_PATH = "./data/Potato_splited/test/"
MODEL_PATH = "./output/model.pth"
MODEL_NAME = "convnext_cbam"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
```

### 4. Train the Model

Run the training script to train the ConvNeXt_CBAM model:

```bash
bash run.sh
```

This script trains the model using the dataset in `data/Potato_splited/`, saves the trained weights to `output/model.pth`, and logs training progress to `output.log`.

---

## 🧪 Testing the Model

Evaluate the trained model on the test dataset:

```bash
bash test.sh
```

This script:
- Loads the trained model from `MODEL_PATH`.
- Evaluates performance using metrics like accuracy, precision, recall, and F1-score.
- Generates visualizations (e.g., confusion matrix, training metrics plots, grad_am) saved in `output/test/{MODEL_NAME}/`.
- Appends evaluation logs to `output.log`.

