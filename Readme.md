# Rice Disease Classification with ConvNeXt & CBAM

This project is a deep learning-based classification model for rice disease detection using a custom ConvNeXt model with CBAM (Convolutional Block Attention Module).

## 📌 Features
- **ConvNeXt Architecture**: Enhanced with CBAM for better feature extraction.
- **PyTorch-based**: Efficient deep learning implementation.
- **Custom Dataset Support**: Works with rice leaf images.
- **Confusion Matrix & Classification Report**: For performance evaluation.

## 🚀 Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/rice-disease-classification.git
   cd rice-disease-classification
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## 📂 Project Structure
```
.
├── train.py        # Model training script
├── test.py         # Model evaluation script
├── model.py        # ConvNeXt_CBAM model definition
├── utils.py        # Helper functions (evaluation, visualization, etc.)
├── config.py       # Configuration settings (paths, hyperparameters, etc.)
├── data/           # Dataset folder (train & test images)
├── models/         # Saved model weights
└── README.md       # Documentation
```

## 🏋️‍♂️ Training
Run the training script:
```sh
python train.py
```

## 📊 Evaluation
Run the test script:
```sh
python test.py
```

## 📌 Configuration
Modify `config.py` to change dataset paths, batch size, learning rate, etc.

## 📞 Contact
For questions, create an issue in the repository or email me at `your-email@example.com`.

