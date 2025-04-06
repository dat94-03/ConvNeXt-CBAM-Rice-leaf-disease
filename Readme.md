Here's a clean and professional `README.md` tailored for your rice leaf disease classification project:

---

# 🌾 Rice Leaf Disease Classification

This project aims to classify rice leaf diseases using deep learning models. It supports standard training and K-Fold cross-validation training. After training, results including logs and prediction images will be saved for evaluation and visualization.

---

## 📁 Project Structure

```
project-root/
├── data/                 # Place your dataset here
├── config.py             # Update config path as needed
├── run.sh                # Script to run standard training
├── run_kfold.sh          # Script to run K-Fold training
├── test.sh               # Script to evaluate trained model
├── test_kfold.py         # Script to evaluate K-Fold trained model
├── output/               # Output images and logs are saved here
└── output.log            # Training and test logs
```

---

## 🚀 Getting Started

### 1. Place Dataset

Put your dataset inside the `data/` folder located at the project root.

### 2. Update Config

Open `config.py` and update the data path or any other configuration parameters to suit your setup.

```python
# Example
DATA_PATH=f"./data/{DATA_SET}"
```

### 3. Train the Model

**Standard Training**

```bash
bash run.sh
```

**K-Fold Training** (use this if your dataset only has `train/` and `test/` folders)

```bash
bash run_kfold.sh
```

All logs will be saved to `output.log`.

---

## 🧪 Testing the Model

After training is complete:

- **Standard model**:

```bash
bash test.sh
```

- **K-Fold model**:

```bash
python test_kfold.py
```

Logs will be appended to `output.log`. Prediction images will be saved in the `output/` folder.

---

## 📦 Output

After testing:

- Logs: `output.log`
- Prediction Results: `output/` folder (with visualization images)

---

## 📌 Notes

- Make sure all dependencies are installed (e.g., PyTorch, torchvision, matplotlib, etc.).
- The model and configuration are customizable for different datasets or experiment settings.

---

Let me know if you'd like to include installation steps, sample images, or model architecture details too!