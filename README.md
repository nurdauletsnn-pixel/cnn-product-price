# 🛒 CNN-Based Product Recognition & Price Estimation System

> Deep Learning Course — Project 1 | SDU, 2026

---

## 📌 Overview

A web application that uses **5 CNN architectures** to recognize grocery products from photos and estimate their total cost. Supports **TTA (Test-Time Augmentation)** and **Ensemble inference** for improved accuracy.

**Best model:** ResNet-50 — **98.3% Top-1 accuracy**, MAE = 15.7 тг

---

## 🏗️ Project Structure

```
project_dl/
├── photos/                        ← Original dataset (16 categories, 419 images)
├── dataset/                       ← Augmented splits (train/val/test)
│   └── class_names.json
├── models/                        ← Trained weights (.pth files)
│   ├── alexnet/best_model.pth
│   ├── vgg16/best_model.pth
│   ├── googlenet/best_model.pth
│   ├── resnet50/best_model.pth
│   ├── efficientnet_b0/best_model.pth
│   └── all_metrics.json
├── price_database.json            ← Product → price mapping (тг)
├── step1_prepare.py               ← Dataset augmentation & splits
├── step2_train.py                 ← Train all 5 CNN models
├── step3_app.py                   ← Streamlit app (basic)
├── step4_improve.py               ← TTA + Ensemble benchmark test
├── step5_app_v2.py                ← Streamlit app v2 (TTA + Ensemble) ⭐
└── requirements.txt
```

---

## 🧠 Models & Results

| Model | Top-1 | Top-5 | MAE (тг) | RMSE (тг) | Size | Inference |
|-------|-------|-------|----------|-----------|------|-----------|
| **ResNet-50** ⭐ | **98.3%** | **99.7%** | **15.7** | **159.7** | 94 MB | 12.6 ms |
| VGG-16 | 97.9% | 99.3% | 76.2 | 709.6 | 537 MB | 22.9 ms |
| GoogLeNet | 95.5% | 99.0% | 74.8 | 600.8 | 23 MB | 8.2 ms |
| AlexNet | 96.2% | 98.6% | 70.0 | 534.6 | 228 MB | 1.6 ms |
| EfficientNet-B0 | 96.2% | 98.6% | 85.0 | 713.8 | 16 MB | 8.8 ms |

---

## 📦 Dataset

**16 grocery product categories** (Kazakhstan market):

| # | Category | # | Category |
|---|----------|---|----------|
| 1 | Шоколадные батончики | 9 | Чипсы |
| 2 | Зубная паста | 10 | Масло |
| 3 | Сок в тетрапакете | 11 | Макароны |
| 4 | Газировка в ж/б | 12 | Лапша б/п |
| 5 | Вода | 13 | Кофе |
| 6 | Хлеб | 14 | Яшкино |
| 7 | Молоко | 15 | Детское питание |
| 8 | Пакетированный чай | 16 | Майонез |

- **419** original images → **~5000+** after augmentation
- Split: **70%** train / **15%** val / **15%** test

---

## ⚙️ Installation & Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/cnn-product-price.git
cd cnn-product-price

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare dataset (augmentation + splits)
python step1_prepare.py

# 4. Train all 5 models (~2-3 hours on Apple M2)
python step2_train.py

# 5. Launch the web app
streamlit run step5_app_v2.py
```

---

## 🚀 Inference Modes

Switchable in the sidebar during demo:

| Mode | Speed | Quality |
|------|-------|---------|
| 🚀 Single Model | ~30 ms | Baseline |
| 🎯 Single + TTA | ~250 ms | +5-10% confidence |
| 🏆 Ensemble + TTA | ~1.2 s | Best accuracy |

**Ensemble weights:** ResNet50 (40%) · VGG16 (25%) · GoogLeNet (15%) · AlexNet (10%) · EfficientNet (10%)

---

## 🛠️ Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingLR |
| Batch size | 32 |
| Max epochs | 25 |
| Early stopping | patience = 6 |
| Image size | 224×224 |
| Device | Apple M2 (MPS) |

**Anti-overfitting:** transfer learning · backbone freezing · label smoothing (0.1) · dropout · gradient clipping · class-weighted loss · 17-type augmentation

---

## 📋 Requirements

```
torch>=2.0.0
torchvision>=0.15.0
streamlit>=1.28.0
plotly>=5.0.0
pillow>=9.0.0
numpy>=1.24.0
pandas>=2.0.0
pillow-heif>=0.13.0
```

---

## 📁 Model Weights

Weights are stored in `models/` (not pushed to GitHub due to size).

**Download from Google Drive:** `[add your link here]`

Or retrain:
```bash
python step2_train.py
```

---

## 👤 Author

**Nurdaulet** — Deep Learning Course, 2025

---

## 📄 License

MIT
