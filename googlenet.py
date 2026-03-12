import os, json, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_DIR  = Path("dataset")
MODELS_DIR   = Path("models")
BATCH_SIZE   = 32
NUM_EPOCHS   = 25
LR           = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE     = 6
NUM_WORKERS  = 0
IMG_SIZE     = 224

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"⚙️  Device: {DEVICE}")

TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomRotation(20),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    transforms.RandomErasing(p=0.1),
])
VAL_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def get_logits(out):
    if hasattr(out, "logits"): return out.logits
    return out

def build_googlenet(nc):
    # Загружаем БЕЗ aux_logits параметра, затем отключаем вручную
    m = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
    m.aux_logits = False
    m.aux1 = None
    m.aux2 = None
    # Freeze все кроме последних 3 блоков + fc
    layers = list(m.named_children())
    for name, child in layers[:-3]:
        for p in child.parameters():
            p.requires_grad = False
    m.fc = nn.Linear(1024, nc)
    return m

def train_googlenet(class_names):
    nc         = len(class_names)
    model_name = "googlenet"

    train_ds = datasets.ImageFolder(DATASET_DIR/"train", transform=TRAIN_TF)
    val_ds   = datasets.ImageFolder(DATASET_DIR/"val",   transform=VAL_TF)
    test_ds  = datasets.ImageFolder(DATASET_DIR/"test",  transform=VAL_TF)

    targets = train_ds.targets
    counts  = np.bincount(targets)
    weights = 1.0 / (counts + 1e-6)
    weights = torch.tensor(weights / weights.sum() * nc, dtype=torch.float32).to(DEVICE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model     = build_googlenet(nc).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    save_dir = MODELS_DIR / model_name
    save_dir.mkdir(exist_ok=True)

    best_val_acc = 0.0
    patience_cnt = 0
    history      = {"tl":[], "vl":[], "ta":[], "va":[]}
    t0           = time.time()

    print(f"\n{'─'*58}")
    print(f"  GOOGLENET            | classes={nc} | device={DEVICE}")
    print(f"{'─'*58}")

    for epoch in range(1, NUM_EPOCHS+1):
        # train
        model.train()
        tl, tc, tt = 0., 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out  = get_logits(model(x))
            loss = criterion(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item()*x.size(0)
            tc += (out.argmax(1)==y).sum().item()
            tt += x.size(0)
        scheduler.step()

        # val
        model.eval()
        vl, vc, vt = 0., 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out  = get_logits(model(x))
                loss = criterion(out, y)
                vl += loss.item()*x.size(0)
                vc += (out.argmax(1)==y).sum().item()
                vt += x.size(0)

        ta, va = tc/tt, vc/vt
        history["tl"].append(tl/tt); history["vl"].append(vl/vt)
        history["ta"].append(ta);    history["va"].append(va)

        marker = " ✔" if va > best_val_acc else ""
        print(f"  ep {epoch:02d}/{NUM_EPOCHS}  "
              f"loss {tl/tt:.4f}/{vl/vt:.4f}  "
              f"acc  {ta:.4f}/{va:.4f}{marker}")

        if va > best_val_acc:
            best_val_acc = va
            torch.save({
                "model_name":  model_name,
                "state_dict":  model.state_dict(),
                "class_names": class_names,
                "val_acc":     va,
                "epoch":       epoch,
            }, save_dir / "best_model.pth")
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  ⏹  Early stop ep={epoch}")
                break

    train_sec = time.time() - t0

    # TEST
    ckpt = torch.load(save_dir/"best_model.pth", map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    preds, labels_list, probs_list = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            out  = get_logits(model(x.to(DEVICE)))
            prob = torch.softmax(out, 1).cpu().numpy()
            preds.extend(out.argmax(1).cpu().numpy())
            labels_list.extend(y.numpy())
            probs_list.extend(prob)

    preds      = np.array(preds)
    labels_arr = np.array(labels_list)
    probs_arr  = np.array(probs_list)

    top1 = (preds == labels_arr).mean()
    top5 = np.mean([labels_arr[i] in np.argsort(probs_arr[i])[-5:]
                    for i in range(len(labels_arr))])

    report = classification_report(labels_arr, preds,
                                   target_names=class_names,
                                   output_dict=True, zero_division=0)

    with open("price_database.json", encoding="utf-8") as f:
        pdb = json.load(f)

    def idx_to_price(idx):
        cn = class_names[idx]
        return pdb["categories"].get(cn, {}).get("avg_price", 0)

    true_prices = np.array([idx_to_price(l) for l in labels_arr], dtype=float)
    pred_prices = np.array([idx_to_price(p) for p in preds],       dtype=float)
    mae  = float(np.mean(np.abs(true_prices - pred_prices)))
    rmse = float(np.sqrt(np.mean((true_prices - pred_prices)**2)))

    dummy = torch.randn(1,3,IMG_SIZE,IMG_SIZE).to(DEVICE)
    with torch.no_grad():
        for _ in range(5): get_logits(model(dummy))
    ts = time.time()
    with torch.no_grad():
        for _ in range(50): get_logits(model(dummy))
    infer_ms = (time.time()-ts)/50*1000

    size_mb = os.path.getsize(save_dir/"best_model.pth")/1e6

    # Confusion matrix
    cm = confusion_matrix(labels_arr, preds)
    fig, ax = plt.subplots(figsize=(14,12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, annot_kws={"size":7})
    ax.set_title(f"googlenet — Confusion Matrix (Top-1={top1:.3f})")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.tight_layout()
    plt.savefig(save_dir/"confusion_matrix.png", dpi=130)
    plt.close()

    # Training curves
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    ep = range(1, len(history["tl"])+1)
    axes[0].plot(ep, history["tl"], label="Train")
    axes[0].plot(ep, history["vl"], label="Val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(ep, history["ta"], label="Train")
    axes[1].plot(ep, history["va"], label="Val")
    axes[1].set_title("Accuracy"); axes[1].legend()
    fig.suptitle("googlenet — Training Curves", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir/"training_curves.png", dpi=130)
    plt.close()

    metrics = {
        "model_name":            model_name,
        "top1_accuracy":         round(top1, 4),
        "top5_accuracy":         round(top5, 4),
        "best_val_acc":          round(best_val_acc, 4),
        "train_time_sec":        round(train_sec, 1),
        "inference_ms":          round(infer_ms, 2),
        "model_size_mb":         round(size_mb, 2),
        "price_mae":             round(mae, 1),
        "price_rmse":            round(rmse, 1),
        "classification_report": report,
    }
    with open(save_dir/"metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  ✅ Top-1={top1:.4f} Top-5={top5:.4f} "
          f"MAE={mae:.0f}тг  Size={size_mb:.1f}MB  Infer={infer_ms:.1f}ms")
    return metrics


def main():
    # Загружаем class_names
    with open(DATASET_DIR/"class_names.json", encoding="utf-8") as f:
        class_names = json.load(f)["class_names"]
    print(f"📋 Классов: {len(class_names)}")

    # Обучаем GoogLeNet
    googlenet_metrics = train_googlenet(class_names)

    # ── Читаем существующие метрики (AlexNet, VGG16, ResNet50, EfficientNet) ──
    metrics_path = MODELS_DIR / "all_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            all_metrics = json.load(f)
        # Удаляем старую запись googlenet если есть
        all_metrics = [m for m in all_metrics if m["model_name"] != "googlenet"]
    else:
        all_metrics = []

    # Добавляем googlenet и сортируем в правильном порядке
    all_metrics.append(googlenet_metrics)
    order = ["alexnet", "vgg16", "googlenet", "resnet50", "efficientnet_b0"]
    all_metrics.sort(key=lambda m: order.index(m["model_name"])
                     if m["model_name"] in order else 99)

    # Сохраняем обновлённые метрики
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Обновляем best model
    best = max(all_metrics, key=lambda x: x["top1_accuracy"])
    (MODELS_DIR/"best_model_name.txt").write_text(best["model_name"])

    # Сравнительный график
    names = [m["model_name"] for m in all_metrics]
    fig, axes = plt.subplots(1,3,figsize=(16,5))
    axes[0].bar(names, [m["top1_accuracy"] for m in all_metrics], color="#4C72B0", label="Top-1")
    axes[0].bar(names, [m["top5_accuracy"] for m in all_metrics], alpha=0.5, color="#DD8452", label="Top-5")
    axes[0].set_title("Accuracy"); axes[0].set_ylim(0,1.05); axes[0].legend()
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].bar(names, [m["train_time_sec"] for m in all_metrics], color="#55A868")
    axes[1].set_title("Train Time (sec)"); axes[1].tick_params(axis="x", rotation=30)
    axes[2].bar(names, [m["inference_ms"] for m in all_metrics], color="#C44E52")
    axes[2].set_title("Inference (ms/img)"); axes[2].tick_params(axis="x", rotation=30)
    plt.suptitle("Model Comparison — All 5 Models", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(MODELS_DIR/"comparison.png", dpi=150)
    plt.close()

    print(f"\nИтоговые результаты всех моделей:")
    print(f"{'Модель':<20} {'Top-1':>8} {'Top-5':>8} {'MAE':>8}")
    print("─" * 50)
    for m in all_metrics:
        print(f"  {m['model_name']:<18} {m['top1_accuracy']*100:>7.2f}% "
              f"{m['top5_accuracy']*100:>7.2f}% {m['price_mae']:>7.0f}тг")
    print(f"\n🏆 Лучшая модель: {best['model_name']}  Top-1={best['top1_accuracy']:.4f}")
    print("\n✅ Готово! Запустите: streamlit run step3_app.py")


if __name__ == "__main__":
    main()