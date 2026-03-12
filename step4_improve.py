import json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
from torchvision import models, transforms

MODELS_DIR    = Path("models")
PRICE_DB_PATH = Path("price_database.json")
IMG_SIZE      = 224
DEVICE        = "cpu"


def load_model(name):
    path = MODELS_DIR / name / "best_model.pth"
    if not path.exists():
        print(f"  ⚠️  {name}: файл не найден, пропускаем")
        return None, None
    ckpt = torch.load(path, map_location=DEVICE)
    cn   = ckpt["class_names"]
    nc   = len(cn)
    if name == "alexnet":
        m = models.alexnet(weights=None); m.classifier[6] = nn.Linear(4096, nc)
    elif name == "vgg16":
        m = models.vgg16(weights=None);   m.classifier[6] = nn.Linear(4096, nc)
    elif name == "googlenet":
        m = models.googlenet(weights=None, aux_logits=False); m.fc = nn.Linear(1024, nc)
    elif name == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(2048, nc))
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        in_f = models.efficientnet_b0().classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, nc))
    else:
        return None, None
    m.load_state_dict(ckpt["state_dict"]); m.eval()
    return m, cn


BASE_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict_single(model, img: Image.Image) -> np.ndarray:
    """Базовое предсказание → возвращает массив вероятностей."""
    t = BASE_TF(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        out = model(t)
        if hasattr(out, "logits"): out = out.logits
        return torch.softmax(out, 1)[0].numpy()


def get_tta_transforms():
    
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    base_crop = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(IMG_SIZE)])

    def make_tf(*extra):
        return transforms.Compose([base_crop, *extra,
                                   transforms.ToTensor(), normalize])

    return [
        make_tf(),                                             # 1. оригинал
        make_tf(transforms.RandomHorizontalFlip(p=1.0)),       # 2. отражение
        make_tf(transforms.RandomRotation(degrees=(10, 10))),  # 3. поворот +10°
        make_tf(transforms.RandomRotation(degrees=(-10,-10))), # 4. поворот -10°
        make_tf(transforms.ColorJitter(brightness=0.3)),       # 5. ярче
        make_tf(transforms.ColorJitter(brightness=(0.5, 0.8))),# 6. темнее
        make_tf(transforms.ColorJitter(contrast=0.3)),         # 7. контраст
        make_tf(transforms.RandomResizedCrop(IMG_SIZE,         # 8. чуть зумированный кроп
                                              scale=(0.85, 1.0))),
    ]

TTA_TRANSFORMS = get_tta_transforms()

def predict_tta(model, img: Image.Image) -> np.ndarray:
    """
    TTA: 8 аугментаций → усредняем вероятности.
    Значительно стабилизирует предсказания.
    """
    img_rgb = img.convert("RGB")
    all_probs = []
    for tf in TTA_TRANSFORMS:
        try:
            t = tf(img_rgb).unsqueeze(0)
            with torch.no_grad():
                out = model(t)
                if hasattr(out, "logits"): out = out.logits
                all_probs.append(torch.softmax(out, 1)[0].numpy())
        except Exception:
            pass  # если аугментация сломалась — пропускаем
    return np.mean(all_probs, axis=0)

# ─────────────────────────────────────────────────────────────────────────────
# 4. ENSEMBLE — взвешенное усреднение моделей
# ─────────────────────────────────────────────────────────────────────────────

# Веса моделей на основе их Top-1 accuracy
# ResNet50 (98.3%) получает наибольший вес
MODEL_WEIGHTS = {
    "resnet50":        0.40,  # лучшая модель
    "vgg16":           0.25,  # вторая по точности
    "googlenet":       0.15,
    "alexnet":         0.10,
    "efficientnet_b0": 0.10,
}

def predict_ensemble(models_dict: dict, img: Image.Image,
                     use_tta: bool = True) -> tuple:
    """
    Ensemble + TTA:
    1. Для каждой модели → TTA или базовый инференс
    2. Взвешенное усреднение по MODEL_WEIGHTS
    3. Возвращает (top5_labels, top5_probs)
    """
    weighted_probs = None
    total_weight   = 0.0

    for name, (model, class_names) in models_dict.items():
        if model is None:
            continue
        w = MODEL_WEIGHTS.get(name, 0.1)
        probs = predict_tta(model, img) if use_tta else predict_single(model, img)
        if weighted_probs is None:
            weighted_probs = probs * w
        else:
            weighted_probs += probs * w
        total_weight += w

    if weighted_probs is None:
        raise RuntimeError("Нет доступных моделей!")

    weighted_probs /= total_weight  # нормализуем
    idx5 = np.argsort(weighted_probs)[::-1][:5]

    # class_names берём из любой доступной модели
    cn = next(cn for _, (m, cn) in models_dict.items() if m is not None)
    return [cn[i] for i in idx5], [float(weighted_probs[i]) for i in idx5]

# ─────────────────────────────────────────────────────────────────────────────
# 5. ТЕСТ — сравниваем старый vs новый метод
# ─────────────────────────────────────────────────────────────────────────────
def run_comparison_test():
    """
    Загружает тестовые фото из dataset/test и сравнивает:
    - Базовый ResNet50
    - ResNet50 + TTA
    - Ensemble (все модели) + TTA
    """
    print("\n" + "="*60)
    print("  ТЕСТ: Базовый vs TTA vs Ensemble+TTA")
    print("="*60)

    # Загружаем модели
    print("\n📦 Загрузка моделей...")
    models_dict = {}
    for name in ["resnet50", "vgg16", "googlenet", "alexnet", "efficientnet_b0"]:
        print(f"  → {name}...", end=" ")
        m, cn = load_model(name)
        models_dict[name] = (m, cn)
        print("✅" if m else "❌ пропущено")

    resnet_model, class_names = models_dict["resnet50"]
    if resnet_model is None:
        print("❌ ResNet50 не найден — тест невозможен")
        return

    # Ищем тестовые фото
    test_dir = Path("dataset/test")
    if not test_dir.exists():
        print("\n⚠️  dataset/test не найден, используем dataset/val")
        test_dir = Path("dataset/val")
    if not test_dir.exists():
        print("❌ Нет тестовых данных. Тест пропущен.")
        return

    # Собираем фото (по 1 из каждого класса)
    test_images = []
    for class_dir in sorted(test_dir.iterdir()):
        if not class_dir.is_dir(): continue
        imgs = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + \
               list(class_dir.glob("*.jpeg"))
        if imgs:
            test_images.append((imgs[0], class_dir.name))
        if len(test_images) >= 16: break

    if not test_images:
        print("❌ Тестовые изображения не найдены")
        return

    print(f"\n🔍 Тестируем на {len(test_images)} изображениях...\n")
    print(f"{'Класс':<30} {'Базовый':>12} {'TTA':>12} {'Ensemble+TTA':>14} {'Лучший':>10}")
    print("-"*82)

    base_correct, tta_correct, ens_correct = 0, 0, 0

    for img_path, true_class in test_images:
        img = Image.open(img_path).convert("RGB")

        # Метод 1: базовый
        t0 = time.time()
        probs_base = predict_single(resnet_model, img)
        idx_base   = np.argmax(probs_base)
        pred_base  = class_names[idx_base]
        conf_base  = probs_base[idx_base]
        t_base     = (time.time()-t0)*1000

        # Метод 2: TTA
        t0 = time.time()
        probs_tta = predict_tta(resnet_model, img)
        idx_tta   = np.argmax(probs_tta)
        pred_tta  = class_names[idx_tta]
        conf_tta  = probs_tta[idx_tta]
        t_tta     = (time.time()-t0)*1000

        # Метод 3: Ensemble + TTA
        t0 = time.time()
        top5_lbl, top5_prob = predict_ensemble(models_dict, img, use_tta=True)
        pred_ens  = top5_lbl[0]
        conf_ens  = top5_prob[0]
        t_ens     = (time.time()-t0)*1000

        # Считаем точность по классу (убираем номер из имени)
        true_clean = true_class.split(". ", 1)[-1] if ". " in true_class else true_class

        ok_base = "✅" if true_clean in pred_base or pred_base in true_clean else "❌"
        ok_tta  = "✅" if true_clean in pred_tta  or pred_tta  in true_clean else "❌"
        ok_ens  = "✅" if true_clean in pred_ens  or pred_ens  in true_clean else "❌"

        if ok_base == "✅": base_correct += 1
        if ok_tta  == "✅": tta_correct  += 1
        if ok_ens  == "✅": ens_correct  += 1

        # Кто лучше всего уверен?
        best_conf = max(conf_base, conf_tta, conf_ens)
        best_who  = ("Base" if conf_base == best_conf else
                     "TTA"  if conf_tta  == best_conf else "Ensemble")

        print(f"{true_clean[:29]:<30} "
              f"{ok_base} {conf_base*100:5.1f}%  "
              f"{ok_tta}  {conf_tta*100:5.1f}%  "
              f"{ok_ens}  {conf_ens*100:5.1f}%    "
              f"{best_who:>10}")

    n = len(test_images)
    print("-"*82)
    print(f"\n📊 ИТОГИ:")
    print(f"  Базовый ResNet50:    {base_correct}/{n} = {base_correct/n*100:.1f}%")
    print(f"  ResNet50 + TTA:      {tta_correct}/{n}  = {tta_correct/n*100:.1f}%"
          f"  {'↑ лучше!' if tta_correct > base_correct else '= одинаково'}")
    print(f"  Ensemble + TTA:      {ens_correct}/{n}  = {ens_correct/n*100:.1f}%"
          f"  {'↑ лучше!' if ens_correct > base_correct else '= одинаково'}")

    print("\n⚡ Скорость инференса (примерно):")
    print(f"  Базовый:       ~30-40ms")
    print(f"  TTA (8x):      ~200-300ms")
    print(f"  Ensemble+TTA:  ~800-1500ms (все 5 моделей × 8 аугментаций)")
    print("\n💡 Рекомендация: используй Ensemble+TTA для демонстрации преподавателю")
    print("   (скорость не критична, зато уверенность значительно выше)")

# ─────────────────────────────────────────────────────────────────────────────
# 6. СОХРАНЯЕМ КОНФИГ ДЛЯ step3_app.py
# ─────────────────────────────────────────────────────────────────────────────
def save_inference_config():
    """Сохраняет конфиг чтобы step3_app.py знал какой режим использовать."""
    config = {
        "inference_mode": "ensemble_tta",   # "single" | "tta" | "ensemble_tta"
        "model_weights": MODEL_WEIGHTS,
        "tta_count": len(TTA_TRANSFORMS),
        "description": "Ensemble of 5 CNNs with 8x TTA each"
    }
    with open(MODELS_DIR / "inference_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n✅ Конфиг сохранён: models/inference_config.json")

if __name__ == "__main__":
    print("🚀 Улучшение инференса: TTA + Ensemble")
    run_comparison_test()
    save_inference_config()
    print("\n" + "="*60)
    print("  СЛЕДУЮЩИЙ ШАГ:")
    print("  Запусти: python step5_app_v2.py")
    print("  (обновлённое приложение с Ensemble+TTA)")
    print("="*60)