import os
import json
import random
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pillow_heif import register_heif_opener
register_heif_opener()
import numpy as np

PHOTOS_DIR  = Path("photos")          
DATASET_DIR = Path("dataset")
TARGET_SIZE = (224, 224)
MIN_PER_CLASS = 120     
TRAIN_R, VAL_R, TEST_R = 0.70, 0.15, 0.15
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

AUGMENTATIONS = [
    lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
    lambda img: img.rotate(10,  fillcolor=(200,200,200)),
    lambda img: img.rotate(-10, fillcolor=(200,200,200)),
    lambda img: img.rotate(15,  fillcolor=(200,200,200)),
    lambda img: img.rotate(-15, fillcolor=(200,200,200)),
    lambda img: ImageEnhance.Brightness(img).enhance(0.75),
    lambda img: ImageEnhance.Brightness(img).enhance(1.30),
    lambda img: ImageEnhance.Contrast(img).enhance(0.80),
    lambda img: ImageEnhance.Contrast(img).enhance(1.25),
    lambda img: ImageEnhance.Color(img).enhance(0.70),
    lambda img: ImageEnhance.Color(img).enhance(1.40),
    lambda img: ImageEnhance.Sharpness(img).enhance(1.50),
    lambda img: img.filter(ImageFilter.GaussianBlur(radius=1)),
    lambda img: ImageOps.autocontrast(img),
]

def random_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    f = random.uniform(0.80, 0.95)
    nw, nh = int(w*f), int(h*f)
    l = random.randint(0, w-nw)
    t = random.randint(0, h-nh)
    return img.crop((l, t, l+nw, t+nh)).resize((w, h), Image.LANCZOS)

def augment(img: Image.Image) -> list:
    out = []
    for fn in AUGMENTATIONS:
        try: out.append(fn(img))
        except: pass
    for _ in range(3):
        out.append(random_crop(img))
    return out

def save_images(imgs, split_dir: Path, cls: str):
    d = split_dir / cls
    d.mkdir(parents=True, exist_ok=True)
    safe = cls.replace("/", "_").replace(" ", "_")
    for i, im in enumerate(imgs):
        im.resize(TARGET_SIZE, Image.LANCZOS).convert("RGB")\
          .save(d / f"{safe}_{i:04d}.jpg", "JPEG", quality=92)

def process_class(cls_dir: Path, cls_name: str):
    exts = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp"}
    paths = [p for p in cls_dir.iterdir() if p.suffix.lower() in exts]

    imgs = []
    for p in paths:
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception as e:
            print(f"   ⚠ skip {p.name}: {e}")

    if not imgs:
        print(f" ❌ {cls_name}: нет изображений!")
        return 0

    print(f"   {cls_name}: {len(imgs)} оригиналов", end=" → ")

    all_imgs = list(imgs)
    i = 0
    while len(all_imgs) < MIN_PER_CLASS:
        all_imgs += augment(imgs[i % len(imgs)])
        i += 1
    all_imgs = all_imgs[:max(MIN_PER_CLASS, len(imgs))]
    random.shuffle(all_imgs)

    n  = len(all_imgs)
    nt = int(n * TRAIN_R)
    nv = int(n * VAL_R)

    save_images(all_imgs[:nt],       DATASET_DIR/"train", cls_name)
    save_images(all_imgs[nt:nt+nv],  DATASET_DIR/"val",cls_name)
    save_images(all_imgs[nt+nv:],    DATASET_DIR/"test",cls_name)

    print(f"train={nt} val={nv} test={n-nt-nv}  (total={n})")
    return n

def main():
    if not PHOTOS_DIR.exists():
        print(f"Папка '{PHOTOS_DIR}' не найдена!")
        print(f"Убедитесь что запускаете из project_dl/")
        return

    class_dirs = sorted([d for d in PHOTOS_DIR.iterdir() if d.is_dir()])
    if not class_dirs:
        print("В папке photos/ нет подпапок.")
        return

    print(f"Найдено {len(class_dirs)} классов в {PHOTOS_DIR}\n")
    DATASET_DIR.mkdir(exist_ok=True)

    class_names = []
    for d in class_dirs:
        n = process_class(d, d.name)
        if n > 0:
            class_names.append(d.name)

    with open(DATASET_DIR / "class_names.json", "w", encoding="utf-8") as f:
        json.dump({"class_names": class_names}, f, ensure_ascii=False, indent=2)

    print(f"\nГотово! Классов: {len(class_names)}")
    print(f"Датасет сохранён в: {DATASET_DIR.resolve()}")
if __name__ == "__main__":
    main()
