"""
download_weights.py
Автоматически скачивает веса моделей с Google Drive.
Запускается автоматически из step5_app_v2.py при старте.
"""

import os
from pathlib import Path

# ── Google Drive file IDs ──────────────────────────────────────────────────
WEIGHTS = {
    "resnet50":        "12hOZwl9IPWtSlnm6mnECO_WEwUsv0956",
    "googlenet":       "13W63c0FrkBWA3TX5Nvj9xf4KQuVZAoT9",
    "alexnet":         "1Erc_MOipSvWlgGmipjFYs1rWPXZmBB_m",
    "vgg16":           "1YzLezC5shRqDUEpP09NFqT_lHQtiyCT8",
    "efficientnet_b0": "1tc82SZQUmtzWFm9xMvpxpgTOlScLsByi",
}

def download_file(file_id: str, dest: Path):
    """Скачивает файл с Google Drive по file_id."""
    import requests
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = requests.Session()
    response = session.get(url, stream=True)

    # Google Drive может вернуть страницу подтверждения для больших файлов
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            url = f"https://drive.google.com/uc?export=download&confirm={value}&id={file_id}"
            response = session.get(url, stream=True)
            break

    dest.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
                total += len(chunk)

    size_mb = total / 1024 / 1024
    return size_mb

def ensure_weights():
    """
    Проверяет наличие весов. Если нет — скачивает с Google Drive.
    Возвращает True если все веса готовы.
    """
    models_dir = Path("models")
    missing = []

    for name in WEIGHTS:
        path = models_dir / name / "best_model.pth"
        if not path.exists():
            missing.append(name)

    if not missing:
        return True  # все веса уже есть

    print(f"⬇️  Скачиваем {len(missing)} модел(и) с Google Drive...")

    for name in missing:
        file_id = WEIGHTS[name]
        dest    = models_dir / name / "best_model.pth"
        print(f"  → {name}...", end=" ", flush=True)
        try:
            mb = download_file(file_id, dest)
            print(f"✅ {mb:.1f} MB")
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            return False

    print("✅ Все веса загружены!\n")
    return True


if __name__ == "__main__":
    ensure_weights()
