"""Explicit, bounded model download; never sends user images."""

import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
REV = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        "face_detection_yunet",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    "face_recognition_sface_2021dec.onnx": (
        "face_recognition_sface",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
}


def verified(path, expected):
    return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected


def paths():
    result = []
    for name, (_, digest) in MODELS.items():
        path = ROOT / "models" / name
        if not verified(path, digest):
            raise ValueError(
                "Модель отсутствует или повреждена. Запусти: python app.py --download-models"
            )
        result.append(path)
    return result


def download():
    folder = ROOT / "models"
    folder.mkdir(exist_ok=True)
    for name, (group, digest) in MODELS.items():
        path = folder / name
        if not verified(path, digest):
            url = f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{REV}/models/{group}/{name}"
            print(f"Downloading {name} ...", flush=True)
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read(50_000_001)
            if len(data) > 50_000_000 or hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"Неверная контрольная сумма: {name}")
            temporary = path.with_suffix(".part")
            temporary.write_bytes(data)
            temporary.replace(path)
        license_path = folder / (group + "-LICENSE")
        if not license_path.exists():
            url = f"https://raw.githubusercontent.com/opencv/opencv_zoo/{REV}/models/{group}/LICENSE"
            with urllib.request.urlopen(url, timeout=30) as response:
                license_path.write_bytes(response.read(100_000))
        print(f"OK: {name}")
