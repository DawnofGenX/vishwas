#!/usr/bin/env python3
"""Run a trained image detector on one image (or a folder) and print P(AI).

    python predict_image.py --model out/image_detector.pt --image photo.jpg
    python predict_image.py --model out/image_detector.pt --image ./some_folder

Prints, per image, the probability that it is AI-generated (0..1) and a label
at threshold 0.5. This is the same inference the Vishwas image slot would run.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def load_model(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    size = ckpt.get("img_size", 224)
    n = ckpt.get("normalize", {"mean": [0.485, 0.456, 0.406],
                               "std": [0.229, 0.224, 0.225]})
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=n["mean"], std=n["std"]),
    ])
    return model, tf, ckpt


@torch.no_grad()
def score(model, tf, device, img_path: Path) -> float:
    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    return float(torch.sigmoid(model(x).squeeze(1)).item())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True, help="an image file or a folder of images")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tf, ckpt = load_model(args.model, device)
    print(f"model: {ckpt.get('arch')} | val_auc="
          f"{ckpt.get('val_metrics', {}).get('roc_auc')} | positive=AI-generated\n")

    target = Path(args.image)
    paths = ([target] if target.is_file()
             else sorted(p for p in target.rglob("*") if p.suffix.lower() in _EXTS))
    if not paths:
        raise SystemExit(f"no images found at {target}")
    for p in paths:
        try:
            prob = score(model, tf, device, p)
        except Exception as e:  # noqa: BLE001
            print(f"{p.name:40s}  ERROR: {e}")
            continue
        label = "AI-GENERATED" if prob >= args.threshold else "REAL"
        print(f"{p.name:40s}  P(AI)={prob:.3f}  -> {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
