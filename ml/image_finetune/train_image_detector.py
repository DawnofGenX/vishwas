#!/usr/bin/env python3
"""Fine-tune an AI-generated-image detector (transfer learning).

Trains a binary classifier — REAL photo (0) vs AI-GENERATED / manipulated (1) —
by fine-tuning a pretrained EfficientNet-B0 backbone. Designed to run on a free
Colab/Kaggle GPU in well under an hour for a few-thousand-image dataset.

WHY transfer learning: a backbone pretrained on ImageNet already knows generic
visual features; we only teach it the (much smaller) real-vs-AI decision. That
needs far less data and compute than training from scratch, and far less than
fine-tuning a spectral model like SPAI.

------------------------------------------------------------------ DATA LAYOUT
Point --data at a folder with exactly two subfolders (torchvision ImageFolder):

    data/
      real/     <- genuine photos           (label 0)
      fake/     <- AI-generated / deepfaked  (label 1)

The script makes its own train/val split (--val-frac); you do NOT pre-split.
Class names are read alphabetically, so 'fake'=0? NO — we force real=0/fake=1
explicitly regardless of folder order (see _CLASS_TO_IDX) so the exported
probability is always P(AI-generated).

------------------------------------------------------------------- QUICK START
    pip install torch torchvision scikit-learn pillow
    python train_image_detector.py --data ./data --epochs 8 --out ./out

Outputs (in --out):
    image_detector.pt   trained weights + metadata (load with predict_image.py)
    metrics.json        val accuracy / ROC-AUC / precision / recall / confusion
    training_log.csv    per-epoch train/val loss and val AUC

Public datasets to build ./data from (mix several for robustness):
  - Real: any photo set (e.g. a subset of ImageNet, COCO, or your own photos).
  - AI:  CIFAKE, "AI-Generated vs Real Images" (Kaggle), or your own
         Midjourney/Stable-Diffusion/DALL-E outputs.
  BALANCE the two classes and include the KINDS of images you actually care
  about (e.g. faces, WhatsApp-compressed JPEGs) — in-domain data is where the
  accuracy gain really comes from.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# real=0, fake/ai=1 — fixed so the exported model's positive class is always
# "AI-generated", no matter how ImageFolder orders the directories.
_CLASS_TO_IDX = {"real": 0, "fake": 1, "ai": 1, "generated": 1, "genuine": 0}

IMG_SIZE = 224


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    # Training augmentation matters here: AI-image artifacts must survive the
    # kind of degradation real-world images suffer (recompression, resizing),
    # or the model latches onto brittle generator fingerprints that vanish once
    # an image is screenshotted or sent over WhatsApp.
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.15, 0.15, 0.15),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
        transforms.ToTensor(),
        norm,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        norm,
    ])
    return train_tf, eval_tf


class _RemappedImageFolder(ImageFolder):
    """ImageFolder that forces real=0 / fake=1 via _CLASS_TO_IDX."""

    def find_classes(self, directory):
        classes = sorted(e.name for e in Path(directory).iterdir() if e.is_dir())
        if not classes:
            raise FileNotFoundError(f"no class subfolders in {directory}")
        mapping = {}
        for c in classes:
            key = c.strip().lower()
            if key not in _CLASS_TO_IDX:
                raise ValueError(
                    f"folder '{c}' not recognised. Name your subfolders one of "
                    f"{sorted(set(_CLASS_TO_IDX))} (real photos vs AI images).")
            mapping[c] = _CLASS_TO_IDX[key]
        return classes, mapping


def build_model(device: torch.device) -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT          # ImageNet-pretrained
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)    # single logit -> BCE
    return model.to(device)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    from sklearn.metrics import (roc_auc_score, precision_score,
                                 recall_score, confusion_matrix)
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        logit = model(x).squeeze(1)
        prob = torch.sigmoid(logit).cpu()
        ps.extend(prob.tolist())
        ys.extend(y.tolist())
    preds = [1 if p >= 0.5 else 0 for p in ps]
    acc = sum(int(a == b) for a, b in zip(preds, ys)) / max(1, len(ys))
    try:
        auc = float(roc_auc_score(ys, ps)) if len(set(ys)) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    tn, fp, fn, tp = confusion_matrix(ys, preds, labels=[0, 1]).ravel()
    return {
        "accuracy": round(acc, 4),
        "roc_auc": round(auc, 4),
        "precision": round(float(precision_score(ys, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(ys, preds, zero_division=0)), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_val": len(ys),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder with real/ and fake/ subdirs")
    ap.add_argument("--out", default="./out", help="output directory")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="train only the new head (faster, needs less data)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cpu":
        print("WARNING: no GPU detected — training will be slow. On Colab: "
              "Runtime > Change runtime type > GPU.")

    train_tf, eval_tf = build_transforms()
    full = _RemappedImageFolder(args.data, transform=train_tf)
    n_val = max(1, int(len(full) * args.val_frac))
    n_train = len(full) - n_val
    if n_train < 1:
        raise SystemExit("dataset too small to split")
    g = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=g)
    # val set must use eval transforms (no augmentation) — wrap a view
    val_ds.dataset = _RemappedImageFolder(args.data, transform=eval_tf)

    n_workers = 2 if device.type == "cuda" else 0
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=n_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=n_workers, pin_memory=(device.type == "cuda"))
    print(f"train={n_train}  val={n_val}  (real=0, fake/ai=1)")

    model = build_model(device)
    if args.freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("classifier"):
                p.requires_grad_(False)
        print("backbone frozen — training classifier head only")

    # class imbalance -> weight the positive term of BCE
    labels = [full.samples[i][1] for i in train_ds.indices]
    n_pos = sum(labels) or 1
    n_neg = (len(labels) - n_pos) or 1
    pos_weight = torch.tensor([n_neg / n_pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_rows = []
    best_auc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.float().to(device)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=use_amp):
                logit = model(x).squeeze(1)
                loss = criterion(logit, y)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            running += loss.item() * x.size(0)
        sched.step()
        train_loss = running / max(1, n_train)
        metrics = evaluate(model, val_loader, device)
        print(f"epoch {epoch:2d}/{args.epochs}  train_loss={train_loss:.4f}  "
              f"val_acc={metrics['accuracy']:.3f}  val_auc={metrics['roc_auc']:.3f}")
        log_rows.append({"epoch": epoch, "train_loss": round(train_loss, 5),
                         **{f"val_{k}": v for k, v in metrics.items()
                            if k not in ("confusion",)}})
        if metrics["roc_auc"] == metrics["roc_auc"] and metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            torch.save({
                "model_state": model.state_dict(),
                "arch": "efficientnet_b0",
                "img_size": IMG_SIZE,
                "positive_class": "ai_generated",
                "normalize": {"mean": [0.485, 0.456, 0.406],
                              "std": [0.229, 0.224, 0.225]},
                "val_metrics": metrics,
                "epoch": epoch,
            }, out / "image_detector.pt")
            (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
            print(f"  -> saved best (val_auc={best_auc:.3f})")

    with (out / "training_log.csv").open("w", newline="") as f:
        if log_rows:
            w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            w.writeheader()
            w.writerows(log_rows)

    print(f"\nDone. Best val ROC-AUC = {best_auc:.3f}")
    print(f"Checkpoint: {out / 'image_detector.pt'}")
    print("Sanity-check it on a single image with:  python predict_image.py "
          f"--model {out/'image_detector.pt'} --image some_photo.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
