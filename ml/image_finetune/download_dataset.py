#!/usr/bin/env python3
"""Download a full real-vs-AI image dataset and arrange it as real/ + fake/.

Run this IN COLAB (or anywhere with disk + network) to get a large dataset for
real training — far bigger than the small starter set shipped alongside.

    pip install datasets pillow
    python download_dataset.py --dataset cifake --out ./data --per-class 8000
    python download_dataset.py --dataset hemg   --out ./data --per-class 6000

Label mapping is VERIFIED from each dataset's ClassLabel names and baked in, so
real/ and fake/ are always correct (getting this backwards trains an inverted
detector):
  - cifake : dragonintelligence/CIFAKE-image-dataset   names ['FAKE','REAL']
             32x32 photos (CIFAR-10 real vs Stable-Diffusion fake). Canonical
             benchmark; low-res, so good for proving accuracy but weak transfer
             to high-res real-world images.
  - hemg   : Hemg/AI-Generated-vs-Real-Images-Datasets names ['AiArtData','RealArt']
             higher-resolution, varied. More realistic image sizes, art-leaning
             content.

Mix in YOUR OWN in-domain images (faces, documents, WhatsApp-compressed photos)
for the accuracy that actually matters — see DATASET_README.md.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# dataset key -> (hf_id, split, index-of-REAL-class). fake = the other class.
_SOURCES = {
    "cifake": ("dragonintelligence/CIFAKE-image-dataset", "train", 1),  # ['FAKE','REAL']
    "hemg":   ("Hemg/AI-Generated-vs-Real-Images-Datasets", "train", 1),  # ['AiArtData','RealArt']
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(_SOURCES), default="cifake")
    ap.add_argument("--out", default="./data")
    ap.add_argument("--per-class", type=int, default=8000,
                    help="images to save per class (real and fake each)")
    ap.add_argument("--max-side", type=int, default=512,
                    help="downscale images whose longer side exceeds this (0=off)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from datasets import load_dataset
    from PIL import Image

    hf_id, split, real_idx = _SOURCES[args.dataset]
    print(f"loading {hf_id} [{split}] ...")
    ds = load_dataset(hf_id, split=split).shuffle(seed=args.seed)

    out = Path(args.out)
    (out / "real").mkdir(parents=True, exist_ok=True)
    (out / "fake").mkdir(parents=True, exist_ok=True)
    counts = {"real": 0, "fake": 0}
    target = args.per_class

    for ex in ds:
        if counts["real"] >= target and counts["fake"] >= target:
            break
        lab = ex.get("label", ex.get("labels"))
        cls = "real" if lab == real_idx else "fake"
        if counts[cls] >= target:
            continue
        try:
            img = ex["image"].convert("RGB")
        except Exception:
            continue
        if args.max_side and max(img.size) > args.max_side:
            img.thumbnail((args.max_side, args.max_side))
        img.save(out / cls / f"{args.dataset}_{cls}_{counts[cls]:06d}.jpg", quality=90)
        counts[cls] += 1
        if sum(counts.values()) % 1000 == 0:
            print(f"  saved real={counts['real']} fake={counts['fake']}")

    print(f"\nDone. real={counts['real']} fake={counts['fake']} -> {out}")
    print("Now train:  python train_image_detector.py --data", str(out), "--epochs 8 --out ./out")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
