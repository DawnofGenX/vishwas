# Vishwas — Image Detector Fine-Tuning

A small, runnable pipeline to train your own **AI-generated-image detector** by
fine-tuning a pretrained backbone. This is the fastest realistic way to lift
image-detection accuracy on *your* domain (Indian faces, WhatsApp-compressed
photos, the specific generators you care about).

> **What this is / isn't.** This trains a *new* transfer-learning classifier
> (EfficientNet-B0 + a real/AI head). It does **not** fine-tune SPAI — that
> needs SPAI's heavy upstream training repo and a real GPU. This is the version
> a student can actually run on Colab's free GPU. The output is a normal
> PyTorch checkpoint you can test immediately and wire into Vishwas later.

## Files
- `train_image_detector.py` — the training script (fine-tune, evaluate, export).
- `predict_image.py` — run the trained model on an image/folder, print P(AI).
- `requirements.txt` — dependencies.

## 1. Get data
Arrange images into two folders (names must be `real` and `fake`):

```
data/
  real/    genuine photos          -> label 0
  fake/    AI-generated / deepfake  -> label 1
```

You do **not** pre-split into train/val — the script does that itself.

**Where to get images (mix several sources for robustness):**
- **Real:** your own photos, or a subset of a public photo set (ImageNet/COCO).
- **AI:** the *CIFAKE* dataset, Kaggle's *"AI-Generated vs Real Images"*, or
  your own Midjourney / Stable-Diffusion / DALL·E outputs.

**The single most important thing:** include the *kinds* of images you actually
verify in production — faces, screenshots, re-compressed JPEGs. In-domain data
is where the accuracy gain really comes from. Keep the two classes roughly
balanced (similar counts), and aim for at least a few hundred images per class
to start (a few thousand is much better).

## 2. Train (Google Colab — free GPU)
1. Open [colab.research.google.com](https://colab.research.google.com) →
   **Runtime ▸ Change runtime type ▸ GPU**.
2. Upload `train_image_detector.py` and `predict_image.py`, and your `data/`
   folder (or mount Google Drive and point `--data` at it).
3. In a cell:

```bash
!pip install -q torch torchvision scikit-learn pillow
!python train_image_detector.py --data ./data --epochs 8 --out ./out
```

That's it. It prints per-epoch validation accuracy and ROC-AUC, and saves the
**best** checkpoint (by val AUC) to `out/image_detector.pt`.

**Useful flags:** `--epochs 12` (train longer), `--batch 64` (if GPU memory
allows), `--freeze-backbone` (faster, for small datasets — trains only the new
head), `--lr 1e-4` (lower LR if training is unstable).

## 3. Test it
```bash
python predict_image.py --model out/image_detector.pt --image some_photo.jpg
# or a whole folder:
python predict_image.py --model out/image_detector.pt --image ./test_images
```
Prints `P(AI)` (0..1) and a REAL/AI label per image.

## 4. Read the results honestly
- **ROC-AUC** is the headline: 0.5 = coin flip, 0.9+ = strong. Look at
  `out/metrics.json` for precision/recall and the confusion matrix.
- **Watch for overfitting:** if train loss keeps dropping but val AUC stalls or
  falls, you need more/varied data or `--freeze-backbone`.
- **Test on held-out, out-of-distribution images** (generators and real sources
  the model never saw) — that number is the one that matters, not the val split.

## 5. Wiring it into Vishwas (later)
The Vishwas image slot (`VISHWAS_IMAGE_FACE_WEIGHTS`) currently expects the SPAI
architecture, so this EfficientNet checkpoint won't drop straight in. To use it,
the repo needs a small **adapter** that registers an `efficientnet_b0` image
arch and maps this checkpoint's `prob(AI)` into an `image_face_forensics` check
(the signal `fusion.py` already reads via `faceforensics.prob`). That's a
focused ~1-file addition — ask Claude to "add an adapter so Vishwas can load the
fine-tuned EfficientNet image detector" once you have a checkpoint you're happy
with.

## Notes
- Deterministic for a fixed `--seed`.
- CPU works but is slow — use a GPU. The script auto-detects and uses CUDA.
- Handles class imbalance (weights the loss) but balanced data is still better.
