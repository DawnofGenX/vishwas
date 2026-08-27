# Vishwas — Starter Image Dataset

A small, **ready-to-train** real-vs-AI image dataset, already arranged in the
exact layout `train_image_detector.py` expects.

```
data/
  real/   800 genuine photos        (label 0)
  fake/   800 AI-generated images   (label 1)
```

## Source & labels (verified)
- From **CIFAKE** (`dragonintelligence/CIFAKE-image-dataset`), test split.
- CIFAKE's real images are CIFAR-10 photos; its fake images are Stable-Diffusion
  generations of the same categories.
- Label mapping was read from the dataset's own class names `['FAKE','REAL']`
  and applied as **real/ = REAL, fake/ = FAKE** — verified, not guessed.

## Train on it immediately
```bash
pip install torch torchvision scikit-learn pillow
python train_image_detector.py --data ./data --epochs 8 --out ./out
```
(Use a GPU — Colab: Runtime ▸ Change runtime type ▸ GPU.)

## Important limitation — read this
CIFAKE images are **32×32 pixels**. That makes this a great set for *proving the
pipeline works* and for benchmarking, but a model trained only on it will not
transfer well to the high-resolution images you actually verify (faces,
screenshots, WhatsApp photos). Treat this as your **starter / smoke-test set**,
not your production data.

## Scale up + make it relevant
`download_dataset.py` pulls a much larger set into the same layout:
```bash
pip install datasets pillow
# full CIFAKE (bigger, still low-res benchmark):
python download_dataset.py --dataset cifake --out ./data --per-class 8000
# higher-resolution, varied images:
python download_dataset.py --dataset hemg --out ./data --per-class 6000
```

**The accuracy that matters comes from in-domain data.** Add your own folders of:
- **Real:** photos of the kind you verify — faces, ID documents, forwarded images.
- **Fake:** AI-generated versions in the same style — Midjourney/DALL·E/SD faces,
  AI document forgeries, deepfake stills.

Just drop them into `data/real/` and `data/fake/` alongside the starter images
(keep the two classes roughly balanced) and retrain. A few thousand in-domain
images per class will beat tens of thousands of off-domain ones.

## Ethics / licensing
CIFAKE is public and widely used for research. If you collect your own images,
make sure you have the right to use them, and avoid real people's images used
without consent — especially for a public-facing safety tool.
