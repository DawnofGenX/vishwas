# _spai/ — Vendored SPAI net (PROVENANCE)

## Upstream
- **Project:** SPAI — Spectral AI-Generated Image Detector
  - CVPR 2025: "Any-Resolution AI-Generated Image Detection by Spectral Learning"
  - arXiv: 2411.19417 — Karageorgiou, Papadopoulos, Kompatsiaris, Gavves (CERTH / UvA).
- **Repo:** https://github.com/mever-team/spai
- **Commit pinned:** `8ff7b3b6779b4fcb43cf313471d9cb1c62d129a4` (2025-08-01, "Update README.md")
- **License:** **Apache-2.0** (LICENSE file + SPDX headers), NOT MIT. The official
  COPY is retained at `_spai/LICENSE`.

## Files vendored (copied verbatim from the pinned commit, imports re-homed)
| vendored file              | upstream path             | re-homes |
|----------------------------|---------------------------|----------|
| `vision_transformer.py`    | `spai/models/vision_transformer.py` | `timm.models.layers.{DropPath,to_2tuple,trunc_normal_}` inlined (no timm dep) |
| `filters.py`               | `spai/models/filters.py`  | none (torch-only) |
| `net_utils.py`             | `spai/models/utils.py`    | none (numpy/torch only) |
| `sid.py`                   | `spai/models/sid.py`      | `timm.data` ImageNet mean/std inlined; `from .utils`→`from .net_utils as utils`; dropped `backbones` import (unreachable with MFM ViT); `spai.utils.save_image_with_attention_overlay`↦ stub (export-only path, unused for scoring); added `from __future__ import annotations` |
| `conv_config.py`           | new — minimal dotted-config reproducing `configs/spai.yaml` over `config.py` defaults (a yacs CfgNode was NOT vendored; see module docstring). Values verified against checkpoint shapes: cls_vector_dim 1096, mlp_ratio 3, attn_embed_dim 1536 (patch_aggregator (12,1,128)), frequencies_mask 224² radius 16, pos_embed (1,197,768) |

Runtime deps of the vendored net: `torch`, `torchvision` (five_crop in the
patch-fallback), `einops`, `PIL`, `numpy`. None of the upstream training/
logging/export deps (yacs, neptune, timm, apex, click, filetype) are required
for scoring.

## Architecture (as configured for the published checkpoint)
`build_mf_vit` → `PatchBasedMFViT` (RESOLUTION_MODE="arbitrary"):
- **Backbone** = MFM ViT-B/16 (`VisionTransformer`, embed 768, depth 12, heads 12,
  patch 16, ImageNet norm, intermediate layers 0..11, mean pooling).
- **FRE** = `FrequencyRestorationEstimator`: per-feature patch projector (dim 768→1024,
  2 layers, no last activation), original-image features branch (proj_dim 1024).
- **SCA head** = spectral-context attention (num_heads 12, attn_embed 1536,
  patch_aggregator) → `norm(1096)` → `ClassificationHead` (mlp_ratio 3), 1 logit.
- Input: raw image float32 **[0,1]** (NOT ImageNet-normalised) per user tensor,
  padded to ≥224; DFS filtering + ImageNet norm happen inside `MFViT.forward`.
- Output: `sigmoid(logit)` = **p_fake** (class 1 == AI-generated; matches the
  `data/{fake,real}_*.csv` label convention).

## Inference contract implemented by `vishwas.model_archs.spai`
Replicates upstream `spai infer` (`validate` in `spai/__main__.py`):
1. decode image → RGB uint8 (converted from the pipeline's cv2 BGR),
2. float32 [0,1] (ToTensorV2 equivalent), PadIfNeeded to ≥224 (zero-pad),
3. `PatchBasedMFViT.forward([t], feature_extraction_batch=400)` → 1×1 logit,
4. `sigmoid` → p_fake.

Polarity: p_fake in [0,1]; higher ⇒ AI-generated.