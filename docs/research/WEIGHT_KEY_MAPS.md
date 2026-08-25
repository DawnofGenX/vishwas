# Model weight key-map provenance (probed 2026-08-21, torch 2.13.0 via docling tree)

Read-only probes of /opt/vishwas/models/* — source of truth for arch vendoring (P1 tasks 1.2–1.4).

## AASIST — /opt/vishwas/models/aasist/best_model.pth (3.79 GB)
Top-level dict: {model_state, optimizer_state, epoch, step, best_metric}; model_state = 733 tensors.
- ROOTS: frontend.* / backend.* / losses.*
- frontend.model.post_extract_proj.weight (1024,512) — wav2vec-style feature extractor trunk ending at 1024-d
- backend.GAT_layer_S/GAT layers (graph-attention trunk, dim 64)
- losses.0.fc.weight (2,160) — final head consumes a **160-d vector** → 2-class (real vs spoof)
Key-map note: `losses.` prefix in state dict means head lived under the training module — port must rename to e.g. self.head.

## EFFORT ×3 — /opt/vishwas/models/effort/{chameleon,ffpp,genimage}/effort_*.pth (1.21 GB each)
OrderedDict, 681 tensors, ALL keys 'module.'-prefixed (DataParallel save).
- Full ViT-L trunk INSIDE checkpoint (no external CLIP): embeddings.patch_embedding (1024,3,14,14), position 257×1024, N encoder layers
- OrthAlign subspace-decomposition heads live IN self-attention: k_proj.weight_main / .bias / .S_residual
- head.weight (2,1024) → real vs gen/AIGC 2-class
License: CC BY-NC 4.0 (YZY-stack/Effort-AIGI-Detection). Chameleon PRIMARY (locked decision #2); ffpp/genimage lazy fallback chain.

## HAVIC ×2 — /opt/vishwas/models/havic/best_ft/best_ft_model.pth (859 MB) + pt200/pt_model.200.pth (973 MB)
456 tensors (best_ft): audio_encoder(164)/visual_encoder(163) independent encoders (ViT-S-ish, dim 768, visual patch 2×16×16 pos 1568), multi-scale AudioTokenReducer_{3,6,9,12,AVI} + Visual* (mlp 768→128→1), AudioVisualInteractionModule (69 tensors), classifier/classifier_audio/classifier_visual (6 each), pool_a, pool_v.
Fallback chain: best_ft primary, pt200 secondary. Expect SLOW tier on i5-8250U CPU.
