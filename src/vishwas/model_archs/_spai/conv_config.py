"""Minimal SPAI inference config (re-homed from mever-team/spai config.py).

SPAI's build path consumes a yacs ``CfgNode``. Rather than drag a yacs runtime
dependency into the vendored net, this module reproduces the EXACT scalar set the
published ``spai.pth`` checkpoint was trained/inferred with (configs/spai.yaml
merged over config.py defaults, commit 8ff7b3b) as a small dotted-attribute
namespace. Only the keys the build path reads are defined.

Every value must match the model the checkpoint was dumped from — see
PROVENANCE.md. Verified against the checkpoint itself:
  cls_vector_dim = 6*12 + PROJECTION_DIM(1024) = 1096   (norm is (1096,))
  mlp_ratio=3  -> cls_head.head.0 is (1096*3, 1096) = (3288,1096)
  attn_embed_dim=1536, num_heads=12 -> dim_head=128      (patch_aggregator (12,1,128))
  frequencies_mask 224x224, masking_radius=16
  pos_embed (1,197,768) -> patch 16, img 224
"""

from __future__ import annotations


class _CN(object):
    """Dotted-attribute node; also supports the few .get()/frozen accesses."""

    def __init__(self, **kw):
        self.__dict__["_frozen"] = False
        for k, v in kw.items():
            setattr(self, k, v)

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError(f"config node is readonly: {name}")
        self.__dict__[name] = value

    def get(self, key, default=None):
        return getattr(self, key, default)

    def clone(self):
        return self

    def freeze(self):
        self.__dict__["_frozen"] = True
        return self

    def defrost(self):
        self.__dict__["_frozen"] = False
        return self

    def __repr__(self):
        return "<spai-config>"


def build_inference_config() -> _CN:
    """Return the dotted config reproducing the SPAI checkpoint's architecture."""
    CONFIG = _CN()
    CONFIG.MODEL_WEIGHTS = "mfm"
    CONFIG.DATA = _CN(IMG_SIZE=224)
    CONFIG.MODEL = _CN(
        TYPE="vit",
        NAME="finetune",
        NUM_CLASSES=2,
        DROP_RATE=0.0,
        SID_DROPOUT=0.5,
        SID_APPROACH="freq_restoration",
        RESOLUTION_MODE="arbitrary",
        FEATURE_EXTRACTION_BATCH=400,
    )
    CONFIG.MODEL.DROP_PATH_RATE = 0.1
    CONFIG.MODEL.VIT = _CN(
        PATCH_SIZE=16,
        IN_CHANS=3,
        EMBED_DIM=768,
        DEPTH=12,
        NUM_HEADS=12,
        MLP_RATIO=4,
        QKV_BIAS=True,
        INIT_VALUES=None,
        USE_APE=True,
        USE_RPB=False,
        USE_SHARED_RPB=False,
        USE_MEAN_POOLING=True,
        USE_INTERMEDIATE_LAYERS=True,
        INTERMEDIATE_LAYERS=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        PROJECTION_DIM=1024,
        PROJECTION_LAYERS=2,
        PATCH_PROJECTION=True,
        PATCH_PROJECTION_PER_FEATURE=True,
        FEATURES_PROCESSOR="rine",
        PATCH_POOLING="mean",
    )
    CONFIG.MODEL.FRE = _CN(
        MASKING_RADIUS=16,
        PROJECTOR_LAST_LAYER_ACTIVATION_TYPE=None,
        ORIGINAL_IMAGE_FEATURES_BRANCH=True,
        DISABLE_RECONSTRUCTION_SIMILARITY=False,
    )
    CONFIG.MODEL.PATCH_VIT = _CN(
        PATCH_STRIDE=224,
        ATTN_EMBED_DIM=1536,
        NUM_HEADS=12,
        MINIMUM_PATCHES=4,
    )
    CONFIG.MODEL.CLS_HEAD = _CN(MLP_RATIO=3)
    CONFIG.TRAIN = _CN(MODE="supervised")
    CONFIG.TEST = _CN(ORIGINAL_RESOLUTION=True)
    CONFIG.freeze()
    return CONFIG