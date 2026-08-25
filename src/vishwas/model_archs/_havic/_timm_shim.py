"""Minimal timm shim for the vendored HAVIC backend (Vishwas task 1.4a).

The upstream HAVIC reference (JielunPeng/HAVIC, MIT) imports five symbols from
``timm``; the Vishwas real-torch tree has no timm installed.  This module
re-implements exactly those symbols with IDENTICAL constructor signatures and
IDENTICAL state-dict parameter names as the classic (pre-1.0) timm layouts the
checkpoint was trained with, so checkpoint keys such as
``AudioVisualInteractionModule.blocks.0.self_attn_video.attn.qkv.weight`` and
``...mlp.fc1.weight`` match 1:1 (verified against
``tests/fixtures/havic_best_ft_key_shapes.txt``).

Symbol-by-symbol decisions (see also havic.py module docstring):
  * ``Attention``  — timm vision_transformer.Attention, CLASSIC fused layout:
                     ``qkv`` is a single ``nn.Linear(dim, 3*dim, bias=qkv_bias)``
                     plus ``proj``.  The best_ft checkpoint carries
                     ``attn.qkv.weight``/``attn.qkv.bias``/``attn.proj.*`` which
                     only matches this layout (newer timm would emit separate
                     ``q_bias``/``v_bias`` params and a bias-less qkv).
  * ``Mlp``        — timm vision_transformer.Mlp: ``fc1``/``fc2`` Linear pair +
                     GELU + dropout.  Checkpoint: ``mlp.fc1.weight``/``mlp.fc2.*``.
  * ``DropPath``   — standard stochastic-depth residual drop (no parameters).
  * ``to_2tuple``  — trivial int->tuple helper.
  * ``PatchEmbed`` — classic timm PatchEmbed (``proj`` Conv2d).  Imported by the
                     vendored interaction_modules but never instantiated there;
                     provided for import fidelity only.
  * ``Block``      — classic timm ViT block (``norm1``/``attn``/``norm2``/``mlp``).
                     Same: imported but never instantiated by HAVIC_FT's path.

NOTE: the audio/visual encoders do NOT come from timm — audio_modules.py and
utils.py each define their own Attention/Block variants (with separate
``q_bias``/``v_bias`` params, matching their checkpoint keys).  Only the
AudioVisualInteractionModule (AVIM) actually instantiates the shim classes.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "to_2tuple",
    "DropPath",
    "Mlp",
    "Attention",
    "PatchEmbed",
    "Block",
]


def to_2tuple(x):
    """timm.models.layers.to_2tuple — wrap non-tuples into a 2-tuple."""
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return (x, x)


class DropPath(nn.Module):
    """Per-sample stochastic depth (timm.models.layers.DropPath, no params)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor)
        return x.div(keep_prob) * random_tensor


class Mlp(nn.Module):
    """timm vision_transformer.Mlp — keys: fc1.{weight,bias}, fc2.{weight,bias}."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Classic timm vision_transformer.Attention (fused-qkv layout).

    Keys: ``qkv.weight``, ``qkv.bias`` (present iff ``qkv_bias=True``),
    ``proj.weight``, ``proj.bias``.  This exact layout is what the best_ft
    checkpoint's AVIM blocks carry.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PatchEmbed(nn.Module):
    """Classic timm PatchEmbed — keys: ``proj.weight``, ``proj.bias``.

    Imported by vendored interaction_modules for fidelity; HAVIC's actual
    audio patch embed is a local class in audio_modules.py and the visual one
    is PatchEmbedding3d in utils.py.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class Block(nn.Module):
    """Classic timm ViT block — keys: norm1.*, attn.*, norm2.*, mlp.*.

    Imported (but not instantiated) by vendored interaction_modules; provided
    for import fidelity.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None,
                 drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
