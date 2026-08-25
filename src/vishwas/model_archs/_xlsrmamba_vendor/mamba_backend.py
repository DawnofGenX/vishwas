# CPU-vendored Mamba block for the XLSR-Mamba-LA backend (MIT).
#
# Vendored from swagshaw/XLSR-Mamba mamba_blocks.py + state-spaces/mamba
# mamba_simple.py, both MIT. Deviations from upstream are marked "CPU-vendored:"
# in place. The checkpoint this tree must strict-load was trained with plain
# per-direction Blocks (mixer+norm only): it carries NO BiBlock.LL_hidden keys,
# so BiBlock is intentionally not vendored.
from __future__ import annotations

import math
import sys as _sys
import os as _os
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

_VENDOR_DIR = _os.path.dirname(__file__)
if _VENDOR_DIR not in _sys.path:
    _sys.path.insert(0, _VENDOR_DIR)

# CPU-vendored: original imports `mamba_ssm.modules.mamba_simple` (CUDA-bound)
# and `mamba_ssm.ops.triton.layernorm.RMSNorm` (Triton-bound). Both replaced by
# pure-PyTorch shims from cpu_shims.py.
from cpu_shims import (
    RMSNorm,
    causal_conv1d_fn,
    selective_scan_fn,
)


class Mamba(nn.Module):
    """Mamba mixer block (reference path only).

    CPU-vendored: upstream dispatches to fused CUDA kernels via
    use_fast_path/mamba_inner_fn; those branches are removed outright here so
    they can never be taken. Numerics of the retained reference path match
    state-spaces/mamba's slow path.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank="auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize dt projection to preserve variance at init (upstream verbatim).
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(dt_init)

        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization (upstream verbatim).
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """hidden_states: (B, L, D) -> same shape."""
        batch, seqlen, dim = hidden_states.shape

        # CPU-vendored: inference_params / step() decoding paths removed —
        # vishwas always scores whole clips, never token-by-token generation.

        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        x, z = xz.chunk(2, dim=1)
        # Short causal convolution.
        assert self.activation in ("silu", "swish")
        x = causal_conv1d_fn(
            x=x,
            weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
            bias=self.conv1d.bias,
            activation=self.activation,
        )

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl, dt_rank+2*N)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        y = rearrange(y, "b d l -> b l d")
        return self.out_proj(y)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_inner, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.d_inner, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state


class Block(nn.Module):
    """Prenorm residual block: Add -> Norm -> Mixer (upstream verbatim minus the
    fused_add_norm branch)."""

    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False,
        residual_in_fp32=False,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        # CPU-vendored: fused_add_norm requires Triton kernels; hard-disable.
        self.fused_add_norm = fused_add_norm and False
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None, inference_params=None,
    ):
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:  # pragma: no cover - unreachable by construction
            hidden_states, residual = F.layer_norm(
                hidden_states, self.norm.normalized_shape,
                self.norm.weight, getattr(self.norm, "bias", None),
                getattr(self.norm, "eps", 1e-5),
            ), residual
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon: float = 1e-5,
    rms_norm: bool = False,
    residual_in_fp32: bool = False,
    fused_add_norm: bool = False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


def _init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True,
                  n_residuals_per_layer=1):
    """GPT-2-style init (upstream verbatim)."""
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MixerModel(nn.Module):
    """Dual-column bidirectional Mamba trunk + attention pooling + head.

    Matches XLSR-Mamba's checkpointed revision: two parallel columns of plain
    Blocks (forward/backward over the flipped sequence), shared final norm on
    each column's residual, per-column attention pooling, concat + LL fusion,
    classifier. (The repo's newer BiBlock refactor adds per-block LL_hidden
    projections that the released LA checkpoint does NOT contain.)
    """

    def __init__(
        self,
        d_model: int,
        n_layer: int,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        if_bidirectional: bool = True,
        initializer_cfg=None,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.if_bidirectional = if_bidirectional
        # CPU-vendored: see Block.__init__ — fused add/norm never available here.
        self.fused_add_norm = fused_add_norm and False

        self.forward_layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )
        if self.if_bidirectional:
            self.backward_layers = nn.ModuleList(
                [
                    create_block(
                        d_model,
                        ssm_cfg=ssm_cfg,
                        norm_epsilon=norm_epsilon,
                        rms_norm=rms_norm,
                        residual_in_fp32=residual_in_fp32,
                        fused_add_norm=fused_add_norm,
                        layer_idx=i,
                        **factory_kwargs,
                    )
                    for i in range(n_layer)
                ]
            )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.f_attention_pool = nn.Linear(d_model, 1)
        if self.if_bidirectional:
            self.b_attention_pool = nn.Linear(d_model, 1)
            self.LL = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, x, inference_params=None):
        hidden_states = self.dropout(x)

        if not self.if_bidirectional:  # pragma: no cover - unused by this checkpoint
            residual = None
            for layer in self.forward_layers:
                hidden_states, residual = layer(hidden_states, residual, inference_params)
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
            pooled = torch.matmul(
                F.softmax(self.f_attention_pool(hidden_states), dim=1).transpose(-1, -2),
                hidden_states,
            ).squeeze(-2)
        else:
            f_hidden_states = hidden_states
            b_hidden_states = hidden_states.flip([1])
            f_residual, b_residual = None, None
            for layer in self.forward_layers:
                f_hidden_states, f_residual = layer(f_hidden_states, f_residual, inference_params)
            for layer in self.backward_layers:
                b_hidden_states, b_residual = layer(b_hidden_states, b_residual, inference_params)

            # Per-column Add+Norm (upstream non-fused branch).
            f_residual = (f_hidden_states + f_residual) if f_residual is not None else f_hidden_states
            f_hidden_states = self.norm_f(f_residual.to(dtype=self.norm_f.weight.dtype))
            b_residual = (b_hidden_states + b_residual) if b_residual is not None else b_hidden_states
            b_hidden_states = self.norm_f(b_residual.to(dtype=self.norm_f.weight.dtype))

            f_pooled = torch.matmul(
                F.softmax(self.f_attention_pool(f_hidden_states), dim=1).transpose(-1, -2),
                f_hidden_states,
            ).squeeze(-2)
            b_pooled = torch.matmul(
                F.softmax(self.b_attention_pool(b_hidden_states), dim=1).transpose(-1, -2),
                b_hidden_states,
            ).squeeze(-2)
            pooled = torch.cat((f_pooled, b_pooled), dim=1)

        out = self.LL(pooled)
        out = self.dropout(out)
        return self.classifier(out)
