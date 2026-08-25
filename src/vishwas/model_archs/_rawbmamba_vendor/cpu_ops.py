"""Pure-PyTorch CPU shims for RawBMamba's CUDA-only dependencies.

Vendored from state-spaces/mamba reference implementations (selective_scan_ref,
RMSNorm) + a causal depthwise conv1d reference. Inference-only (no backward):
~10-50x slower than the fused CUDA kernels, which is fine for per-clip
analysis on the CPU-only vishwas host.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                      delta_softplus=False, return_last_state=False):
    """Reference selective scan (inference).

    u: (B D L), delta: (B D L), A: (D N) real, B/C: (B N L) variable,
    D: (D,), z: (B D L) gate or None.
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3

    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum("bdl,dn,bdl->bdln", delta, B, u)
    else:
        B = B.float()
        deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)

    if is_variable_C:
        C = C.float()

    x = A.new_zeros((batch, dim, dstate))
    outs = []
    for t in range(u.shape[2]):
        x = deltaA[:, :, t] * x + deltaB_u[:, :, t]
        if not is_variable_C:
            y = torch.einsum("bdn,dn->bd", x, C)
        elif C.dim() == 3:
            y = torch.einsum("bdn,bn->bd", x, C[:, :, t])
        else:
            y = torch.einsum("bdn,bgn->bd", x, C[:, :, :, t])
        if D is not None:
            y = y + u[:, :, t] * D.float()
        outs.append(y)
    out = torch.stack(outs, dim=2)
    last_state = x
    if z is not None:
        out = out * F.silu(z.float())
    if return_last_state:
        return out.to(dtype_in), last_state
    return out.to(dtype_in)


def causal_conv1d_fn(x, weight, bias=None, activation=None, seq_idx=None):
    """Causal depthwise conv1d reference.

    x: (B, D, L); weight: (D, 1, k); bias: (D,). Left-pad k-1 so output length
    matches input (strict causality).
    """
    k = weight.shape[-1]
    if weight.dim() == 2:  # caller passes (D, k) via rearrange("d 1 w -> d w")
        weight = weight.unsqueeze(1)
    xp = F.pad(x, (k - 1, 0))
    y = F.conv1d(xp, weight, bias=bias, groups=x.shape[1])
    if activation == "silu" or activation == "swish":
        y = F.silu(y)
    elif activation == "gelu":
        y = F.gelu(y)
    return y


def causal_conv1d_update(x, conv_state, weight, bias=None, activation=None):
    """Step update used by generation paths: x (B, D), conv_state (B, D, k)."""
    k = weight.shape[-1]
    conv_state.copy_(torch.cat([conv_state[:, :, 1:], x.unsqueeze(-1)], dim=-1))
    y = (conv_state * weight.squeeze(1)).sum(-1)
    if bias is not None:
        y = y + bias
    if activation == "silu":
        y = F.silu(y)
    return y


class RMSNorm(nn.Module):
    """Reference RMSNorm replacing the Triton kernel."""

    def __init__(self, dim: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self._is_cpu_shim = True
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x, residual=None, **kwargs):
        # Non-fused contract (Block.forward plain path): return a SINGLE tensor.
        if residual is not None:
            x = x + residual
        dtype = x.dtype
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed.to(dtype) * self.weight.to(dtype)


def layer_norm_fn(x, weight, bias, residual=None, eps=1e-5, **kwargs):
    if residual is not None:
        x = x + residual
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps), residual


def rms_norm_fn(x, weight, bias, residual=None, eps=1e-5, **kwargs):
    return RMSNorm(weight.shape[0], eps).to(x.device)(x)[0], residual


# Names the vendored code imports but must NOT use on CPU:
mamba_inner_fn = None  # forces slow path (see patched mamba_simple.py)
