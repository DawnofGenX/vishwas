# CPU-vendored shims for XLSR-Mamba's CUDA-only dependencies.
#
# The upstream repo (swagshaw/XLSR-Mamba, MIT) imports two CUDA-bound packages:
#   * mamba_ssm.ops.selective_scan_interface.selective_scan_fn  (+ mamba_inner_fn)
#   * causal_conv1d.causal_conv1d_fn
# Neither ships prebuilt wheels usable here, and the RawBMamba integration
# already proved pure-PyTorch reference implementations are numerically sound
# for per-clip inference (~10-50x slower than fused kernels, irrelevant at one
# clip per job). selective_scan_ref is vendored from state-spaces/mamba
# (MIT/BSD-3); causal-conv reference follows the RawBMamba cpu_ops recipe.
#
# Inference-only: no backward pass is implemented or needed.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                      delta_softplus=False, return_last_state=False):
    """Reference selective scan (inference).

    u: (B D L), delta: (B D L), A: (D N) real, B/C: (B N L) variable,
    D: (D,), z: (B D L) gate or None.

    CPU-vendored: replaces mamba_ssm's fused CUDA kernel with the loop
    recurrence from state-spaces/mamba's selective_scan_ref.
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

    x = torch.zeros((batch, dim, dstate), device=u.device, dtype=u.dtype)
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

    x: (B, D, L); weight: (D, k) or (D, 1, k); bias: (D,). Left-pads k-1 so
    output length matches input (strict causality) — same contract as the
    causal_conv1d package entry point Mamba.forward calls.

    CPU-vendored: replaces causal_conv1d_cuda.
    """
    k = weight.shape[-1]
    if weight.dim() == 2:
        # Caller passes (D, k) via rearrange("d 1 w -> d w"); conv1d needs 3-D.
        weight = weight.unsqueeze(1)
    xp = F.pad(x, (k - 1, 0))
    y = F.conv1d(xp, weight, bias=bias, groups=x.shape[1])
    if activation == "silu" or activation == "swish":
        y = F.silu(y)
    elif activation == "gelu":
        y = F.gelu(y)
    return y


class RMSNorm(nn.Module):
    """Reference RMSNorm replacing mamba_ssm's Triton kernel.

    CPU-vendored. Carries `_is_cpu_shim` so vendored block code can recognise
    it where upstream asserted isinstance() against its own triton class.
    """

    def __init__(self, dim: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self._is_cpu_shim = True
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x, residual=None, **kwargs):
        # Non-fused contract (plain path): consume an optional residual,
        # return a SINGLE tensor.
        if residual is not None:
            x = x + residual
        dtype = x.dtype
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed.to(dtype) * self.weight.to(dtype)


# Names the vendored modules import but must never resolve to a real kernel on
# this box; importing them stays honest by binding None instead.
mamba_inner_fn = None          # forces Mamba down the reference slow path
selective_state_update = None  # single-token generation only; unused for clips
causal_conv1d_update = None    # ditto
