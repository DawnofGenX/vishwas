# CPU-vendored fairseq-style wav2vec 2.0 frontend (XLSR-300M class).
#
# Vendored from facebookresearch/fairseq @ a54021305d6b3c (MIT) — the exact
# commit XLSR-Mamba pins — files fairseq/models/wav2vec/wav2vec2.py,
# fairseq/modules/{layer_norm,same_pad,transpose_last,grad_multiply,
# fp32_group_norm,multihead_attention,gumbel_vector_quantizer}.py,
# fairseq/{utils,data/data_utils}.py. Deviations are marked "CPU-vendored:".
#
# Scope: inference-only feature extraction for the SSL_Anti-spoofing/XLSR-Mamba
# call contract `model(wav, mask=False, features_only=True)["x"]` with
# post-norm (layer_norm_first=False) configuration. Pre-training machinery
# (masking application path, negative sampling, contrastive loss) is kept only
# where the checkpoint's state dict requires the modules to exist (quantizer,
# project_q, final_proj) — they are never exercised at inference.
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# tiny fairseq modules (verbatim behaviour)
# ---------------------------------------------------------------------------

class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-2, -1)


class SamePad(nn.Module):
    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


class Fp32LayerNorm(nn.LayerNorm):
    """LayerNorm in fp32 (upstream verbatim)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class Fp32GroupNorm(nn.GroupNorm):
    """GroupNorm in fp32 (upstream verbatim)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


def _fairseq_layer_norm(normalized_shape, eps=1e-5, elementwise_affine=True):
    # CPU-vendored: upstream returns apex FusedLayerNorm when apex is present;
    # plain nn.LayerNorm is numerically equivalent and always available.
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


def pad_to_multiple(x, multiple, dim=-1, value=0):
    # fairseq/models/wav2vec/utils.py (verbatim)
    if x is None:
        return None, 0
    tsz = x.size(dim)
    m = tsz / multiple
    remainder = math.ceil(m) * multiple - tsz
    if m.is_integer():
        return x, 0
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(x, (*pad_offset, 0, remainder), value=value), remainder


def buffered_arange(max):
    # fairseq/utils.py (verbatim)
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


def is_xla_tensor(tensor):
    return torch.is_tensor(tensor) and tensor.device.type == "xla"


def index_put(tensor, indices, value):
    # fairseq/utils.py (verbatim minus XLA branch — unreachable here)
    tensor[indices] = value
    return tensor


def init_bert_params(module):
    """BERT-style init used by TransformerEncoder.apply (upstream verbatim)."""
    def normal_(data):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


# ---------------------------------------------------------------------------
# MultiheadAttention — fairseq/modules/multihead_attention.py
#
# CPU-vendored: incremental decoding / ONNX / TPU / bias_k branches removed;
# the fast path is now an explicit SDPA call with the same q/k/v/out weights
# and identical masking semantics (additive -inf on padding columns).
# ---------------------------------------------------------------------------

class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
    ):
        super().__init__()
        assert q_noise == 0.0  # quant-noise unused at inference
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = float(dropout)

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert not self.self_attention or self.qkv_same_dim

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        assert not add_bias_kv and not add_zero_attn  # unused by wav2vec2 cfgs

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state=None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        src_len = tgt_len

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        else:  # pragma: no cover - cross-attn unused by wav2vec2
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q = q * self.scaling

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        k = (
            k.contiguous()
            .view(-1, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        v = (
            v.contiguous()
            .view(-1, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_float = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(
            attn_weights_float, p=self.dropout_module, training=self.training
        )

        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        attn_weights_out: Optional[Tensor] = None
        if need_weights:
            attn_weights_out = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                attn_weights_out = attn_weights_out.mean(dim=0)

        return attn, attn_weights_out


# ---------------------------------------------------------------------------
# GumbelVectorQuantizer — fairseq/modules/gumbel_vector_quantizer.py
# (exists so the pre-trained codebook loads; inference path never runs it)
# ---------------------------------------------------------------------------

class GumbelVectorQuantizer(nn.Module):
    def __init__(
        self,
        dim,
        num_vars,
        temp,
        groups,
        combine_groups,
        vq_dim,
        time_first,
        activation=nn.GELU(),
        weight_proj_depth=1,
        weight_proj_factor=1,
    ):
        super().__init__()
        self.groups = groups
        self.combine_groups = combine_groups
        self.input_dim = dim
        self.num_vars = num_vars
        self.time_first = time_first

        assert vq_dim % groups == 0
        var_dim = vq_dim // groups
        num_groups = groups if not combine_groups else 1

        self.vars = nn.Parameter(torch.FloatTensor(1, num_groups * num_vars, var_dim))
        nn.init.uniform_(self.vars)

        if weight_proj_depth > 1:  # pragma: no cover - unused by this config
            def block(input_dim, output_dim):
                return nn.Sequential(nn.Linear(input_dim, output_dim), activation)

            inner_dim = self.input_dim * weight_proj_factor
            self.weight_proj = nn.Sequential(
                *[
                    block(self.input_dim if i == 0 else inner_dim, inner_dim)
                    for i in range(weight_proj_depth - 1)
                ],
                nn.Linear(inner_dim, groups * num_vars),
            )
        else:
            self.weight_proj = nn.Linear(self.input_dim, groups * num_vars)
            nn.init.normal_(self.weight_proj.weight, mean=0, std=1)
            nn.init.zeros_(self.weight_proj.bias)

        if isinstance(temp, str):
            import ast
            temp = ast.literal_eval(temp)
        assert len(temp) == 3
        self.max_temp, self.min_temp, self.temp_decay = temp
        self.curr_temp = self.max_temp
        self.codebook_indices = None

    def forward(self, x, produce_targets=False):  # pragma: no cover - training only
        result = {"num_vars": self.num_vars * self.groups}
        if not self.time_first:
            x = x.transpose(1, 2)
        bsz, tsz, fsz = x.shape
        x = x.reshape(-1, fsz)
        x = self.weight_proj(x)
        x = x.view(bsz * tsz * self.groups, -1)
        _, k = x.max(-1)
        hard_x = (
            x.new_zeros(*x.shape)
            .scatter_(-1, k.view(-1, 1), 1.0)
            .view(bsz * tsz, self.groups, -1)
        )
        x = hard_x.view(bsz * tsz, -1)
        vars_ = self.vars
        if self.combine_groups:
            vars_ = vars_.repeat(1, self.groups, 1)
        if produce_targets:
            result["targets"] = (
                x.view(bsz * tsz * self.groups, -1)
                .argmax(dim=-1)
                .view(bsz, tsz, self.groups)
                .detach()
            )
        x = x.unsqueeze(-1) * vars_
        x = x.view(bsz * tsz, self.groups, self.num_vars, -1)
        x = x.sum(-2)
        x = x.view(bsz, tsz, -1)
        if not self.time_first:
            x = x.transpose(1, 2)
        result["x"] = x
        return result


# ---------------------------------------------------------------------------
# ConvFeatureExtractionModel + TransformerEncoder + Wav2Vec2Model
# (fairseq/models/wav2vec/wav2vec2.py)
# ---------------------------------------------------------------------------

class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
    ):
        super().__init__()

        def block(n_in, n_out, k, stride, is_layer_norm=False, is_group_norm=False,
                  conv_bias=False):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (is_layer_norm and is_group_norm) == False

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(n_out, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(n_out, n_out, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            dim, k, stride = cl
            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):
        # BxT -> BxCxT
        x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)
        return x


class TransformerSentenceEncoderLayer(nn.Module):
    """Post-norm transformer layer (fairseq verbatim; activation_fn gelu fixed)."""

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "gelu",
        layer_norm_first: bool = False,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        assert activation_fn == "gelu"
        self.activation_fn = F.gelu
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        self.self_attn_layer_norm = _fairseq_layer_norm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)
        self.final_layer_norm = _fairseq_layer_norm(self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
    ):
        residual = x

        if self.layer_norm_first:  # pragma: no cover - this model is post-norm
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x, key=x, value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn = self.self_attn(
                query=x, key=x, value=x,
                key_padding_mask=self_attn_padding_mask,
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, attn


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.dropout = args["dropout"]
        self.embedding_dim = args["encoder_embed_dim"]
        self.required_seq_len_multiple = args.get("required_seq_len_multiple", 1)

        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=args["conv_pos"],
            padding=args["conv_pos"] // 2,
            groups=args["conv_pos_groups"],
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (args["conv_pos"] * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        # fairseq verbatim: weight_norm(..., dim=2). NOTE this yields
        # weight_g of shape (1, 1, K) because norm_except_dim keeps the
        # normalized axis as singletons — verified against the checkpoint's
        # pos_conv.0.weight_g (1, 1, 128).
        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args["conv_pos"]), nn.GELU())

        layers = []
        for _ in range(args["encoder_layers"]):
            layer = TransformerSentenceEncoderLayer(
                embedding_dim=self.embedding_dim,
                ffn_embedding_dim=args["encoder_ffn_embed_dim"],
                num_attention_heads=args["encoder_attention_heads"],
                dropout=self.dropout,
                attention_dropout=args["attention_dropout"],
                activation_dropout=args["activation_dropout"],
                activation_fn=args["activation_fn"],
                layer_norm_first=args["layer_norm_first"],
            )
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        self.layer_norm_first = args["layer_norm_first"]
        self.layer_norm = _fairseq_layer_norm(self.embedding_dim)
        self.layerdrop = args["encoder_layerdrop"]

        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None, layer=None):
        x, layer_results = self.extract_features(x, padding_mask, layer)
        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)
        return x, layer_results

    def extract_features(self, x, padding_mask=None, tgt_layer=None):
        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        # pad to the sequence length dimension
        x, pad_length = pad_to_multiple(
            x, self.required_seq_len_multiple, dim=-2, value=0
        )
        if pad_length > 0 and padding_mask is None:
            padding_mask = x.new_zeros((x.size(0), x.size(1)), dtype=torch.bool)
            padding_mask[:, -pad_length:] = True
        else:
            padding_mask, _ = pad_to_multiple(
                padding_mask, self.required_seq_len_multiple, dim=-1, value=True
            )
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        r = None
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
                if tgt_layer is not None:
                    if pad_length > 0:
                        layer_results.append(
                            (
                                x[:-pad_length],
                                z[:, :-pad_length, :-pad_length] if z is not None else z,
                            )
                        )
                    else:
                        layer_results.append((x, z))
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        # undo padding
        if pad_length > 0:
            x = x[:, :-pad_length]

        return x, layer_results


class Wav2Vec2Config(dict):
    """Minimal stand-in for fairseq's FairseqDataclass config.

    Attribute access falls back to fairseq defaults so callers can pass only
    the overrides they care about.
    """

    _DEFAULTS = {
        "extractor_mode": "default",
        "encoder_layers": 12,
        "encoder_embed_dim": 768,
        "encoder_ffn_embed_dim": 3072,
        "encoder_attention_heads": 12,
        "activation_fn": "gelu",
        "dropout": 0.1,
        "attention_dropout": 0.1,
        "activation_dropout": 0.0,
        "encoder_layerdrop": 0.0,
        "dropout_input": 0.0,
        "dropout_features": 0.0,
        "final_dim": 0,
        "layer_norm_first": False,
        "conv_feature_layers":
            "[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
        "conv_bias": False,
        "logit_temp": 0.1,
        "quantize_targets": False,
        "quantize_input": False,
        "same_quantizer": False,
        "target_glu": False,
        "feature_grad_mult": 1.0,
        "quantizer_depth": 1,
        "quantizer_factor": 3,
        "latent_vars": 320,
        "latent_groups": 2,
        "latent_dim": 0,
        "mask_length": 10,
        "mask_prob": 0.65,
        "mask_selection": "static",
        "mask_other": 0,
        "no_mask_overlap": False,
        "mask_min_space": 1,
        "mask_channel_length": 10,
        "mask_channel_prob": 0.0,
        "mask_channel_before": False,
        "mask_channel_selection": "static",
        "mask_channel_other": 0,
        "no_mask_channel_overlap": False,
        "mask_channel_min_space": 1,
        "num_negatives": 100,
        "negatives_from_everywhere": False,
        "cross_sample_negatives": 0,
        "codebook_negatives": 0,
        "conv_pos": 128,
        "conv_pos_groups": 16,
        "latent_temp": (2, 0.5, 0.999995),
        "checkpoint_activations": False,
        "required_seq_len_multiple": 1,
        "crop_seq_to_multiple": 1,
    }

    def __init__(self, overrides: Optional[dict] = None):
        super().__init__()
        merged = dict(self._DEFAULTS)
        if overrides:
            merged.update({k: v for k, v in overrides.items() if v is not None})
        self.update(merged)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - mirrors attr semantics
            raise AttributeError(name) from exc


class Wav2Vec2Model(nn.Module):
    """CPU-vendored fairseq Wav2Vec2Model.

    Inference-relevant paths preserved verbatim (feature extractor, post_extract_proj,
    dropout_input/features, mask application when mask=True — off at eval,
    encoder, final projection when features_only=False); pre-training-only
    negative-sampling/contrastive machinery removed (never called by the
    anti-spoofing frontend contract).
    """

    def __init__(self, cfg: Wav2Vec2Config):
        super().__init__()
        self.cfg = cfg

        feature_enc_layers = eval(cfg.conv_feature_layers)  # noqa: S307 (static cfg string)
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim and not cfg.quantize_input
            else None
        )

        self.crop_seq_to_multiple = cfg.crop_seq_to_multiple

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_before = cfg.mask_channel_before
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult

        self.input_quantizer = None

        self.logit_temp = cfg.logit_temp

        # CPU-vendored: fairseq falls back to encoder_embed_dim when
        # final_dim <= 0, but the XLSR-300M checkpoint family stores the
        # pre-training heads at 768 (quantizer.vars var_dim*groups = 384*2,
        # project_q/final_proj in/out = 768). Mirror that geometry.
        final_dim = cfg.final_dim if cfg.final_dim > 0 else 768

        # CPU-vendored deviation from fairseq: fairseq builds self.quantizer
        # only when cfg.quantize_targets is set. This tree serves XLSR-300M
        # anti-spoofing checkpoints, which always carry quantizer.* weights
        # (vars (1,640,384) -> num_vars=320 x groups=2, var_dim=vq_dim/2=384,
        # weight_proj (640,512)), even though spoof frontends never execute
        # it; build unconditionally with vq_dim == final_dim == 768 so strict
        # load passes. Inference never calls forward() on it.
        if True:
            vq_dim = final_dim
            self.quantizer = GumbelVectorQuantizer(
                dim=self.embed,
                num_vars=cfg.latent_vars,
                temp=cfg.latent_temp,
                groups=cfg.latent_groups,
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
                weight_proj_depth=cfg.quantizer_depth,
                weight_proj_factor=cfg.quantizer_factor,
            )
            # CPU-vendored: project_q.weight stored as (768, 768) =
            # Linear(vq_dim=768, final_dim=768), i.e. fairseq's
            # quantize_targets construction with vq_dim == final_dim.
            self.project_q = nn.Linear(vq_dim, final_dim)
        else:
            self.project_q = nn.Linear(self.embed, final_dim)

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = _fairseq_layer_norm(self.embed)

        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )

        self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)

    # -- inference contract -------------------------------------------------

    def apply_mask(self, x, padding_mask, mask_indices=None, mask_channel_indices=None):
        """Masking application only (span SAMPLING lives in fairseq.data; at
        inference mask_prob=0 / mask=False makes this a no-op passthrough)."""
        B, T, C = x.shape
        if self.mask_channel_prob > 0 and self.mask_channel_before:
            raise NotImplementedError("channel-before masking unused by this model")
        if self.mask_prob > 0 and mask_indices is not None:
            x = index_put(x, mask_indices, self.mask_emb)
        if self.mask_channel_prob > 0 and not self.mask_channel_before:
            if mask_channel_indices is None:
                raise NotImplementedError("random channel masking unused at inference")
            x = index_put(x, mask_channel_indices, 0)
        return x, mask_indices

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)

        conv_cfg_list = eval(self.cfg.conv_feature_layers)  # noqa: S307
        for i in range(len(conv_cfg_list)):
            input_lengths = _conv_out_length(
                input_lengths, conv_cfg_list[i][1], conv_cfg_list[i][2]
            )
        return input_lengths.to(torch.long)

    def forward(
        self,
        source,
        padding_mask=None,
        mask=True,
        features_only=False,
        layer=None,
        mask_indices=None,
        mask_channel_indices=None,
        padding_count=None,
    ):

        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None and padding_mask.any():
            input_lengths = (1 - padding_mask.long()).sum(-1)
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)

            padding_mask = torch.zeros(
                features.shape[:2], dtype=features.dtype, device=features.device
            )
            padding_mask[
                (
                    torch.arange(padding_mask.shape[0], device=padding_mask.device),
                    output_lengths - 1,
                )
            ] = 1
            padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        else:
            padding_mask = None

        time_steps_to_drop = features.size(1) % self.crop_seq_to_multiple
        if time_steps_to_drop != 0:
            features = features[:, :-time_steps_to_drop]
            unmasked_features = unmasked_features[:, :-time_steps_to_drop]
            if padding_mask is not None:
                padding_mask = padding_mask[:, :-time_steps_to_drop]

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        if mask:
            x, mask_indices = self.apply_mask(
                features,
                padding_mask,
                mask_indices=mask_indices,
                mask_channel_indices=mask_channel_indices,
            )
        else:
            x = features
            mask_indices = None

        x, layer_results = self.encoder(x, padding_mask=padding_mask, layer=layer)

        if features_only:
            return {
                "x": x,
                "padding_mask": padding_mask,
                "features": unmasked_features,
                "layer_results": layer_results,
            }

        # CPU-vendored: full pretraining head (negative sampling + contrastive
        # logits) removed — the anti-spoofing frontends only ever call this with
        # features_only=True. Kept: final_proj application so state-dict keys
        # exist and any future logits use stays honest.
        if not is_xla_tensor(x):
            x = x[mask_indices].view(x.size(0), -1, x.size(-1)) if mask_indices is not None else x
        x = self.final_proj(x)
        result = {
            "x": x,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
        }
        return result

    def extract_features(self, source, padding_mask, mask=False, layer=None):
        res = self.forward(source, padding_mask, mask=mask, features_only=True, layer=layer)
        return res

    def remove_pretraining_modules(self):
        self.quantizer = None
        self.project_q = None
        self.target_glu = None
        self.final_proj = None
