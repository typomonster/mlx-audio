"""Supertonic-2 vector estimator (33M params) — flow-matching denoiser.

    noisy_latent [B, 144, T_lat]
      → proj_in (Conv1d 144→512 k=1) → [B, T_lat, 512]
      → 24 heterogeneous blocks (4 cycles × 6 block types):
          0: ConvNeXtStack4      — 4 non-causal ConvNeXt, dilations [1,2,4,8]
          1: TimeFiLM             — additive time conditioning (64→512)
          2: ConvNeXt1            — single ConvNeXt block
          3: LARoPECrossAttnText  — cross-attn to text, LARoPE (rotated at runtime)
          4: ConvNeXt1
          5: CrossAttnStyle       — cross-attn to style (K from a learned constant)
      → 4× ConvNeXt (last stack)
      → proj_out (Conv1d 512→144 k=1) → velocity
      → Euler step: denoised = noisy + velocity / total_step
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .duration_predictor import EPS_LN, ConvNeXtBlock

DIM = 512
CONVNEXT_HIDDEN = 1024
CONVNEXT_K = 5
TIME_EMB_DIM = 64
TIME_MLP_HIDDEN = 256
LATENT_CH = 144

ATTN_KV_DIM = 256
STYLE_HEADS = 2
STYLE_HEAD_DIM = 128
TEXT_HEADS = 4
TEXT_HEAD_DIM = 64
STYLE_LEN = 50

BLOCKS_PER_CYCLE = 6
N_CYCLES = 4
BLOCK_TYPES = ("stack4", "time", "cn1", "text_attn", "cn1", "style_attn")


def mish(x: mx.array) -> mx.array:
    return x * mx.tanh(mx.logaddexp(x, mx.array(0.0, dtype=x.dtype)))


class ConvNeXtStack4(nn.Module):
    """Four dilated ConvNeXt blocks (dilations 1,2,4,8)."""

    def __init__(self):
        super().__init__()
        self.blocks = [
            ConvNeXtBlock(DIM, CONVNEXT_HIDDEN, kernel=CONVNEXT_K, dilation=d)
            for d in (1, 2, 4, 8)
        ]

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        for b in self.blocks:
            x = b(x, mask)
        return x


class ConvNeXt1(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = ConvNeXtBlock(DIM, CONVNEXT_HIDDEN, kernel=CONVNEXT_K, dilation=1)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        return self.block(x, mask)


class TimeFiLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(TIME_EMB_DIM, DIM, bias=True)

    def __call__(self, x: mx.array, t_emb: mx.array, mask: mx.array) -> mx.array:
        return (x + self.linear(t_emb)) * mask


class CrossAttnStyle(nn.Module):
    """Cross-attention with V from style_ttl and K from a learned constant.

    ``K_source`` is a shared learned constant across all four cycles (baked
    into the ONNX graph at ``/Expand_output_0``), projected through ``W_key``
    with a ``tanh`` on top. V comes from ``style_ttl``.
    """

    def __init__(self):
        super().__init__()
        self.W_query = nn.Linear(DIM, ATTN_KV_DIM, bias=True)
        self.W_key = nn.Linear(ATTN_KV_DIM, ATTN_KV_DIM, bias=True)
        self.W_value = nn.Linear(ATTN_KV_DIM, ATTN_KV_DIM, bias=True)
        self.out_fc = nn.Linear(ATTN_KV_DIM, DIM, bias=True)
        self.norm = nn.LayerNorm(DIM, eps=EPS_LN)
        self.K_source = mx.zeros((1, STYLE_LEN, ATTN_KV_DIM))

    def __call__(self, x: mx.array, style: mx.array, latent_mask: mx.array) -> mx.array:
        B, T_lat, _ = x.shape
        q = self.W_query(x)
        k_src = mx.broadcast_to(self.K_source, (B, STYLE_LEN, ATTN_KV_DIM))
        k = mx.tanh(self.W_key(k_src))
        v = self.W_value(style)

        def heads(t: mx.array) -> mx.array:
            B_, L_, _ = t.shape
            return t.reshape(B_, L_, STYLE_HEADS, STYLE_HEAD_DIM).transpose(2, 0, 1, 3)

        q = heads(q); k = heads(k); v = heads(v)
        scale = 1.0 / (ATTN_KV_DIM ** 0.5)
        logits = (q @ k.transpose(0, 1, 3, 2)) * scale
        attn = mx.softmax(logits, axis=-1)
        attn = attn * latent_mask[None, :, :, :]
        out = attn @ v
        out = out.transpose(1, 2, 0, 3).reshape(B, T_lat, ATTN_KV_DIM)
        out = self.out_fc(out) * latent_mask
        residual = out + x * latent_mask
        return self.norm(residual) * latent_mask


def _rope_half(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Half-rotation RoPE: split last dim in two; rotate as a pair."""
    D = x.shape[-1]
    half = D // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return mx.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


class LARoPECrossAttnText(nn.Module):
    """Length-Aware RoPE cross-attention to the text embedding.

    Q positions are normalised by ``len_lat``; K positions by ``len_text``.
    This lets the model generalise to arbitrary lengths without retraining.
    """

    def __init__(self):
        super().__init__()
        self.W_query = nn.Linear(DIM, ATTN_KV_DIM, bias=True)
        self.W_key = nn.Linear(ATTN_KV_DIM, ATTN_KV_DIM, bias=True)
        self.W_value = nn.Linear(ATTN_KV_DIM, ATTN_KV_DIM, bias=True)
        self.out_fc = nn.Linear(ATTN_KV_DIM, DIM, bias=True)
        self.theta = mx.zeros((1, 1, TEXT_HEAD_DIM // 2))
        self.increments = mx.zeros((1, 1000, 1))
        self.norm = nn.LayerNorm(DIM, eps=EPS_LN)

    def _heads(self, t: mx.array) -> mx.array:
        B_, L_, _ = t.shape
        return t.reshape(B_, L_, TEXT_HEADS, TEXT_HEAD_DIM).transpose(2, 0, 1, 3)

    def __call__(
        self,
        x: mx.array,         # [B, T_lat, 512]
        text_emb: mx.array,  # [B, T_text, 256]
        latent_mask: mx.array,  # [B, T_lat, 1]
        text_mask: mx.array,    # [B, T_text, 1]
        len_lat: mx.array,      # [B]
        len_text: mx.array,     # [B]
    ) -> mx.array:
        B, T_lat, _ = x.shape
        _, T_text, _ = text_emb.shape

        q = self._heads(self.W_query(x))
        k = self._heads(self.W_key(text_emb))
        v = self._heads(self.W_value(text_emb))

        inc_q = self.increments[0, :T_lat, 0]
        inc_k = self.increments[0, :T_text, 0]
        pos_q = inc_q[None, :] / len_lat[:, None]  # [B, T_lat]
        pos_k = inc_k[None, :] / len_text[:, None]  # [B, T_text]
        th = self.theta.reshape(-1)

        freq_q = pos_q[..., None] * th[None, None, :]
        freq_k = pos_k[..., None] * th[None, None, :]
        cos_q = mx.cos(freq_q)[None, :, :, :]
        sin_q = mx.sin(freq_q)[None, :, :, :]
        cos_k = mx.cos(freq_k)[None, :, :, :]
        sin_k = mx.sin(freq_k)[None, :, :, :]
        q = _rope_half(q, cos_q, sin_q)
        k = _rope_half(k, cos_k, sin_k)

        scale = 1.0 / (ATTN_KV_DIM ** 0.5)
        logits = (q @ k.transpose(0, 1, 3, 2)) * scale

        tm = text_mask[..., 0][None, :, None, :]  # [1, B, 1, T_text]
        neg = mx.array(-1e4, dtype=logits.dtype)
        logits = mx.where(tm.astype(mx.bool_), logits, neg)
        attn = mx.softmax(logits, axis=-1)
        attn = attn * latent_mask[None, :, :, :]

        out = attn @ v
        out = out.transpose(1, 2, 0, 3).reshape(B, T_lat, ATTN_KV_DIM)
        out = self.out_fc(out) * latent_mask
        residual = out + x * latent_mask
        return self.norm(residual) * latent_mask


class TimeEncoder(nn.Module):
    """Sinusoidal encoding of t = step/total_step, then a 2-layer MLP."""

    def __init__(self):
        super().__init__()
        # Default frequencies; overwritten at load-time from the ONNX constant.
        self.freqs = mx.power(10000.0, -mx.arange(32, dtype=mx.float32) / 32.0)
        self.linear1 = nn.Linear(TIME_EMB_DIM, TIME_MLP_HIDDEN, bias=True)
        self.linear2 = nn.Linear(TIME_MLP_HIDDEN, TIME_EMB_DIM, bias=True)

    def __call__(self, current_step: mx.array, total_step: mx.array) -> mx.array:
        t = (current_step / total_step) * 1000.0
        scaled = t[:, None] * self.freqs[None, :]
        emb = mx.concatenate([mx.sin(scaled), mx.cos(scaled)], axis=-1)
        h = self.linear1(emb)
        h = mish(h)
        return self.linear2(h)


class VectorEstimator(nn.Module):
    N_CYCLES = N_CYCLES
    BLOCK_TYPES = BLOCK_TYPES

    def __init__(self):
        super().__init__()
        self.proj_in = nn.Conv1d(LATENT_CH, DIM, kernel_size=1, bias=False)
        self.time_encoder = TimeEncoder()
        self.blocks = []
        for _ in range(N_CYCLES):
            for t in BLOCK_TYPES:
                if t == "stack4":
                    self.blocks.append(ConvNeXtStack4())
                elif t == "time":
                    self.blocks.append(TimeFiLM())
                elif t == "cn1":
                    self.blocks.append(ConvNeXt1())
                elif t == "text_attn":
                    self.blocks.append(LARoPECrossAttnText())
                elif t == "style_attn":
                    self.blocks.append(CrossAttnStyle())
        self.last_convnext = [
            ConvNeXtBlock(DIM, CONVNEXT_HIDDEN, kernel=CONVNEXT_K, dilation=1)
            for _ in range(4)
        ]
        self.proj_out = nn.Conv1d(DIM, LATENT_CH, kernel_size=1, bias=False)

    def __call__(
        self,
        noisy_latent: mx.array,  # [B, 144, T_lat]
        text_emb: mx.array,      # [B, 256, T_text]  (channel-first, from text encoder)
        style_ttl: mx.array,     # [B, 50, 256]
        latent_mask: mx.array,   # [B, 1, T_lat]
        text_mask: mx.array,     # [B, 1, T_text]
        current_step: mx.array,  # [B]
        total_step: mx.array,    # [B]
    ) -> mx.array:
        x = noisy_latent.transpose(0, 2, 1)  # [B, T_lat, 144]
        x = self.proj_in(x)                   # [B, T_lat, 512]
        lat_m = latent_mask.transpose(0, 2, 1)
        x = x * lat_m

        text_m = text_mask.transpose(0, 2, 1)
        text_emb_cl = text_emb.transpose(0, 2, 1)

        t_emb = self.time_encoder(current_step, total_step)  # [B, 64]
        t_emb_cl = t_emb[:, None, :]

        len_lat = latent_mask.sum(axis=(1, 2))
        len_text = text_mask.sum(axis=(1, 2))

        block_types_flat = [t for _ in range(N_CYCLES) for t in BLOCK_TYPES]
        for blk, bt in zip(self.blocks, block_types_flat):
            if bt in ("stack4", "cn1"):
                x = blk(x, lat_m)
            elif bt == "time":
                x = blk(x, t_emb_cl, lat_m)
            elif bt == "text_attn":
                x = blk(x, text_emb_cl, lat_m, text_m, len_lat, len_text)
            elif bt == "style_attn":
                x = blk(x, style_ttl, lat_m)

        for b in self.last_convnext:
            x = b(x, lat_m)

        velocity = self.proj_out(x) * lat_m
        velocity = velocity * (1.0 / total_step[:, None, None])
        velocity_cf = velocity.transpose(0, 2, 1)
        return (noisy_latent + velocity_cf) * latent_mask
