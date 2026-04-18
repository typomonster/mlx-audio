"""Supertonic-2 text encoder (6.8M params) in MLX.

    text_ids [B,T]
      → char_embed [B,T,256] * mask
      → 6× ConvNeXt(dim=256, hidden=1024, k=5)
      → 4× VITS post-norm self-attn + FFN (Shaw rel-pos, 4h × 64d)
      → global residual (attn output + convnext output)
      → 2× cross-attention to style_ttl (LARoPE K pre-baked at export time)
          • parallel residual: both cross1 and cross2 add to x_pre
      → final LayerNorm → transpose to channel-first [B, 256, T]
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .duration_predictor import (
    EPS_LN,
    ConvNeXtBlock,
    FFNRelu,
    RelPosSelfAttn,
)


class LARoPECrossAttn(nn.Module):
    """Cross-attention with LARoPE-rotated K pre-baked as a constant.

    Supertonic-2's text encoder hard-codes the length of ``style_ttl`` to 50,
    so the LARoPE-rotated K tensor (``tanh(Wk(K_const))``) is computed once
    at export time and stored as a frozen ``[heads, 1, head_dim, 50]``
    buffer. At runtime this block is therefore a standard cross-attention
    with Q from text, V from style, and a static K.
    """

    DIM = 256
    HEADS = 2
    HEAD_DIM = 128
    STYLE_LEN = 50

    def __init__(self):
        super().__init__()
        self.W_query = nn.Linear(self.DIM, self.DIM, bias=True)
        self.W_value = nn.Linear(self.DIM, self.DIM, bias=True)
        self.out_fc = nn.Linear(self.DIM, self.DIM, bias=True)
        self.K = mx.zeros((self.HEADS, 1, self.HEAD_DIM, self.STYLE_LEN))

    def _to_heads(self, x: mx.array) -> mx.array:
        B, L, _ = x.shape
        return x.reshape(B, L, self.HEADS, self.HEAD_DIM).transpose(2, 0, 1, 3)

    def body(self, x_q: mx.array, style: mx.array, mask: mx.array) -> mx.array:
        B, T, _ = x_q.shape
        q = self._to_heads(self.W_query(x_q))  # [H, B, T, D]
        v = self._to_heads(self.W_value(style))  # [H, B, 50, D]
        scale = 1.0 / (self.DIM ** 0.5)
        logits = (q @ self.K) * scale  # [H, B, T, 50]
        attn = mx.softmax(logits, axis=-1)
        attn = attn * mask[None, :, :, :]  # zero padded Q rows
        out = attn @ v  # [H, B, T, D]
        out = out.transpose(1, 2, 0, 3).reshape(B, T, self.DIM)
        return self.out_fc(out) * mask

    def __call__(self, x: mx.array, style: mx.array, mask: mx.array) -> mx.array:
        return x + self.body(x, style, mask)


class TextEncoder(nn.Module):
    DIM = 256
    CN_HIDDEN = 1024
    FFN_HIDDEN = 1024
    ATTN_HEADS = 4
    ATTN_HEAD_DIM = 64
    N_CONVNEXT = 6
    N_SELF_ATTN = 4

    def __init__(self, vocab_size: int = 163):
        super().__init__()
        self.embedding = mx.zeros((vocab_size, self.DIM))
        self.convnext = [
            ConvNeXtBlock(self.DIM, self.CN_HIDDEN, kernel=5)
            for _ in range(self.N_CONVNEXT)
        ]
        self.attn = [
            RelPosSelfAttn(self.DIM, self.ATTN_HEADS, self.ATTN_HEAD_DIM)
            for _ in range(self.N_SELF_ATTN)
        ]
        self.ffn = [
            FFNRelu(self.DIM, self.FFN_HIDDEN) for _ in range(self.N_SELF_ATTN)
        ]
        self.norm1 = [nn.LayerNorm(self.DIM, eps=EPS_LN) for _ in range(self.N_SELF_ATTN)]
        self.norm2 = [nn.LayerNorm(self.DIM, eps=EPS_LN) for _ in range(self.N_SELF_ATTN)]
        self.cross1 = LARoPECrossAttn()
        self.cross2 = LARoPECrossAttn()
        self.final_norm = nn.LayerNorm(self.DIM, eps=EPS_LN)

    def _self_attn_stack(self, x: mx.array, m: mx.array) -> mx.array:
        for blk in self.convnext:
            x = blk(x, m)
        mm = m.squeeze(-1)
        attn_mask = mm[:, None, None, :] * mm[:, None, :, None]
        x_conv = x
        for i in range(self.N_SELF_ATTN):
            y = self.attn[i](x, attn_mask)
            x = self.norm1[i](x + y)
            y = self.ffn[i](x, m)
            x = self.norm2[i](x + y)
            x = x * m
        return x + x_conv

    def __call__(
        self,
        text_ids: mx.array,  # [B, T] int64
        style_ttl: mx.array,  # [B, 50, 256]
        text_mask: mx.array,  # [B, 1, T]
    ) -> mx.array:
        """Returns ``text_emb`` in channel-first layout ``[B, 256, T]``."""
        x = self.embedding[text_ids]
        m = text_mask.transpose(0, 2, 1)
        x = x * m
        x = self._self_attn_stack(x, m)
        # Parallel-residual cross-attn: both branches add to the same pre tensor.
        x_pre = x
        x1 = self.cross1(x_pre, style_ttl, m)
        x = x_pre + self.cross2.body(x1, style_ttl, m)
        x = self.final_norm(x)
        return x.transpose(0, 2, 1)
