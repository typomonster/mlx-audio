"""Supertonic-2 duration predictor (33 k params) in MLX.

Pipeline (all tensors channel-last inside the module):

    text_ids [B,T]
      → char_embed [B,T,64] * mask
      → prepend sentence_token → [B, T+1, 64]
      → 6× ConvNeXt(dim=64, hidden=256, k=5, sym-edge pad=2)
      → 2× VITS post-norm (self-attn + FFN), Shaw rel-pos
      → global residual (attn_encoder output + convnext output)
      → take the sentence-token slot → proj_out → [B, 64]
      → concat flattened style_dp [B, 128] → [B, 192]
      → Linear(192→128) → PReLU → Linear(128→1) → exp → duration [B]
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

# LayerNorm eps is 1e-6 throughout Supertonic-2 (NOT the MLX default 1e-5).
EPS_LN = 1e-6
# Exact (non-tanh) GELU: x * 0.5 * (1 + erf(x / sqrt(2))).
GELU_C = 2 ** -0.5


def gelu_exact(x: mx.array) -> mx.array:
    return x * 0.5 * (1.0 + mx.erf(x * GELU_C))


def pad_sym_edge(x: mx.array, pad: int) -> mx.array:
    """Symmetric replicate-edge pad on the time axis (axis=1 for [B, T, C])."""
    if pad == 0:
        return x
    left = mx.broadcast_to(x[:, :1, :], (x.shape[0], pad, x.shape[2]))
    right = mx.broadcast_to(x[:, -1:, :], (x.shape[0], pad, x.shape[2]))
    return mx.concatenate([left, x, right], axis=1)


class ConvNeXtBlock(nn.Module):
    """Shared ConvNeXt block. Used by DP (dim=64), text encoder (dim=256),
    and vector estimator (dim=512)."""

    def __init__(self, dim: int, hidden: int, kernel: int = 5, dilation: int = 1):
        super().__init__()
        self.dim = dim
        self.dilation = dilation
        self.pad = dilation * (kernel - 1) // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=kernel, padding=0, dilation=dilation,
            groups=dim, bias=True,
        )
        self.norm = nn.LayerNorm(dim, eps=EPS_LN)
        # Pointwise Conv1d k=1 ≡ Linear on the channel dim.
        self.pwconv1 = nn.Linear(dim, hidden, bias=True)
        self.pwconv2 = nn.Linear(hidden, dim, bias=True)
        self.gamma = mx.zeros((dim,))

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        residual = x
        y = pad_sym_edge(x, self.pad)
        y = self.dwconv(y)
        y = y * mask
        y = self.norm(y)
        y = self.pwconv1(y)
        y = gelu_exact(y)
        y = self.pwconv2(y)
        y = self.gamma * y
        y = residual + y
        return y * mask


class RelPosSelfAttn(nn.Module):
    """Multi-head self-attention with Shaw-style relative position bias,
    VITS post-norm. Shared between DP (2h×32) and text encoder (4h×64)."""

    def __init__(self, dim: int, heads: int, head_dim: int, window: int = 4):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim
        self.window = window
        self.conv_q = nn.Linear(dim, dim, bias=True)
        self.conv_k = nn.Linear(dim, dim, bias=True)
        self.conv_v = nn.Linear(dim, dim, bias=True)
        self.conv_o = nn.Linear(dim, dim, bias=True)
        # [1, 2*window+1, head_dim]
        self.emb_rel_k = mx.zeros((1, 2 * window + 1, head_dim))
        self.emb_rel_v = mx.zeros((1, 2 * window + 1, head_dim))

    @staticmethod
    def _rel_to_abs(x: mx.array) -> mx.array:
        """[B, h, L, 2L-1] → [B, h, L, L]."""
        B, h, L, _ = x.shape
        x = mx.concatenate([x, mx.zeros((B, h, L, 1), dtype=x.dtype)], axis=-1)
        x_flat = x.reshape(B, h, L * 2 * L)
        x_flat = mx.concatenate([x_flat, mx.zeros((B, h, L - 1), dtype=x.dtype)], axis=-1)
        x_final = x_flat.reshape(B, h, L + 1, 2 * L - 1)
        return x_final[:, :, :L, L - 1 :]

    @staticmethod
    def _abs_to_rel(x: mx.array) -> mx.array:
        """[B, h, L, L] → [B, h, L, 2L-1] (inverse of _rel_to_abs)."""
        B, h, L, _ = x.shape
        x = mx.concatenate([x, mx.zeros((B, h, L, L - 1), dtype=x.dtype)], axis=-1)
        x_flat = x.reshape(B, h, L * (2 * L - 1))
        x_flat = mx.concatenate([mx.zeros((B, h, L), dtype=x.dtype), x_flat], axis=-1)
        x_final = x_flat.reshape(B, h, L, 2 * L)
        return x_final[:, :, :, 1:]

    def _slice_rel_emb(self, rel: mx.array, length: int) -> mx.array:
        """rel [1, 2W+1, d] → [1, 2L-1, d] (slice or zero-pad to match L)."""
        pad_l = max(length - (self.window + 1), 0)
        if pad_l > 0:
            zero = mx.zeros((1, pad_l, rel.shape[-1]), dtype=rel.dtype)
            padded = mx.concatenate([zero, rel, zero], axis=1)
        else:
            padded = rel
        start = max(self.window + 1 - length, 0)
        end = start + 2 * length - 1
        return padded[:, start:end]

    def __call__(self, x: mx.array, attn_mask: mx.array) -> mx.array:
        B, T, C = x.shape
        q = self.conv_q(x).reshape(B, T, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.conv_k(x).reshape(B, T, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.conv_v(x).reshape(B, T, self.heads, self.head_dim).transpose(0, 2, 1, 3)

        scale = 1.0 / (self.head_dim ** 0.5)
        logits = (q @ k.transpose(0, 1, 3, 2)) * scale

        rel_k = self._slice_rel_emb(self.emb_rel_k, T)
        rel_logits = (q @ rel_k[None, :, :, :].transpose(0, 1, 3, 2)) * scale
        rel_logits = self._rel_to_abs(rel_logits)
        logits = logits + rel_logits

        neg_inf = mx.array(-1e4, dtype=logits.dtype)
        logits = mx.where(attn_mask.astype(mx.bool_), logits, neg_inf)
        attn = mx.softmax(logits, axis=-1)

        out = attn @ v
        rel_v = self._slice_rel_emb(self.emb_rel_v, T)
        rel_weights = self._abs_to_rel(attn)
        out = out + rel_weights @ rel_v[None, :, :, :]

        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.conv_o(out)


class FFNRelu(nn.Module):
    """FFN used inside Supertonic-2 self-attn stacks. ReLU, not GELU, per the
    ONNX export."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.conv_1 = nn.Linear(dim, hidden, bias=True)
        self.conv_2 = nn.Linear(hidden, dim, bias=True)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        y = self.conv_1(x * mask)
        y = mx.maximum(y, 0)
        y = y * mask
        y = self.conv_2(y)
        return y * mask


class DurationPredictor(nn.Module):
    DIM = 64
    HEADS = 2
    HEAD_DIM = 32
    WINDOW = 4
    CN_HIDDEN = 256
    FFN_HIDDEN = 256
    N_CONVNEXT = 6
    N_ATTN = 2

    def __init__(self, vocab_size: int = 163, style_n: int = 8, style_dim: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.style_flat_dim = style_n * style_dim  # = 128

        self.embedding = mx.zeros((vocab_size, self.DIM))
        self.sentence_token = mx.zeros((1, 1, self.DIM))

        self.convnext = [
            ConvNeXtBlock(self.DIM, self.CN_HIDDEN, kernel=5)
            for _ in range(self.N_CONVNEXT)
        ]
        self.attn = [
            RelPosSelfAttn(self.DIM, self.HEADS, self.HEAD_DIM, window=self.WINDOW)
            for _ in range(self.N_ATTN)
        ]
        self.ffn = [FFNRelu(self.DIM, self.FFN_HIDDEN) for _ in range(self.N_ATTN)]
        self.norm1 = [nn.LayerNorm(self.DIM, eps=EPS_LN) for _ in range(self.N_ATTN)]
        self.norm2 = [nn.LayerNorm(self.DIM, eps=EPS_LN) for _ in range(self.N_ATTN)]
        self.proj_out = nn.Linear(self.DIM, self.DIM, bias=False)

        self.head_lin1 = nn.Linear(self.DIM + self.style_flat_dim, 128, bias=True)
        self.head_prelu = mx.zeros((1,))
        self.head_lin2 = nn.Linear(128, 1, bias=True)

    def __call__(
        self,
        text_ids: mx.array,  # [B, T] int64
        style_dp: mx.array,  # [B, 8, 16]
        text_mask: mx.array,  # [B, 1, T]
    ) -> mx.array:
        B, T = text_ids.shape
        x = self.embedding[text_ids]  # [B, T, C]
        m = text_mask.transpose(0, 2, 1)  # [B, T, 1]
        x = x * m

        st = mx.broadcast_to(self.sentence_token, (B, 1, self.DIM))
        x = mx.concatenate([st, x], axis=1)
        m = mx.concatenate([mx.ones((B, 1, 1), dtype=m.dtype), m], axis=1)
        x = x * m

        for blk in self.convnext:
            x = blk(x, m)

        mm = m.squeeze(-1)
        attn_mask = mm[:, None, None, :] * mm[:, None, :, None]  # [B, 1, T', T']
        x_conv = x
        for i in range(self.N_ATTN):
            y = self.attn[i](x, attn_mask)
            x = self.norm1[i](x + y)
            y = self.ffn[i](x, m)
            x = self.norm2[i](x + y)
            x = x * m
        x = x + x_conv

        sent = x[:, :1, :]
        sent = self.proj_out(sent).squeeze(1)  # [B, C]

        style_flat = style_dp.reshape(B, -1)
        h = mx.concatenate([sent, style_flat], axis=-1)

        h = self.head_lin1(h)
        h = mx.where(h >= 0, h, self.head_prelu * h)
        dur = self.head_lin2(h).reshape(B)
        return mx.exp(dur)
