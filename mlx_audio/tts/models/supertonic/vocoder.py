"""Supertonic-2 vocoder (25.3M params) — Vocos-style 14 Hz latent → 44.1 kHz.

    latent [B, 144, T]
      → /= scale                               # inverse of encoder normaliser
      → reshape [B, 24, T*6]                   # de-compress along time
      → (* std + mean)                         # de-normalise
      → embed Conv1d(24→512, k=7, causal)
      → 10× ConvNeXt(dim=512, hidden=2048, k=7, dilations [1,2,4,1,2,4,1,1,1,1])
      → BatchNorm1d(512)                       # eval-time: running stats only
      → head: Conv1d(512→2048, k=3, causal) → PReLU → Conv1d(2048→512, k=1)
      → transpose + flatten → waveform [B, T*6*512]
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

EPS_LN = 1e-6
EPS_BN = 1e-5
GELU_C = 2 ** -0.5
DILATIONS = [1, 2, 4, 1, 2, 4, 1, 1, 1, 1]


def gelu_exact(x: mx.array) -> mx.array:
    return x * 0.5 * (1.0 + mx.erf(x * GELU_C))


def pad_causal_edge(x: mx.array, pad_left: int) -> mx.array:
    """Replicate-edge pad on the left of the time axis (dim 1) — causal."""
    if pad_left == 0:
        return x
    first = x[:, :1, :]
    pad = mx.broadcast_to(first, (x.shape[0], pad_left, x.shape[2]))
    return mx.concatenate([pad, x], axis=1)


class CausalConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, kernel: int, dilation: int):
        super().__init__()
        self.dilation = dilation
        self.pad_left = (kernel - 1) * dilation
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=kernel, padding=0,
            dilation=dilation, groups=dim, bias=True,
        )
        self.norm = nn.LayerNorm(dim, eps=EPS_LN)
        self.pwconv1 = nn.Linear(dim, hidden, bias=True)
        self.pwconv2 = nn.Linear(hidden, dim, bias=True)
        self.gamma = mx.zeros((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        y = pad_causal_edge(x, self.pad_left)
        y = self.dwconv(y)
        y = self.norm(y)
        y = self.pwconv1(y)
        y = gelu_exact(y)
        y = self.pwconv2(y)
        y = self.gamma * y
        return residual + y


class BatchNormChannelLast(nn.Module):
    """BatchNorm1d in eval mode applied on the channel (last) axis."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.bias = mx.zeros((dim,))
        self.running_mean = mx.zeros((dim,))
        self.running_var = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        inv = mx.rsqrt(self.running_var + EPS_BN)
        return (x - self.running_mean) * inv * self.weight + self.bias


class Vocoder(nn.Module):
    LATENT_DIM = 24
    CHUNK_FACTOR = 6
    DIM = 512
    HIDDEN = 2048
    NUM_BLOCKS = 10

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        chunk_factor: int = CHUNK_FACTOR,
        dim: int = DIM,
        hidden: int = HIDDEN,
        num_blocks: int = NUM_BLOCKS,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.chunk_factor = chunk_factor
        self.dim = dim

        self.embed_pad = 6
        self.embed = nn.Conv1d(latent_dim, dim, kernel_size=7, padding=0, bias=True)
        self.convnext = [
            CausalConvNeXtBlock(dim, hidden, kernel=7, dilation=DILATIONS[i])
            for i in range(num_blocks)
        ]
        self.final_norm = BatchNormChannelLast(dim)

        self.head_pad = 2
        self.head1 = nn.Conv1d(dim, hidden, kernel_size=3, padding=0, bias=True)
        self.prelu = mx.zeros((1,))
        # head2 has no bias in the ONNX graph.
        self.head2 = nn.Conv1d(hidden, dim, kernel_size=1, padding=0, bias=False)

        self.latent_mean = mx.zeros((1, 1, latent_dim))
        self.latent_std = mx.ones((1, 1, latent_dim))
        self.normalizer_scale = mx.array(1.0)

    def __call__(self, latent_bct: mx.array) -> mx.array:
        """``latent_bct`` in ONNX layout ``[B, 144, T]``; returns ``[B, T*6*512]``."""
        x = latent_bct / self.normalizer_scale
        B, C, T = x.shape
        assert C == self.latent_dim * self.chunk_factor
        x = x.reshape(B, self.latent_dim, self.chunk_factor, T)
        x = x.transpose(0, 1, 3, 2).reshape(B, self.latent_dim, T * self.chunk_factor)
        x = x.transpose(0, 2, 1)  # [B, T*6, 24]
        x = x * self.latent_std + self.latent_mean

        x = pad_causal_edge(x, self.embed_pad)
        x = self.embed(x)
        for blk in self.convnext:
            x = blk(x)
        x = self.final_norm(x)

        x = pad_causal_edge(x, self.head_pad)
        x = self.head1(x)
        x = mx.where(x >= 0, x, self.prelu * x)
        x = self.head2(x)

        Tout = x.shape[1]
        return x.reshape(B, Tout * self.dim)
