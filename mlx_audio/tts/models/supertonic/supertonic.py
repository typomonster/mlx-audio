"""Supertonic-2 TTS — Model + ModelConfig + generate() for mlx-audio.

Pipeline per utterance:
    dur = duration_predictor(text_ids, style_dp)
    emb = text_encoder(text_ids, style_ttl)
    x   = randn(latent_shape)
    for step in range(N):
        x = vector_estimator(x, emb, style_ttl, …, step, N)   # bakes one Euler step
    wav = vocoder(x)

`generate()` handles multi-chunk text, wraps each chunk with
``<{lang}>…</{lang}>`` via ``preprocess_text``, tokenises with the unicode
indexer loaded at ``post_load_hook`` time, and yields one
``GenerationResult`` per chunk with the concatenated audio for long inputs.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import BaseModelArgs, GenerationResult
from .duration_predictor import DurationPredictor
from .text import (
    SUPPORTED_LANGUAGES,
    chunk_text,
    encode_text,
    preprocess_text,
    resolve_chunk_max_len,
)
from .text_encoder import TextEncoder
from .vector_estimator import VectorEstimator
from .vocoder import Vocoder


@dataclass
class ModelConfig(BaseModelArgs):
    """Runtime config; see also the preserved ``tts.json`` in the ckpt dir."""

    model_type: str = "supertonic"
    sample_rate: int = 44_100
    base_chunk_size: int = 512            # ae.base_chunk_size
    chunk_compress_factor: int = 6        # ttl.chunk_compress_factor
    latent_dim: int = 24                  # ttl.latent_dim (per-chunk)
    vocab_size: int = 163
    style_len: int = 50                   # style_ttl length
    default_steps: int = 5                # Euler steps
    default_speed: float = 1.05           # duration scaling (>1 speaks faster)
    languages: List[str] = field(default_factory=lambda: list(SUPPORTED_LANGUAGES))


class Model(nn.Module):
    """Supertonic-2 end-to-end wrapper for mlx-audio."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        if isinstance(config, dict):
            config = ModelConfig.from_dict(config)
        self.config = config
        self.duration_predictor = DurationPredictor(vocab_size=config.vocab_size)
        self.text_encoder = TextEncoder(vocab_size=config.vocab_size)
        self.vector_estimator = VectorEstimator()
        self.vocoder = Vocoder()

        # Populated by post_load_hook from the checkpoint dir.
        self._unicode_indexer: Optional[Any] = None
        self._voices: Dict[str, Dict[str, mx.array]] = {}

    # ------------------------------------------------------------------
    # Loading hooks (called by mlx_audio.utils.base_load_model)
    # ------------------------------------------------------------------
    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        # Checkpoints already use MLX-native keys/layout, so no remap here.
        return weights

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        model_path = Path(model_path)
        model._load_unicode_indexer(model_path / "unicode_indexer.json")
        voices_dir = model_path / "voice_styles"
        if voices_dir.exists():
            for vf in sorted(voices_dir.glob("*.json")):
                model._load_voice(vf)
        return model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    @property
    def available_voices(self) -> List[str]:
        return sorted(self._voices.keys())

    def generate(
        self,
        text: str,
        voice: str = "M1",
        lang: str = "en",
        speed: Optional[float] = None,
        steps: Optional[int] = None,
        seed: int = 0,
        chunk_max_len: Optional[int] = None,
        silence_between_chunks: float = 0.3,
        **_: Any,
    ) -> Iterator[GenerationResult]:
        """Synthesise ``text`` chunk-by-chunk. One ``GenerationResult`` per chunk."""
        if not text or not text.strip():
            return
        if self._unicode_indexer is None:
            raise RuntimeError("unicode_indexer.json not loaded — check checkpoint dir")
        if voice not in self._voices:
            raise ValueError(
                f"voice {voice!r} not found. Available: {self.available_voices}"
            )
        if lang not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"language {lang!r} not supported. Choose from: {list(SUPPORTED_LANGUAGES)}"
            )

        steps = int(steps if steps is not None else self.config.default_steps)
        speed = float(speed if speed is not None else self.config.default_speed)
        max_len = resolve_chunk_max_len(lang, chunk_max_len)
        chunks = chunk_text(text, max_len) or [text.strip()]

        voice_data = self._voices[voice]
        style_ttl_np = voice_data["style_ttl"]  # [B, 50, 256]
        style_dp_np = voice_data["style_dp"]    # [B, 8, 16]

        sr = self.config.sample_rate
        base_chunk = self.config.base_chunk_size
        ccf = self.config.chunk_compress_factor
        chunk_wav = base_chunk * ccf
        lat_dim = self.config.latent_dim * ccf

        silence = (
            np.zeros(int(silence_between_chunks * sr), dtype=np.float32)
            if silence_between_chunks > 0
            else None
        )

        peak_memory_fn = getattr(mx, "get_peak_memory", None)

        for i, chunk in enumerate(chunks):
            t0 = time.time()
            audio_np = self._synthesize_one(
                chunk,
                lang,
                style_ttl_np,
                style_dp_np,
                steps,
                speed,
                seed=seed + i,
                chunk_wav=chunk_wav,
                lat_dim=lat_dim,
            )
            if silence is not None and i < len(chunks) - 1:
                audio_np = np.concatenate([audio_np, silence], axis=0)
            elapsed = time.time() - t0

            samples = int(audio_np.shape[0])
            duration_s = samples / sr if sr else 0.0
            rtf = (elapsed / duration_s) if duration_s > 0 else 0.0
            d_h = int(duration_s // 3600)
            d_m = int((duration_s % 3600) // 60)
            d_s = int(duration_s % 60)
            d_ms = int((duration_s - int(duration_s)) * 1000)
            peak_mem = (peak_memory_fn() / 1e9) if peak_memory_fn else 0.0

            yield GenerationResult(
                audio=mx.array(audio_np),
                samples=samples,
                sample_rate=sr,
                segment_idx=i,
                token_count=len(chunk),
                audio_duration=f"{d_h:02d}:{d_m:02d}:{d_s:02d}.{d_ms:03d}",
                real_time_factor=round(rtf, 3),
                prompt={
                    "tokens": len(chunk),
                    "tokens-per-sec": round(len(chunk) / elapsed, 2) if elapsed > 0 else 0,
                },
                audio_samples={
                    "samples": samples,
                    "samples-per-sec": round(samples / elapsed, 2) if elapsed > 0 else 0,
                },
                processing_time_seconds=elapsed,
                peak_memory_usage=peak_mem,
            )
            mx.clear_cache() if hasattr(mx, "clear_cache") else None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _synthesize_one(
        self,
        text_chunk: str,
        lang: str,
        style_ttl_np: np.ndarray,
        style_dp_np: np.ndarray,
        steps: int,
        speed: float,
        *,
        seed: int,
        chunk_wav: int,
        lat_dim: int,
    ) -> np.ndarray:
        preprocessed = preprocess_text(text_chunk, lang)
        text_ids_np = encode_text(preprocessed, self._unicode_indexer)
        B, T = text_ids_np.shape
        text_mask_np = np.ones((B, 1, T), dtype=np.float32)

        text_ids = mx.array(text_ids_np)
        text_mask = mx.array(text_mask_np)
        style_ttl = mx.array(style_ttl_np)
        style_dp = mx.array(style_dp_np)

        dur_arr = self.duration_predictor(text_ids, style_dp, text_mask)
        dur = np.asarray(dur_arr) / speed  # [B]

        text_emb = self.text_encoder(text_ids, style_ttl, text_mask)

        sr = self.config.sample_rate
        wav_len = int(dur[0] * sr)
        lat_len = max(1, (wav_len + chunk_wav - 1) // chunk_wav)

        rng = np.random.RandomState(int(seed))
        noisy = rng.randn(B, lat_dim, lat_len).astype(np.float32)
        latent_mask_np = np.ones((B, 1, lat_len), dtype=np.float32)
        xt = mx.array(noisy * latent_mask_np)
        latent_mask = mx.array(latent_mask_np)
        total_step = mx.array([float(steps)] * B, dtype=mx.float32)
        for step_idx in range(steps):
            cur = mx.array([float(step_idx)] * B, dtype=mx.float32)
            xt = self.vector_estimator(
                xt, text_emb, style_ttl, latent_mask, text_mask, cur, total_step
            )

        wav = np.asarray(self.vocoder(xt))
        return wav[0, : int(sr * dur[0])].astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # Asset loaders (internal)
    # ------------------------------------------------------------------
    def _load_unicode_indexer(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"unicode_indexer.json not found at {path}. "
                "The checkpoint directory must contain unicode_indexer.json "
                "and voice_styles/ alongside the four safetensors files."
            )
        with open(path, encoding="utf-8") as f:
            self._unicode_indexer = json.load(f)

    def _load_voice(self, path: Path) -> None:
        with open(path, encoding="utf-8") as f:
            style = json.load(f)
        ttl = np.asarray(style["style_ttl"]["data"], dtype=np.float32).reshape(
            style["style_ttl"]["dims"]
        )
        dp = np.asarray(style["style_dp"]["data"], dtype=np.float32).reshape(
            style["style_dp"]["dims"]
        )
        self._voices[path.stem] = {"style_ttl": ttl, "style_dp": dp}
