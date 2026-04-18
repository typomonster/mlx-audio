"""Supertonic-2 TTS (Supertone) — MLX port.

Four-stage pipeline (duration predictor → text encoder → flow-matching
vector estimator → Vocos-style vocoder). 5 languages (en, ko, es, pt, fr),
10 preset voices (M1–M5, F1–F5), 44.1 kHz output.

Checkpoint: https://huggingface.co/typomonster/supertonic-2-mlx
"""
from .supertonic import Model, ModelConfig

__all__ = ["Model", "ModelConfig"]
