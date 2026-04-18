# Supertonic-2 audio samples

Generated with [`typomonster/supertonic-2-mlx`](https://huggingface.co/typomonster/supertonic-2-mlx) via `mlx_audio.tts.load(...)` on Apple Silicon. Click a 🎧 to open the raw WAV in your browser.

Supported languages: `en`, `ko`, `es`, `pt`, `fr`. Voices: `M1`–`M5` (male), `F1`–`F5` (female). Output: 44.1 kHz mono WAV.

## Long-form samples

### English

> "Hello. Today, I would like to talk about one of the long-standing philosophical debates: 'Which came first, the chicken or the egg?' This question may seem like simple curiosity, but in fact, it is a topic that allows us to deeply explore how we understand life, evolution, cause, and effect."

**M3** (male) — [en_M3.wav](https://github.com/typomonster/mlx-audio/raw/main/docs/supertonic/raw/en_M3.wav)

<video src="https://github.com/typomonster/mlx-audio/raw/main/docs/supertonic/en_M3.mp4" controls width="480"></video>

**F1** (female) — [en_F1.wav](https://github.com/typomonster/mlx-audio/raw/main/docs/supertonic/raw/en_F1.wav)

<video src="https://github.com/typomonster/mlx-audio/raw/main/docs/supertonic/en_F1.mp4" controls width="480"></video>

### Korean

> "안녕하세요. 오늘 저는 오랜 철학적 논쟁 중 하나인, '달걀이 먼저인가, 닭이 먼저인가'라는 주제에 대해 이야기하려 합니다. 이 질문은 단순한 호기심처럼 보이지만, 사실 우리가 생명과 진화, 원인과 결과를 어떻게 이해하는지 깊이 탐구할 수 있는 주제입니다."

**M1** (male) — [ko_M1.wav](https://github.com/typomonster/mlx-audio/raw/main/docs/supertonic/raw/ko_M1.wav)

<video src="https://github.com/typomonster/mlx-audio/raw/main/docs/supertonic/ko_M1.mp4" controls width="480"></video>

**F3** (female) — [ko_F3.wav](https://github.com/typomonster/mlx-audio/raw/main/docs/supertonic/raw/ko_F3.wav)

<video src="https://github.com/typomonster/mlx-audio/raw/main/docs/supertonic/ko_F3.mp4" controls width="480"></video>

> The play controls come from `<video>` tags embedding audio-only MP4s (AAC). GitHub renders `<video>` inline at the top level but sanitizes `<audio>` tags — and it also breaks `<video>` tags placed inside markdown tables, so each sample sits as its own block element.

## Short benchmark samples

Used for the post-warmup performance measurement in the top-level README.

| Input | Voice | Lang | Sample |
| ----- | ----- | ---- | ------ |
| `"Hello world."`                     | M1 | en | [🎧 short_en_M1.wav](https://github.com/typomonster/mlx-audio/raw/main/docs/supertonic/raw/short_en_M1.wav) |
| `"오늘 아침 공원을 산책했어요."`     | F1 | ko | [🎧 short_ko_F1.wav](https://github.com/typomonster/mlx-audio/raw/main/docs/supertonic/raw/short_ko_F1.wav) |

## Reproduce

```python
import numpy as np, soundfile as sf
from mlx_audio.tts import load

model = load("typomonster/supertonic-2-mlx")

cases = [
    ("en_M3.wav", "Hello. Today, I would like to talk about ...", "M3", "en"),
    ("en_F1.wav", "Hello. Today, I would like to talk about ...", "F1", "en"),
    ("ko_M1.wav", "안녕하세요. 오늘 저는 오랜 철학적 논쟁 ...", "M1", "ko"),
    ("ko_F3.wav", "안녕하세요. 오늘 저는 오랜 철학적 논쟁 ...", "F3", "ko"),
]
for fname, text, voice, lang in cases:
    pieces = [np.asarray(r.audio) for r in model.generate(text, voice=voice, lang=lang)]
    sf.write(fname, np.concatenate(pieces) if len(pieces) > 1 else pieces[0], model.sample_rate)
```

Defaults used: `steps=5` (Euler), `speed=1.05`, `seed=0`.
