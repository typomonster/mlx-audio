# Supertonic-2 audio samples

Generated with [`typomonster/supertonic-2-mlx`](https://huggingface.co/typomonster/supertonic-2-mlx) via `mlx_audio.tts.load(...)` on Apple Silicon. Click a 🎧 to open the raw WAV in your browser.

Supported languages: `en`, `ko`, `es`, `pt`, `fr`. Voices: `M1`–`M5` (male), `F1`–`F5` (female). Output: 44.1 kHz mono WAV.

## Long-form samples

### English

> "Hello. Today, I would like to talk about one of the long-standing philosophical debates: 'Which came first, the chicken or the egg?' This question may seem like simple curiosity, but in fact, it is a topic that allows us to deeply explore how we understand life, evolution, cause, and effect."

**M3** (male) — [en_M3.wav](https://github.com/typomonster/mlx-audio/raw/main/docs/supertonic/raw/en_M3.wav)

https://github.com/user-attachments/assets/80245c60-76bb-4c83-a1dd-84078681432a

**F1** (female) — [en_F1.wav](https://github.com/typomonster/mlx-audio/raw/main/docs/supertonic/raw/en_F1.wav)

https://github.com/user-attachments/assets/78c0a3d6-0586-4a41-bb0f-8d4083cbfcbd

### Korean

> "안녕하세요. 오늘 저는 오랜 철학적 논쟁 중 하나인, '달걀이 먼저인가, 닭이 먼저인가'라는 주제에 대해 이야기하려 합니다. 이 질문은 단순한 호기심처럼 보이지만, 사실 우리가 생명과 진화, 원인과 결과를 어떻게 이해하는지 깊이 탐구할 수 있는 주제입니다."

**M1** (male) — [ko_M1.wav](https://github.com/typomonster/mlx-audio/raw/main/docs/supertonic/raw/ko_M1.wav)

https://github.com/user-attachments/assets/e5abbc0b-b007-4c90-bef6-1efbba4a2cba

**F3** (female) — [ko_F3.wav](https://github.com/typomonster/mlx-audio/raw/main/docs/supertonic/raw/ko_F3.wav)

https://github.com/user-attachments/assets/237f2daa-3452-4a68-8987-10cc183bea97

> The bare `github.com/user-attachments/assets/…` URLs above auto-embed as GitHub media players. Those URLs come from dragging the audio-only MP4s (in `docs/supertonic/`) into the GitHub web UI — there's no CLI/API to mint them, and auto-embed only works when the URL stands on its own line (not inside a table cell).

## Short benchmark samples

Used for the post-warmup performance measurement in the top-level README.

**`"Hello world."`** — voice M1, en — [WAV](https://github.com/typomonster/mlx-audio/raw/main/docs/supertonic/raw/short_en_M1.wav)

https://github.com/user-attachments/assets/9244766c-deb9-4fb6-bda9-5d534ec15223

**`"오늘 아침 공원을 산책했어요."`** — voice F1, ko — [WAV](https://github.com/typomonster/mlx-audio/raw/main/docs/supertonic/raw/short_ko_F1.wav)

https://github.com/user-attachments/assets/0934f7ac-f151-4082-90c2-98246e259f7b

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
