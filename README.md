# PC Agent (Linux, simple MVP)

Локальный агент на Python, который:

1) **слушает wake-word** (твоя `beavis.onnx` через `openwakeword`)  
2) после срабатывания включает **VAD → ASR** (Vosk или Whisper)  
3) превращает речь в текстовую команду  
4) пытается выполнить её **по готовой библиотеке команд** (`core/command_library_linux.yaml`)  
5) если готовой команды нет — делает **LLM fallback** (генерит короткий скрипт и выполняет)  
6) если действие успешно — сохраняет в память “текст → действие”, чтобы в следующий раз выполнять мгновенно без LLM.

Проект **очищен от Windows-специфики**: PowerShell/WinAPI и прочее убрано или заменено Linux-аналогами.

---

## Быстрый старт

### 1) Создать venv и активировать (Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

> Если ты писал `source .venv/scripts/activate` — это путь **для Windows**.  
> В Linux путь **`.venv/bin/activate`**.

### 2) Установить зависимости

```bash
pip install -r requirements.txt
```

Опционально (если хочешь лучший VAD и Whisper):
```bash
pip install torch    # улучшает VAD (Silero)
pip install faster-whisper
```

### 3) Запуск

CLI агент:
```bash
python3 app.py
```

GUI-настройка голоса (wake/VAD/ASR):
```bash
python3 ui.py
```

---

## Предопределённые команды (примеры)

Все шаблоны лежат в `core/command_library_linux.yaml` — можно добавлять свои.

- `открой сайт https://youtube.com`
- `открой папку ~/Downloads`
- `найди файл report.pdf`
- `найди текст "TODO" в ~/projects`
- `запусти steam` / `запусти discord`
- `громче` / `тише` (или `громче 10`)
- `пауза` (playerctl)
- `вайфай статус` / `включи вайфай` / `выключи вайфай`
- `сделай скриншот` (области)


---

## Голосовой пайплайн

### Wake-word

Модель лежит тут:
- `voice_agent/models/beavis.onnx`

Настройки:
- `voice_agent/config.yaml` → `wake_word.*`

Проверка модели отдельно:
```bash
python3 scripts/test_beavis.py
```

### ASR (распознавание речи)

- **Vosk** — быстрый и лёгкий (по умолчанию)
- **Whisper (faster-whisper)** — точнее, но тяжелее

Выбор в `voice_agent/config.yaml`:
```yaml
asr:
  backend: vosk   # или whisper
```

Для Vosk нужно положить модель в `./models/...` или указать путь:
```yaml
asr:
  backend: vosk
  vosk_model_path: /path/to/vosk-model-small-ru-0.22
```

---

## LLM без LM Studio

Есть 2 простых варианта.

### Вариант A: любой OpenAI-compatible сервер (HTTP)

Подойдут: **llama.cpp server**, **vLLM**, **text-generation-webui** (если поднял OpenAI API).

Пример:
```bash
export LLM_BACKEND=openai_compatible
export OPENAI_BASE_URL=http://localhost:8080/v1
export OPENAI_API_KEY=not-needed
export FAST_MODEL=local-model
```

### Вариант B: in-process GGUF через llama-cpp-python (без сервера)

```bash
pip install llama-cpp-python

export LLM_BACKEND=llama_cpp
export LLM_MODEL_PATH=/path/to/model.gguf
export LLM_CTX=4096
export LLM_GPU_LAYERS=0   # если есть сборка с CUDA — можно поставить >0
```

---

## Библиотека быстрых команд

Файл: `core/command_library_linux.yaml`

Туда уже добавлены команды типа:
- открыть URL / поиск в интернете
- открыть папку/файл
- найти файл / найти текст в папке
- процессы / kill
- диск (df/lsblk), сеть (ip)
- VS Code открыть путь
- git status/pull в папке

Если хочешь — добавляй свои “скиллы” туда же: это самый быстрый и надёжный путь без LLM.

---



## VAD: залипает “speaking=YES” или фраза не заканчивается

Симптомы:
- ты замолчал, но в `ui.py` всё ещё показывается что ты говоришь
- состояние остаётся `LISTENING`, ASR не отдаёт финальный текст, всё “тупит/лагает”

Что сделано в этом проекте (фикс):
1) **Silero VAD теперь сбрасывает внутреннее состояние** (`model.reset_states()` на `vad.reset()`).
   Это критично: Silero в потоковом режиме **stateful**, и без сброса иногда “залипает”.
2) **Runtime перестал накапливать бесконечную очередь audio.chunk событий**.
   В `voice_agent/main.py` добавлен `bus.drain()` + coalesce: мы обрабатываем *последний* `audio.chunk`
   и последние статус-ивенты, а устаревшие куски аудио **выкидываем**. Это делает пайплайн “реалтайм”
   вместо “через 3 секунды догонит”.
3) Добавлен **hard-timeout в LISTENING**: если VAD всё же не смог завершить фразу —
   runtime принудительно завершит сегмент и перейдёт в `DECODING` (в UI увидишь `vad.forced_end`).
   Таймаут рассчитывается из `asr.max_utterance_s` (+ небольшой запас).

Как тюнить под свой микрофон (в `voice_agent/config.yaml` или через вкладку Voice в UI):
- `vad.threshold` — порог “это речь” (Silero prob). Если слишком низко → ложные “speaking”.
- `vad.end_silence_ms` — сколько тишины нужно, чтобы закрыть фразу. Типично 400–900 мс.
- `vad.min_speech_ms` — сколько подряд речи нужно, чтобы стартануть `speech_start`.
- `vad.min_rms` и `vad.noise_ratio` — шумовой гейт. Если у микрофона шумный фон:
  подними `noise_ratio` (например 2.5–3.5) или `min_rms`.

Если всё ещё “залипает”:
- временно отключи Silero (быстро проверить): в `voice_agent/vad.py` поставь `prefer_silero=False`
  (или просто не ставь torch) — будет energy VAD.
- проверь, что `audio.sample_rate=16000` и `audio.chunk_ms` 20–32 (меньше можно, но смысла мало).
- включи `logging.level: debug` и смотри события `vad.score` (prob/rms/gate).


## Если звук не работает (sounddevice/PortAudio)

На Ubuntu иногда нужно поставить PortAudio:
```bash
sudo apt update
sudo apt install -y libportaudio2 portaudio19-dev
```

## DEBUG

```bash
PC_AGENT_DEBUG=1 python3 app.py
```

Логи: `logs/pc_agent.log`



### VAD: prob≈1.0, но речь не стартует (приходится кричать)

Это почти всегда **RMS‑гейт**: у Silero `prob` высокий, но микрофон даёт маленькую амплитуду (`rms < gate`), поэтому старт не происходит.

**Решение (рекомендуется):** доверять Silero и отключить RMS‑гейт для него:

- `voice_agent/config.yaml` → `vad.use_rms_gate_with_silero: false` (по умолчанию)

UI теперь показывает `rms`, `gate`, `thr`, и флаги `gate_on / voice` — по ним легко понять, что именно блокирует старт.

Если у тебя много ложных стартов на музыке/шуме, включай `use_rms_gate_with_silero: true` и подбирай `min_rms / noise_ratio`.
