#!/usr/bin/env python3
"""Wake-word test for beavis.onnx (Linux-friendly).

Requires:
  pip install openwakeword sounddevice numpy

Run:
  python scripts/test_beavis.py

Optional:
  OWW_MODEL=path/to/beavis.onnx
  OWW_THRESHOLD=0.6
"""

import os
import time
from collections import deque
from pathlib import Path
from queue import Queue, Empty

import numpy as np
import sounddevice as sd
from openwakeword.model import Model


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "voice_agent" / "models" / "beavis.onnx"

MODEL_PATH = Path(os.getenv("OWW_MODEL", str(DEFAULT_MODEL))).expanduser()
THRESHOLD = float(os.getenv("OWW_THRESHOLD", "0.6"))
SAMPLE_RATE = 16000
CHUNK_MS = 20
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)

HISTORY_SIZE = 40
timeline = deque(["-"] * HISTORY_SIZE, maxlen=HISTORY_SIZE)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Wake-word model not found: {MODEL_PATH}")

print(f"Model: {MODEL_PATH}")
print(f"Threshold: {THRESHOLD}")
print("Speak wake-word... (Ctrl+C to stop)")

model = Model(wakeword_models=[str(MODEL_PATH)], inference_framework="onnx", vad_threshold=0.0)

q: Queue[np.ndarray] = Queue()


def audio_cb(indata, frames, time_info, status):
    if status:
        # print(status)  # uncomment for debug
        pass
    q.put(indata.copy().reshape(-1).astype(np.int16, copy=False))


try:
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=CHUNK_SAMPLES,
        callback=audio_cb,
    ):
        last_print = 0.0
        while True:
            try:
                chunk = q.get(timeout=0.5)
            except Empty:
                continue

            # Basic noise gate (optional)
            rms = float(np.sqrt(np.mean((chunk.astype(np.float32) / 32768.0) ** 2)))
            if rms < 0.002:
                continue

            scores = model.predict(chunk)
            best_name = max(scores, key=lambda k: float(scores[k]))
            best = float(scores[best_name])

            timeline.append("#" if best >= THRESHOLD else "-")

            now = time.time()
            if now - last_print > 0.2:
                print(f"{best_name}={best:.2f}  {''.join(timeline)}", end="\r", flush=True)
                last_print = now

            if best >= THRESHOLD:
                print()
                print(f"DETECTED: {best_name} score={best:.2f}")
                # cooldown
                time.sleep(1.0)

except KeyboardInterrupt:
    print("\nStopped.")
