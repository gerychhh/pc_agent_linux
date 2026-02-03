from __future__ import annotations

import argparse
import os
import time
from collections import deque
from pathlib import Path

import numpy as np

try:
    import pyaudio
except Exception as e:
    pyaudio = None
    _pyaudio_err = e

import openwakeword
from openwakeword.model import Model


def main() -> int:
    parser = argparse.ArgumentParser(description="Live microphone test for an openWakeWord ONNX model.")
    parser.add_argument("--model", type=str, default=str(Path(__file__).resolve().parent / "models" / "beavis.onnx"),
                        help="Path to .onnx wakeword model (default: ./training_results/beavis.onnx)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Trigger threshold (0..1)")
    parser.add_argument("--chunk", type=int, default=1280, help="Audio chunk size at 16kHz")
    args = parser.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}")
        return 2

    if pyaudio is None:
        print("[ERROR] pyaudio is not installed, live test cannot start.")
        print("Install on Linux: sudo apt-get install -y portaudio19-dev && pip install pyaudio")
        print("Install on Windows: pip install pyaudio")
        print("Underlying error:", _pyaudio_err)
        return 3

    HISTORY_SIZE = 40
    timeline = deque(['-'] * HISTORY_SIZE, maxlen=HISTORY_SIZE)

    oww = Model(wakeword_models=[str(model_path)], inference_framework="onnx")

    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=args.chunk)

    print("\n" + "=" * 60)
    print("  LIVE WAKEWORD TEST")
    print(f"  model: {model_path}")
    print("  Press Ctrl+C to exit")
    print("=" * 60 + "\n")

    try:
        while True:
            data = stream.read(args.chunk, exception_on_overflow=False)
            audio_frame = np.frombuffer(data, dtype=np.int16)

            prediction = oww.predict(audio_frame)
            prob = float(list(prediction.values())[0])

            if prob > 0.99:
                char = '█'
            elif prob > 0.8:
                char = '▓'
            elif prob > 0.7:
                char = '▒'
            else:
                char = '░'

            timeline.append(char)
            timeline_str = "".join(timeline)
            bar = "█" * int(prob * 20)

            print(f"Audio: [{timeline_str}] | score: {prob:.2f} | {bar:<20}", end="\r")

            if prob > args.threshold:
                current_time = time.strftime("%H:%M:%S")
                print(f"\n[{current_time}] >> DETECTED! score={prob:.2f} {'!'*5}")
                time.sleep(0.3)

    except KeyboardInterrupt:
        print("\n\n[OK] Stopped.")
        return 0
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
