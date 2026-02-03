from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import sounddevice as sd

from .bus import Event, EventBus


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int
    channels: int
    chunk_ms: int
    device: int | None = None
    input_dtype: str = "int16"


class AudioCapture:
    """Low-level microphone capture.

    Publishes:
      - audio.chunk: {"data": np.int16[...,1], "ts": float}
      - audio.level: {"rms": float(0..1), "peak": float(0..1), "ts": float}
    """

    def __init__(self, config: AudioConfig, bus: EventBus) -> None:
        self.config = config
        self.bus = bus
        self.logger = logging.getLogger("voice_agent.audio")
        self._stream: sd.InputStream | None = None

        self._stream_sr: int = int(config.sample_rate)  # actual opened stream SR
        self._last_level_emit = 0.0
        self._muted = False
        self._input_dtype = (config.input_dtype or "float32").strip().lower()
        self._last_clip_warn = 0.0

    def _to_int16(self, data: np.ndarray) -> np.ndarray:
        if data.dtype == np.int16:
            return data
        if data.dtype.kind == "f":
            clipped = np.clip(data, -1.0, 1.0)
            return np.round(clipped * 32767.0).astype(np.int16)
        if data.dtype == np.int32:
            return np.clip(data, -32768, 32767).astype(np.int16)
        return data.astype(np.int16, copy=False)

    def set_muted(self, muted: bool) -> None:
        self._muted = muted

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags) -> None:
        if status:
            # non-fatal flags happen; don't crash the stream
            self.logger.warning("Audio stream status: %s", status)

        ts = time.monotonic()

        data = self._to_int16(indata)
        if data.size:
            peak = int(np.max(np.abs(data)))
            if peak >= 32700 and (ts - self._last_clip_warn) > 2.0:
                self.logger.warning("Audio clipping detected (peak=%s). Reduce mic gain or preamp.", peak)
                self._last_clip_warn = ts
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if not self._muted:
            self.bus.publish(Event("audio.chunk", {"data": data.copy(), "ts": ts}))

        # Emit level ~8 Hz
        if ts - self._last_level_emit >= 0.12:
            if data.size:
                x = data.astype(np.float32)
                if x.ndim > 1:
                    x = x[:, 0]
                x /= 32768.0
                rms = float(np.sqrt(np.mean(x * x)))
                peak = float(np.max(np.abs(x)))
            else:
                rms = 0.0
                peak = 0.0
            self.bus.publish(Event("audio.level", {"rms": rms, "peak": peak, "ts": ts}))
            self._last_level_emit = ts

    def start(self) -> None:
        if self._stream:
            return

        target_sr = int(self.config.sample_rate)
        blocksize = int(target_sr * (self.config.chunk_ms / 1000.0))

        try:
            self._stream_sr = target_sr
            self._stream = sd.InputStream(
                samplerate=target_sr,
                channels=self.config.channels,
                blocksize=blocksize,
                dtype=self._input_dtype,
                device=self.config.device,
                callback=self._callback,
            )
            self._stream.start()

        except sd.PortAudioError as e:
            msg = str(e)
            if ("Invalid sample rate" in msg) or ("paInvalidSampleRate" in msg) or ("-9997" in msg):
                # Most common on ALSA hw:* devices: open at default SR and resample to target
                try:
                    info = sd.query_devices(self.config.device, "input")
                    fallback_sr = int(info.get("default_samplerate") or 48000)
                except Exception:
                    fallback_sr = 48000

                self.logger.warning(
                    "Requested sample_rate=%s not supported by device=%s. Falling back to %s Hz without resampling. "
                    "Set audio.sample_rate=%s in config to avoid mismatched ASR/VAD sample rates.",
                    target_sr, self.config.device, fallback_sr, fallback_sr
                )

                self._stream_sr = fallback_sr
                blocksize_fb = int(fallback_sr * (self.config.chunk_ms / 1000.0))
                self._stream = sd.InputStream(
                    samplerate=fallback_sr,
                    channels=self.config.channels,
                    blocksize=blocksize_fb,
                    dtype=self._input_dtype,
                    device=self.config.device,
                    callback=self._callback,
                )
                self._stream.start()
            else:
                raise

        self.logger.info(
            "Audio started target_sr=%s stream_sr=%s chunk_ms=%s device=%s blocksize=%s",
            target_sr, self._stream_sr, self.config.chunk_ms, self.config.device,
            int(self._stream_sr * (self.config.chunk_ms / 1000.0))
        )

    def stop(self) -> None:
        if self._stream:
            try:
                self._stream.stop()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
            self.logger.info("Audio stopped.")
