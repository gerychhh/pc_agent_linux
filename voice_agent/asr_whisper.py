from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from .bus import Event, EventBus


@dataclass(frozen=True)
class AsrConfig:
    model: str = "small"
    device: str = "auto"          # auto|cpu|cuda
    compute_type: str = "int8"    # float16|int8|int8_float16|...
    beam_size: int = 1
    language: str = "ru"
    max_utterance_s: int = 10

    # partials (optional, can be expensive)
    partial_interval_ms: int = 250
    partial_min_delta: int = 3
    min_partial_s: float = 0.6
    min_buffer_s: float = 0.6

    sample_rate: int = 16000
    no_speech_threshold: float = 0.8
    log_prob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4


class FasterWhisperASR:
    """Streaming ASR backend using faster-whisper.

    Contract (used by VoiceAgentRuntime):
      - reset()
      - speech_start()
      - speech_end(ts=None)
      - accept_audio(pcm_int16, ts)
    Emits events:
      - asr.partial {"ts": float, "text": str}
      - asr.final   {"ts": float, "text": str}
    """

    def __init__(self, config: AsrConfig, bus: EventBus) -> None:
        self.config = config
        self.bus = bus
        self.logger = logging.getLogger("voice_agent.asr_whisper")

        self._model = None
        self._speaking = False
        self._buf: list[np.ndarray] = []
        self._last_partial_at = 0.0
        self._last_partial_text = ""

        self._load_model()

    def _load_model(self) -> None:
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception as exc:
            raise RuntimeError("faster-whisper is not installed. Install: pip install faster-whisper") from exc

        device = (self.config.device or "auto").strip().lower()
        if device == "auto":
            try:
                import torch  # type: ignore
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        self._model = WhisperModel(self.config.model, device=device, compute_type=self.config.compute_type)
        self.logger.info("ASR whisper model loaded: model=%s device=%s compute=%s", self.config.model, device, self.config.compute_type)

    def reset(self) -> None:
        self._speaking = False
        self._buf.clear()
        self._last_partial_at = 0.0
        self._last_partial_text = ""

    def speech_start(self) -> None:
        self.reset()
        self._speaking = True
        self.bus.publish(Event("asr.speech_start", {"ts": time.time()}))

    def speech_end(self, ts: float | None = None) -> None:
        if not self._speaking:
            return
        self._speaking = False
        self._finalize()

    def accept_audio(self, pcm_int16: np.ndarray, ts: float) -> None:
        if not self._speaking:
            return
        if pcm_int16 is None or pcm_int16.size == 0:
            return

        # Keep only max_utterance_s seconds
        max_samples = int(self.config.max_utterance_s * self.config.sample_rate)
        self._buf.append(pcm_int16.astype(np.int16, copy=False))
        total = sum(b.size for b in self._buf)
        if total > max_samples:
            # drop from front
            drop = total - max_samples
            while self._buf and drop > 0:
                if self._buf[0].size <= drop:
                    drop -= self._buf[0].size
                    self._buf.pop(0)
                else:
                    self._buf[0] = self._buf[0][drop:]
                    drop = 0

        if self._should_emit_partial(ts):
            text = self._transcribe(joined_only=True)
            if self._is_significant_partial(text):
                self._last_partial_text = text
                self._last_partial_at = ts
                self.bus.publish(Event("asr.partial", {"ts": ts, "text": text}))

    def _buffer_duration_s(self) -> float:
        total = sum(b.size for b in self._buf)
        return total / float(self.config.sample_rate)

    def _should_emit_partial(self, ts: float) -> bool:
        if self.config.partial_interval_ms <= 0:
            return False
        if self._buffer_duration_s() < float(self.config.min_partial_s):
            return False
        interval = self.config.partial_interval_ms / 1000.0
        return (ts - self._last_partial_at) >= interval

    def _is_significant_partial(self, text: str) -> bool:
        t = (text or "").strip()
        if len(t) < self.config.partial_min_delta:
            return False
        prev = (self._last_partial_text or "").strip()
        if not prev:
            return True
        # simple delta heuristic
        return abs(len(t) - len(prev)) >= self.config.partial_min_delta

    def _finalize(self) -> None:
        if self._buffer_duration_s() < float(self.config.min_buffer_s):
            # too short
            self.bus.publish(Event("asr.final", {"ts": time.time(), "text": ""}))
            self.reset()
            return

        text = self._transcribe(joined_only=False)
        self.bus.publish(Event("asr.final", {"ts": time.time(), "text": (text or "").strip()}))
        self.reset()

    def _transcribe(self, *, joined_only: bool) -> str:
        if not self._model:
            return ""
        if not self._buf:
            return ""
        audio_i16 = np.concatenate(self._buf, axis=0)
        # normalize to float32 [-1, 1]
        audio = (audio_i16.astype(np.float32) / 32768.0).clip(-1.0, 1.0)

        try:
            segments, info = self._model.transcribe(
                audio,
                language=(self.config.language or None),
                beam_size=int(self.config.beam_size or 1),
                vad_filter=True,
                no_speech_threshold=float(self.config.no_speech_threshold),
                log_prob_threshold=float(self.config.log_prob_threshold),
                compression_ratio_threshold=float(self.config.compression_ratio_threshold),
            )
            text = "".join(seg.text for seg in segments).strip()
            return text
        except Exception as exc:
            self.logger.error("ASR transcribe error: %s", exc)
            return ""
