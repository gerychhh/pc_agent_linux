from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .bus import Event, EventBus

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


# main.py expects:
#   from .vad import SileroVAD, VadConfig
# and will build VadConfig(...) from UI/config with keys like:
#   device, threshold, end_silence_ms, min_speech_ms, sample_rate, min_rms,
#   noise_floor_alpha, noise_ratio, score_emit_ms
@dataclass(frozen=True)
class VadConfig:
    sample_rate: int = 16000

    # UI / config keys
    device: str = "auto"              # "auto" | "cpu" | "cuda"
    threshold: float = 0.50           # start threshold (silero speech prob)
    end_silence_ms: int = 700         # end-of-utterance silence required to stop
    min_speech_ms: int = 150          # required consecutive speech duration to trigger speech_start

    # Noise gate (RMS-based) – prevents false speech on silence/noise
    min_rms: float = 0.0025
    noise_floor_alpha: float = 0.04   # EMA update speed for noise floor
    noise_ratio: float = 3.0          # gate = max(min_rms, noise_floor * noise_ratio)

    # Debug emit cadence
    score_emit_ms: int = 120

    # Optional tuning (safe defaults)
    prefer_silero: bool = True

    # When using Silero probabilities, RMS gate can be too strict on quiet mics.
    # Default: OFF (trust Silero). Enable only if you get false triggers on noise/music.
    use_rms_gate_with_silero: bool = False

    # Hysteresis: end threshold is usually lower than start threshold.
    # Helps avoid "choppy" toggling while also preventing stuck states when model state lingers.
    end_threshold: float | None = None
    hysteresis: float = 0.15  # end = max(0, threshold - hysteresis) if end_threshold is None

    # Used for frame sizing fallback
    chunk_ms: int = 20


class SileroVAD:
    """Streaming VAD with Silero (if available) + lightweight energy gate.

    Goals for this project:
      1) Be real-time (no long backlog / no "lags forever")
      2) Never get stuck in speaking=YES after speech ends
      3) Never crash the pipeline (VAD is allowed to fail-open to energy mode)

    Publishes:
      - vad.speech_start {"ts": float}
      - vad.speech_end   {"ts": float}
      - vad.score        {"ts": float, "prob": float, "rms": float, "noise_gate": float, "speaking": bool}
    """

    def __init__(self, config: VadConfig, bus: EventBus) -> None:
        self.config = config
        self.bus = bus
        self.logger = logging.getLogger("voice_agent.vad")

        # device selection
        self._device = "cpu"
        if torch is not None:
            d = (config.device or "auto").lower()
            if d == "cuda":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            elif d == "cpu":
                self._device = "cpu"
            else:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model: Optional[object] = None
        self._use_silero = False

        # buffer of int16 mono
        self._buf = np.zeros((0,), dtype=np.int16)

        # state
        self._speaking = False
        self._speech_ms = 0
        self._silence_ms = 0

        self._noise_floor = float(config.min_rms)
        self._last_emit_ts = 0.0

        # thresholds
        self._start_thr = float(config.threshold)
        self._end_thr = float(config.end_threshold) if config.end_threshold is not None else max(0.0, self._start_thr - float(config.hysteresis))

        if torch is not None and config.prefer_silero:
            try:
                model, _ = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=False,
                    onnx=False,
                )
                model.eval()
                try:
                    model.to(self._device)  # type: ignore[attr-defined]
                except Exception:
                    self._device = "cpu"

                # IMPORTANT: silero-vad is stateful in streaming mode.
                # Always reset states at init, and on every VAD.reset().
                try:
                    if hasattr(model, "reset_states"):
                        model.reset_states()  # type: ignore[attr-defined]
                except Exception:
                    pass

                self.model = model
                self._use_silero = True
                self.logger.info("VAD backend=silero sr=%s start_thr=%.2f end_thr=%.2f device=%s",
                                 config.sample_rate, self._start_thr, self._end_thr, self._device)
            except Exception as e:
                self.logger.warning("Silero VAD load failed; fallback to energy VAD: %s", e)
                self._use_silero = False
        else:
            self.logger.info("VAD backend=energy sr=%s start_thr=%.2f end_thr=%.2f", config.sample_rate, self._start_thr, self._end_thr)

    def reset(self) -> None:
        self._buf = np.zeros((0,), dtype=np.int16)
        self._speaking = False
        self._speech_ms = 0
        self._silence_ms = 0
        self._noise_floor = float(self.config.min_rms)
        self._last_emit_ts = 0.0

        # reset silero internal state to avoid "stuck speaking"
        if self._use_silero and self.model is not None:
            try:
                if hasattr(self.model, "reset_states"):
                    self.model.reset_states()  # type: ignore[attr-defined]
            except Exception:
                pass

    @property
    def speaking(self) -> bool:
        return self._speaking

    def process_chunk(self, pcm_i16: np.ndarray, ts: float) -> None:
        if pcm_i16 is None:
            return

        x = np.asarray(pcm_i16)
        if x.size == 0:
            return
        if x.ndim == 2:
            x = x[:, 0]
        if x.dtype != np.int16:
            x = x.astype(np.int16, copy=False)

        # Append into buffer
        self._buf = np.concatenate([self._buf, x])

        sr = int(self.config.sample_rate)

        # Silero streaming works best with fixed 512 samples @16k (~32ms).
        # For other sample rates, use ~32ms frames.
        frame_len = 512 if sr == 16000 else max(160, int(round(sr * 0.032)))
        if frame_len <= 0:
            return

        # Consume buffered frames
        while self._buf.size >= frame_len:
            frame = self._buf[:frame_len]
            self._buf = self._buf[frame_len:]

            prob, rms, gate = self._score_frame(frame, sr)

            frame_ms = max(1, int(round((len(frame) / float(sr)) * 1000.0)))

            # IMPORTANT: use hysteresis.
            # - when not speaking: need stronger evidence to start
            # - when already speaking: allow lower prob to continue (prevents choppiness)
            thr = self._start_thr if not self._speaking else self._end_thr

            rms_gate_enabled = (not self._use_silero) or bool(self.config.use_rms_gate_with_silero)
            is_voice = (prob >= thr) and (rms >= gate if rms_gate_enabled else True)

            if not self._speaking:
                # require consecutive speech frames to start
                if is_voice:
                    self._speech_ms += frame_ms
                    self._silence_ms = 0
                else:
                    self._speech_ms = 0
                    self._silence_ms += frame_ms

                if self._speech_ms >= int(self.config.min_speech_ms):
                    self._speaking = True
                    self._speech_ms = 0
                    self._silence_ms = 0
                    self.bus.publish(Event("vad.speech_start", {"ts": ts}))
            else:
                # while speaking: count silence even if occasional false positives
                if is_voice:
                    self._silence_ms = 0
                else:
                    self._silence_ms += frame_ms

                if self._silence_ms >= int(self.config.end_silence_ms):
                    self._speaking = False
                    self._speech_ms = 0
                    self._silence_ms = 0
                    self.bus.publish(Event("vad.speech_end", {"ts": ts}))

            # Emit score occasionally for UI/debug
            if self._last_emit_ts == 0.0 or (ts - self._last_emit_ts) * 1000.0 >= int(self.config.score_emit_ms):
                self._last_emit_ts = ts
                self.bus.publish(Event("vad.score", {
                    "ts": ts,
                    "prob": float(prob),
                    "rms": float(rms),
                    "noise_gate": float(gate),
                    "thr": float(thr),
                    "is_voice": bool(is_voice),
                    "rms_gate_enabled": bool(rms_gate_enabled),
                    "speaking": bool(self._speaking),
                }))

    def _score_frame(self, frame_i16: np.ndarray, sr: int) -> tuple[float, float, float]:
        # float audio
        audio = (frame_i16.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
        rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0

        # noise floor update:
        # - always track silence/noise slowly
        # - even during speaking, allow floor to decrease a bit on low-energy frames
        a = float(self.config.noise_floor_alpha)
        if (not self._speaking) or (rms < self._noise_floor * 1.25):
            self._noise_floor = (1.0 - a) * self._noise_floor + a * rms

        gate = max(float(self.config.min_rms), self._noise_floor * float(self.config.noise_ratio))

        # Silero prob
        if self._use_silero and torch is not None and self.model is not None:
            try:
                t = torch.from_numpy(audio).unsqueeze(0)  # (1, n)
                if t.dtype != torch.float32:
                    t = t.float()
                if self._device != "cpu":
                    t = t.to(self._device)
                with torch.no_grad():
                    prob = float(self.model(t, sr).item())  # type: ignore[operator]
                # clamp
                if prob != prob:  # NaN guard
                    prob = 0.0
                prob = float(max(0.0, min(1.0, prob)))
                return prob, rms, gate
            except Exception as e:
                # never kill pipeline; fallback to energy
                self.logger.debug("Silero VAD frame failed, fallback to energy: %s", e)

        # Energy fallback: map rms to pseudo-probability
        prob = float(min(1.0, max(0.0, (rms - 0.006) / 0.03)))
        return prob, rms, gate
