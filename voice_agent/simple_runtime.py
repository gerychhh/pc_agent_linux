from __future__ import annotations

"""
voice_agent.simple_runtime

A "simple but rock-solid" voice pipeline:

    wake-word -> record with lightweight VAD -> ASR (whisper/vosk) -> callback(text)

Design goals:
- One thread, one sounddevice stream, minimal shared state.
- No event bus, no complex state machine.
- Defensive fallbacks:
    * wake backend: openwakeword -> (fallback) vosk keyword spotting
    * asr backend: faster-whisper -> openai-whisper -> (fallback) vosk
- Very explicit status callbacks for UI/debugging.

Audio format:
- Capture int16 mono from sounddevice.
- Internally we keep 16kHz mono int16 for wake/VAD/ASR.
"""

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Optional

import numpy as np
import sounddevice as sd
import yaml


def _rms_i16(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    xf = x.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(xf * xf)))


def _resample_int16_mono(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Same idea as audio_capture._resample_int16_mono but duplicated to keep this module standalone."""
    if src_sr == dst_sr:
        y = x
    else:
        x = np.asarray(x)
        if x.ndim == 2:
            x = x[:, 0]
        if x.dtype != np.int16:
            x = x.astype(np.int16, copy=False)

        if src_sr % dst_sr == 0:
            factor = src_sr // dst_sr
            n = (len(x) // factor) * factor
            if n <= 0:
                return np.zeros((0,), dtype=np.int16)
            y = x[:n].reshape(-1, factor).astype(np.float32).mean(axis=1)
        else:
            n_dst = int(round(len(x) * (dst_sr / float(src_sr))))
            if n_dst <= 0:
                return np.zeros((0,), dtype=np.int16)
            xp = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
            fp = x.astype(np.float32)
            xnew = np.linspace(0.0, 1.0, num=n_dst, endpoint=False)
            y = np.interp(xnew, xp, fp)
    return np.clip(np.round(y), -32768, 32767).astype(np.int16)


@dataclass(frozen=True)
class SimpleConfig:
    # audio
    device: int | None = None            # None = default device
    device_sr: int = 48000               # open stream at this SR if possible; we resample to 16k
    target_sr: int = 16000
    chunk_ms: int = 20                   # 20ms chunks are stable for VAD
    channels: int = 1

    # wake
    wake_enabled: bool = True
    wake_backend: str = "openwakeword"   # openwakeword | vosk
    wake_model_paths: tuple[str, ...] = ()
    wake_threshold: float = 0.60
    wake_patience_frames: int = 2
    wake_cooldown_ms: int = 1200
    wake_min_rms: float = 0.003
    wake_keyword: str = "бивис"
    wake_keyword_aliases: tuple[str, ...] = ()

    # record/VAD (energy-based)
    preroll_ms: int = 450
    max_utterance_s: int = 8
    vad_start_rms: float = 0.006         # speech starts when rms >= this
    vad_end_rms: float = 0.004           # speech ends when rms < this for end_silence_ms
    min_speech_ms: int = 180
    end_silence_ms: int = 700

    # ASR
    asr_backend: str = "whisper"         # whisper | vosk
    language: str = "ru"
    whisper_model: str = "small"         # faster-whisper/openai-whisper names
    whisper_device: str = "auto"         # auto|cpu|cuda
    whisper_compute_type: str = "int8"   # faster-whisper only
    vosk_model_path: str | None = None

    # misc
    log_level: str = "INFO"


def load_simple_config(config_path: Path) -> SimpleConfig:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    audio = data.get("audio", {}) or {}
    wake = data.get("wake_word", {}) or {}
    vad = data.get("vad", {}) or {}
    asr = data.get("asr", {}) or {}
    logging_cfg = data.get("logging", {}) or {}

    base = config_path.parent

    def _as_int_or_none(v) -> int | None:
        """Parse config value to int or None.

        Accepts:
          - int
          - numeric strings like "0", "2"
          - None / "" -> None
        """
        if v is None:
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            # allow 0.0-like values if someone saved JSON
            if v.is_integer():
                return int(v)
            return None
        s = str(v).strip()
        if not s:
            return None
        # IMPORTANT: str(None) == "None"; treat it as unset.
        if s.lower() == "none":
            return None
        # allow "-1" too (some setups use -1 for default)
        if s.lstrip("-").isdigit():
            try:
                return int(s)
            except Exception:
                return None
        return None

    def _as_int(v, default: int) -> int:
        """Parse to int, fallback to default on None/invalid."""
        parsed = _as_int_or_none(v)
        return default if parsed is None else int(parsed)

    def _as_float(v, default: float) -> float:
        """Parse to float, fallback to default on None/invalid."""
        if v is None:
            return float(default)
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip()
        if not s or s.lower() == "none":
            return float(default)
        try:
            return float(s)
        except Exception:
            return float(default)

    def _p(p: str | None) -> str | None:
        if not p:
            return None
        pp = Path(p)
        if not pp.is_absolute():
            pp = (base / pp).resolve()
        return str(pp)

    return SimpleConfig(
        device=_as_int_or_none(audio.get("device")),
        device_sr=_as_int(audio.get("device_sr", audio.get("sample_rate", 48000)), 48000),
        target_sr=_as_int(audio.get("sample_rate", 16000), 16000),
        chunk_ms=_as_int(audio.get("chunk_ms", 20), 20),
        channels=_as_int(audio.get("channels", 1), 1),

        wake_enabled=bool(wake.get("enabled", True)),
        wake_backend=str(wake.get("backend", "openwakeword")).strip().lower(),
        wake_model_paths=tuple(wake.get("model_paths", []) or []),
        wake_threshold=_as_float(wake.get("threshold", 0.6), 0.6),
        wake_patience_frames=max(1, _as_int(wake.get("patience_frames", 2), 2)),
        wake_cooldown_ms=_as_int(wake.get("cooldown_ms", 1200), 1200),
        wake_min_rms=_as_float(wake.get("min_rms", 0.003), 0.003),
        wake_keyword=str(wake.get("keyword", wake.get("agent_name", "бивис"))).strip().lower(),
        wake_keyword_aliases=tuple(wake.get("keyword_aliases", []) or []),

        preroll_ms=_as_int(wake.get("preroll_ms", 450), 450),
        max_utterance_s=_as_int(asr.get("max_utterance_s", 8), 8),
        vad_start_rms=_as_float(vad.get("start_rms", vad.get("min_rms", 0.006)), 0.006),
        vad_end_rms=_as_float(vad.get("end_rms", 0.004), 0.004),
        min_speech_ms=_as_int(vad.get("min_speech_ms", 180), 180),
        end_silence_ms=_as_int(vad.get("end_silence_ms", 700), 700),

        asr_backend=str(asr.get("backend", "whisper")).strip().lower(),
        language=str(asr.get("language", "ru") or "ru"),
        whisper_model=str(asr.get("model", "small") or "small"),
        whisper_device=str(asr.get("device", "auto") or "auto").strip().lower(),
        whisper_compute_type=str(asr.get("compute_type", "int8") or "int8"),
        vosk_model_path=_p(asr.get("vosk_model_path") or asr.get("model_path")),

        log_level=str(logging_cfg.get("level", "INFO")).upper(),
    )


class _WakeOWW:
    def __init__(self, model_paths: tuple[str, ...], threshold: float, patience: int, sample_rate: int = 16000) -> None:
        self.threshold = float(threshold)
        self.patience = int(patience)
        self.sample_rate = sample_rate

        try:
            from openwakeword.model import Model  # type: ignore
        except Exception as exc:
            raise RuntimeError("openwakeword is not installed") from exc

        paths = [p for p in model_paths if p]
        if not paths:
            raise RuntimeError("No wake_word.model_paths provided for openwakeword backend")
        self.model = Model(wakeword_models=paths, inference_framework="onnx")
        self.names = list(self.model.model_names) if hasattr(self.model, "model_names") else [Path(paths[0]).stem]

        self._buf = np.zeros((0,), dtype=np.int16)
        self._frame = 1280  # 80ms at 16kHz (matches openWakeWord examples)
        self._hit = {n: 0 for n in self.names}
        self._cooldown_until = 0.0

    def process(self, pcm16: np.ndarray, now: float) -> tuple[bool, dict[str, float], str]:
        if now < self._cooldown_until:
            return (False, {}, "")
        if pcm16.size == 0:
            return (False, {}, "")

        self._buf = np.concatenate([self._buf, pcm16])
        fired = False
        scores_last: dict[str, float] = {}
        best_name = ""
        while self._buf.size >= self._frame:
            frame = self._buf[: self._frame]
            self._buf = self._buf[self._frame :]
            pred = self.model.predict(frame)  # dict name->prob
            scores_last = {k: float(v) for k, v in (pred or {}).items()}
            if scores_last:
                best_name = max(scores_last, key=scores_last.get)
                best = scores_last[best_name]
                if best >= self.threshold:
                    self._hit[best_name] = self._hit.get(best_name, 0) + 1
                else:
                    self._hit[best_name] = 0
                if self._hit.get(best_name, 0) >= self.patience:
                    fired = True
                    break
        if fired:
            self._cooldown_until = now + 1.0
            for k in list(self._hit.keys()):
                self._hit[k] = 0
        return (fired, scores_last, best_name)


class _WakeVoskKeyword:
    def __init__(self, model_path: Path, keyword: str, aliases: tuple[str, ...], sample_rate: int = 16000) -> None:
        try:
            from vosk import Model, KaldiRecognizer  # type: ignore
        except Exception as exc:
            raise RuntimeError("vosk is not installed") from exc

        self.model = Model(str(model_path))
        self.rec = KaldiRecognizer(self.model, sample_rate)
        self.keyword = keyword.strip().lower()
        self.aliases = tuple(a.strip().lower() for a in aliases if a.strip())
        self._cooldown_until = 0.0

    def _match(self, text: str) -> bool:
        t = (text or "").strip().lower()
        if not t:
            return False
        tokens = t.split()
        if self.keyword in tokens:
            return True
        for a in self.aliases:
            if a in tokens:
                return True
        return False

    def process(self, pcm16: np.ndarray, now: float) -> tuple[bool, dict[str, float], str]:
        if now < self._cooldown_until:
            return (False, {}, "")
        if pcm16.size == 0:
            return (False, {}, "")
        raw = pcm16.astype(np.int16, copy=False).tobytes()
        fired = False
        txt = ""
        if self.rec.AcceptWaveform(raw):
            res = json.loads(self.rec.Result() or "{}")
            txt = (res.get("text") or "").strip()
            fired = self._match(txt)
        else:
            part = json.loads(self.rec.PartialResult() or "{}")
            txt = (part.get("partial") or "").strip()
            fired = self._match(txt)
        if fired:
            self._cooldown_until = now + 1.0
        return (fired, {}, self.keyword)


class _ASR:
    def __init__(self, cfg: SimpleConfig, base_path: Path) -> None:
        self.cfg = cfg
        self.base_path = base_path
        self.backend = cfg.asr_backend

        self._vosk = None
        self._fw = None
        self._ow = None

        # Prepare Vosk model if needed as fallback too
        self._vosk_model_path = None
        if cfg.vosk_model_path:
            self._vosk_model_path = Path(cfg.vosk_model_path)
        else:
            candidates = [
                base_path.parent / "models" / "vosk-model-small-ru-0.22",
                base_path.parent / "models" / "vosk-model-ru-0.22",
                base_path / "models" / "vosk-model-small-ru-0.22",
                base_path / "models" / "vosk-model-ru-0.22",
            ]
            self._vosk_model_path = next((p for p in candidates if p.exists()), None)

        if self.backend == "vosk":
            self._init_vosk()
        else:
            # whisper: try faster-whisper then openai-whisper
            if not self._init_faster_whisper():
                self._init_openai_whisper()

    def _init_vosk(self) -> bool:
        try:
            from vosk import Model, KaldiRecognizer  # type: ignore
        except Exception:
            return False
        if not self._vosk_model_path or not self._vosk_model_path.exists():
            return False
        self._vosk = (Model(str(self._vosk_model_path)), KaldiRecognizer)
        return True

    def _init_faster_whisper(self) -> bool:
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception:
            return False
        device = self.cfg.whisper_device
        if device == "auto":
            # faster-whisper supports "cuda" when available, else cpu
            device = "cuda"
        try:
            self._fw = WhisperModel(self.cfg.whisper_model, device=device, compute_type=self.cfg.whisper_compute_type)
            return True
        except Exception:
            self._fw = None
            return False

    def _init_openai_whisper(self) -> bool:
        try:
            import whisper  # type: ignore
        except Exception:
            return False
        dev = self.cfg.whisper_device
        if dev == "auto":
            dev = "cpu"
            try:
                import torch  # type: ignore
                if torch.cuda.is_available():
                    dev = "cuda"
            except Exception:
                pass
        try:
            self._ow = whisper.load_model(self.cfg.whisper_model, device=dev)
            return True
        except Exception:
            self._ow = None
            return False

    def transcribe(self, pcm16_16k: np.ndarray) -> str:
        audio = pcm16_16k.astype(np.float32) / 32768.0
        if audio.size == 0:
            return ""
        # whisper
        if self._fw is not None:
            try:
                segs, info = self._fw.transcribe(audio, language=self.cfg.language, beam_size=1, vad_filter=False)
                text = " ".join((s.text or "").strip() for s in segs).strip()
                return text
            except Exception:
                # fall back
                pass
        if self._ow is not None:
            try:
                use_fp16 = False
                try:
                    import torch  # type: ignore
                    use_fp16 = torch.cuda.is_available()
                except Exception:
                    pass
                res = self._ow.transcribe(audio, language=self.cfg.language, fp16=use_fp16)
                return (res.get("text") or "").strip()
            except Exception:
                pass
        # vosk fallback
        if self._vosk is None:
            self._init_vosk()
        if self._vosk is not None:
            model, KaldiRecognizer = self._vosk
            rec = KaldiRecognizer(model, 16000)
            rec.AcceptWaveform(pcm16_16k.tobytes())
            out = json.loads(rec.FinalResult() or "{}")
            return (out.get("text") or "").strip()
        return ""


class SimpleVoiceRuntime:
    def __init__(
        self,
        config_path: Path,
        *,
        on_final: Callable[[str], None] | None = None,
        on_partial: Callable[[str], None] | None = None,
        on_audio_level: Callable[[float], None] | None = None,
        on_status: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.cfg = load_simple_config(self.config_path)

        logging.basicConfig(
            level=getattr(logging, self.cfg.log_level, logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
            force=True,
        )
        self.logger = logging.getLogger("voice_agent.simple")

        self._on_final = on_final
        self._on_partial = on_partial
        self._on_audio_level = on_audio_level
        self._on_status = on_status

        self._running = threading.Event()
        self._thread: threading.Thread | None = None

        self._wake = None
        self._asr = _ASR(self.cfg, self.config_path.parent)

    # -------------- public --------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()

    # -------------- internals --------------
    def _emit(self, kind: str, payload: dict[str, Any]) -> None:
        if self._on_status:
            try:
                self._on_status(kind, payload)
            except Exception:
                pass

    def _set_state(self, name: str) -> None:
        self._emit("state", {"to": name, "ts": time.monotonic()})

    def _init_wake(self) -> None:
        if not self.cfg.wake_enabled:
            self._wake = None
            return
        backend = self.cfg.wake_backend
        base = self.config_path.parent

        if backend == "openwakeword":
            try:
                paths = []
                for p in self.cfg.wake_model_paths:
                    if not p:
                        continue
                    pp = Path(p)
                    if not pp.is_absolute():
                        pp = (base / pp).resolve()
                    paths.append(str(pp))
                self._wake = _WakeOWW(tuple(paths), self.cfg.wake_threshold, self.cfg.wake_patience_frames, 16000)
                self.logger.info("Wake backend: openwakeword (%s)", ", ".join(self._wake.names))
                return
            except Exception as exc:
                self.logger.warning("openwakeword init failed: %s. Falling back to Vosk keyword spotting.", exc)

        # fallback: Vosk keyword spotting
        try:
            # find model
            mp = None
            candidates = [
                base.parent / "models" / "vosk-model-small-ru-0.22",
                base.parent / "models" / "vosk-model-ru-0.22",
                base / "models" / "vosk-model-small-ru-0.22",
                base / "models" / "vosk-model-ru-0.22",
            ]
            mp = next((p for p in candidates if p.exists()), None)
            if mp is None:
                raise FileNotFoundError("vosk model not found (needed for wake fallback)")
            self._wake = _WakeVoskKeyword(mp, self.cfg.wake_keyword, self.cfg.wake_keyword_aliases, 16000)
            self.logger.info("Wake backend: vosk keyword='%s'", self.cfg.wake_keyword)
        except Exception as exc:
            self._wake = None
            self.logger.error("Wake disabled (no backend): %s", exc)

    def _run(self) -> None:
        self._init_wake()

        chunk = int(self.cfg.device_sr * (self.cfg.chunk_ms / 1000.0))
        if chunk <= 0:
            chunk = 960
        preroll_n = int(self.cfg.target_sr * (self.cfg.preroll_ms / 1000.0))
        preroll: Deque[np.ndarray] = deque(maxlen=max(1, int(preroll_n // (self.cfg.target_sr * self.cfg.chunk_ms / 1000.0)) + 2))

        self._set_state("IDLE")
        mode = "WAIT_WAKE" if self.cfg.wake_enabled and self._wake else "LISTEN"
        last_trigger = 0.0

        # recorder state
        recording: list[np.ndarray] = []
        speech_ms = 0
        silence_ms = 0
        started = False

        def _reset_recording() -> None:
            nonlocal recording, speech_ms, silence_ms, started
            recording = []
            speech_ms = 0
            silence_ms = 0
            started = False

        _reset_recording()

        try:
            with sd.InputStream(
                samplerate=self.cfg.device_sr,
                device=self.cfg.device,
                channels=self.cfg.channels,
                dtype="int16",
                blocksize=chunk,
            ) as stream:
                self.logger.info("Mic stream: device=%s sr=%s chunk=%s", self.cfg.device, self.cfg.device_sr, chunk)
                while self._running.is_set():
                    data, overflowed = stream.read(chunk)
                    if overflowed:
                        self._emit("audio.warn", {"kind": "overflow"})
                    x = np.asarray(data).reshape(-1)
                    x16k = _resample_int16_mono(x, self.cfg.device_sr, self.cfg.target_sr)
                    if x16k.size == 0:
                        continue

                    rms = _rms_i16(x16k)
                    if self._on_audio_level:
                        try:
                            self._on_audio_level(rms)
                        except Exception:
                            pass

                    preroll.append(x16k)

                    now = time.monotonic()

                    if mode == "WAIT_WAKE":
                        # ignore too quiet
                        if rms < self.cfg.wake_min_rms:
                            continue

                        if self._wake:
                            fired, scores, best_name = self._wake.process(x16k, now)
                            if scores:
                                best = float(max(scores.values())) if scores else 0.0
                                self._emit("wake.scores", {"best": best, "best_name": best_name, "scores": scores})
                            if fired and (now - last_trigger) * 1000.0 >= self.cfg.wake_cooldown_ms:
                                last_trigger = now
                                self._emit("wake.detected", {"best_name": best_name, "scores": scores})
                                self._set_state("LISTENING")
                                mode = "LISTEN"
                                _reset_recording()
                                # add preroll immediately
                                for pr in list(preroll):
                                    recording.append(pr)
                                continue
                        continue

                    # LISTEN / RECORD
                    if mode == "LISTEN":
                        recording.append(x16k)
                        if not started:
                            if rms >= self.cfg.vad_start_rms:
                                speech_ms += self.cfg.chunk_ms
                            else:
                                speech_ms = 0
                            if speech_ms >= self.cfg.min_speech_ms:
                                started = True
                                self._emit("vad", {"state": "speech"})
                        else:
                            if rms >= self.cfg.vad_end_rms:
                                silence_ms = 0
                                self._emit("vad", {"state": "speech"})
                            else:
                                silence_ms += self.cfg.chunk_ms
                                self._emit("vad", {"state": "silence", "ms": silence_ms})

                        # stop conditions
                        total_ms = int(len(recording) * self.cfg.chunk_ms)
                        if total_ms >= int(self.cfg.max_utterance_s * 1000):
                            self._emit("vad", {"state": "max_len"})
                            mode = "DECODE"
                        elif started and silence_ms >= self.cfg.end_silence_ms:
                            mode = "DECODE"

                    if mode == "DECODE":
                        self._set_state("DECODING")
                        pcm = np.concatenate(recording, axis=0)
                        # optional trim: keep last max_utterance_s
                        max_n = int(self.cfg.target_sr * self.cfg.max_utterance_s)
                        if pcm.size > max_n:
                            pcm = pcm[-max_n:]
                        text = self._asr.transcribe(pcm)
                        self._emit("asr.final", {"text": text})
                        if text and self._on_final:
                            try:
                                self._on_final(text)
                            except Exception:
                                pass
                        # back to wake
                        self._set_state("IDLE")
                        mode = "WAIT_WAKE" if self.cfg.wake_enabled and self._wake else "LISTEN"
                        _reset_recording()
                        preroll.clear()
        except Exception as exc:
            self.logger.exception("Voice runtime crashed: %s", exc)
            self._emit("error", {"error": str(exc)})
            self._set_state("ERROR")
