"""Opt-in Windows WASAPI system audio loopback & mic capture module."""

from __future__ import annotations

from collections import deque
import logging
import math
from pathlib import Path
import sys
import threading
import time
import wave
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)

CaptureMode = Literal["system", "both"]

TARGET_SAMPLE_RATE = 44100
TICK_INTERVAL_SEC = 0.1
LIVE_BUFFER_SEC = 14
MAX_JITTER_BACKLOG_SEC = 0.5

try:
    import pyaudiowpatch as pa  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - non-Windows or missing wheel
    pa = None


def is_system_audio_available() -> bool:
    """Return True if native WASAPI loopback capture is available on this host."""
    if sys.platform != "win32" or pa is None:
        return False
    p = None
    try:
        p = pa.PyAudio()
        return _find_loopback_device(p) is not None
    except Exception:
        return False
    finally:
        if p is not None:
            try:
                p.terminate()
            except Exception:
                pass


def _find_loopback_device(p: Any) -> dict[str, Any] | None:
    """Locate the active Windows default WASAPI output loopback device."""
    try:
        dev = p.get_default_wasapi_loopback()
        if dev and int(dev.get("maxInputChannels", 0)) > 0:
            return dict(dev)
    except Exception:
        pass

    try:
        wasapi_info = p.get_host_api_info_by_type(pa.paWASAPI)
        default_out_idx = int(wasapi_info.get("defaultOutputDevice", -1))
        default_out_name = ""
        if default_out_idx >= 0:
            default_out_name = str(p.get_device_info_by_index(default_out_idx).get("name", ""))
        candidates = [
            dict(d)
            for d in p.get_device_info_generator()
            if d.get("isLoopbackDevice") and int(d.get("maxInputChannels", 0)) > 0
        ]
        if not candidates:
            return None
        if default_out_name:
            for cand in candidates:
                if default_out_name in str(cand.get("name", "")):
                    return cand
        return candidates[0]
    except Exception:
        return None


def _find_default_mic_device(p: Any) -> dict[str, Any] | None:
    """Locate the default Windows WASAPI (or fallback) microphone input device."""
    try:
        wasapi_info = p.get_host_api_info_by_type(pa.paWASAPI)
        idx = int(wasapi_info.get("defaultInputDevice", -1))
        if idx >= 0:
            info = dict(p.get_device_info_by_index(idx))
            if int(info.get("maxInputChannels", 0)) > 0 and not info.get("isLoopbackDevice"):
                return info
    except Exception:
        pass
    try:
        info = dict(p.get_default_input_device_info())
        if int(info.get("maxInputChannels", 0)) > 0:
            return info
    except Exception:
        pass
    return None


def resample_mono(
    pcm_bytes: bytes,
    src_rate: int,
    channels: int,
    dst_rate: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """Convert raw 16-bit PCM bytes (mono or multi-channel) to mono int16 at dst_rate."""
    if not pcm_bytes or src_rate <= 0 or dst_rate <= 0 or channels <= 0:
        return np.zeros(0, dtype=np.int16)

    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    usable = (len(samples) // channels) * channels
    if usable <= 0:
        return np.zeros(0, dtype=np.int16)
    samples = samples[:usable]

    if channels > 1:
        mono = samples.reshape(-1, channels).astype(np.float32).mean(axis=1)
    else:
        mono = samples.astype(np.float32)

    if src_rate != dst_rate and len(mono) > 1:
        dst_len = max(1, int(round(len(mono) * dst_rate / src_rate)))
        src_x = np.linspace(0.0, 1.0, num=len(mono), endpoint=False)
        dst_x = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
        mono = np.interp(dst_x, src_x, mono)

    return np.clip(np.rint(mono), -32768, 32767).astype(np.int16)


def mix_pcm_frames(sys_samples: np.ndarray, mic_samples: np.ndarray) -> np.ndarray:
    """Sum two mono int16 arrays safely with soft clamping to avoid clipping."""
    length = max(len(sys_samples), len(mic_samples))
    if length == 0:
        return np.zeros(0, dtype=np.int16)

    a = np.zeros(length, dtype=np.int32)
    b = np.zeros(length, dtype=np.int32)
    if len(sys_samples) > 0:
        a[: len(sys_samples)] = sys_samples.astype(np.int32)
    if len(mic_samples) > 0:
        b[: len(mic_samples)] = mic_samples.astype(np.int32)

    return np.clip(a + b, -32768, 32767).astype(np.int16)


class _PcmRing:
    """Thread-safe queue of int16 samples with bounded real-time latency."""

    def __init__(self, max_samples: int) -> None:
        self._chunks: deque[np.ndarray] = deque()
        self._total = 0
        self._max_samples = max_samples
        self._lock = threading.Lock()

    def push(self, arr: np.ndarray) -> None:
        if len(arr) == 0:
            return
        with self._lock:
            self._chunks.append(arr)
            self._total += len(arr)
            while self._total > self._max_samples and self._chunks:
                dropped = self._chunks.popleft()
                self._total -= len(dropped)

    def pop_exact(self, count: int) -> np.ndarray:
        """Pop up to count samples, right-padding with zeros if buffer has fewer."""
        if count <= 0:
            return np.zeros(0, dtype=np.int16)
        out = np.zeros(count, dtype=np.int16)
        pos = 0
        with self._lock:
            while pos < count and self._chunks:
                head = self._chunks[0]
                take = min(count - pos, len(head))
                out[pos : pos + take] = head[:take]
                pos += take
                self._total -= take
                if take == len(head):
                    self._chunks.popleft()
                else:
                    self._chunks[0] = head[take:]
        return out


class SystemAudioRecorder:
    """Thread-safe singleton managing an active Windows WASAPI loopback recording."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._recording = False
        self._mode: CaptureMode = "system"
        self._session_id: str | None = None
        self._rec_id: str | None = None
        self._wav_path: Path | None = None
        self._tracks_path: Path | None = None
        self._device_name: str = ""
        self._started_at: float = 0.0
        self._level: float = 0.0
        self._written_samples: int = 0

        self._stop_event = threading.Event()
        self._writer_thread: threading.Thread | None = None
        self._pa_instance: Any = None
        self._sys_stream: Any = None
        self._mic_stream: Any = None
        self._wav_file: wave.Wave_write | None = None
        self._tracks_wav: wave.Wave_write | None = None

        max_backlog = int(TARGET_SAMPLE_RATE * MAX_JITTER_BACKLOG_SEC)
        self._sys_ring = _PcmRing(max_backlog)
        self._mic_ring = _PcmRing(max_backlog)
        self._live_lock = threading.Lock()
        self._live_chunks: deque[np.ndarray] = deque()
        self._live_samples = 0
        self._max_live_samples = TARGET_SAMPLE_RATE * LIVE_BUFFER_SEC

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            elapsed = (time.monotonic() - self._started_at) if self._recording else 0.0
            return {
                "recording": self._recording,
                "available": is_system_audio_available(),
                "mode": self._mode if self._recording else None,
                "session_id": self._session_id if self._recording else None,
                "rec_id": self._rec_id if self._recording else None,
                "elapsed_sec": round(max(0.0, elapsed), 2),
                "level": round(self._level, 3) if self._recording else 0.0,
                "device_name": self._device_name if self._recording else "",
            }

    def start(
        self,
        session_id: str,
        rec_id: str,
        output_path: Path,
        mode: CaptureMode = "system",
    ) -> dict[str, Any]:
        """Start capturing WASAPI loopback (and optionally microphone) to output_path."""
        if sys.platform != "win32" or pa is None:
            raise RuntimeError("La captura de audio del equipo solo está disponible en Windows.")
        if mode not in ("system", "both"):
            raise ValueError(f"Modo de captura no válido: {mode}")

        with self._lock:
            if self._recording:
                raise RuntimeError("Ya hay una captura de audio del equipo en curso.")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            max_backlog = int(TARGET_SAMPLE_RATE * MAX_JITTER_BACKLOG_SEC)
            self._sys_ring = _PcmRing(max_backlog)
            self._mic_ring = _PcmRing(max_backlog)
            with self._live_lock:
                self._live_chunks.clear()
                self._live_samples = 0

            p = pa.PyAudio()
            try:
                loop_dev = _find_loopback_device(p)
                if not loop_dev:
                    raise RuntimeError(
                        "No se encontró un dispositivo de salida WASAPI para capturar audio del equipo."
                    )

                sys_rate = int(round(float(loop_dev["defaultSampleRate"])))
                sys_ch = max(1, int(loop_dev["maxInputChannels"]))
                sys_idx = int(loop_dev["index"])

                mic_dev = _find_default_mic_device(p) if mode == "both" else None
                if mode == "both" and not mic_dev:
                    raise RuntimeError(
                        "No se encontró un micrófono activo para combinar con el audio del equipo."
                    )

                wav_f = wave.open(str(output_path), "wb")
                wav_f.setnchannels(1)
                wav_f.setsampwidth(2)
                wav_f.setframerate(TARGET_SAMPLE_RATE)

                tracks_wav = None
                tracks_path: Path | None = None
                if mode == "both":
                    tracks_path = output_path.with_suffix(".tracks.wav")
                    tracks_wav = wave.open(str(tracks_path), "wb")
                    tracks_wav.setnchannels(2)
                    tracks_wav.setsampwidth(2)
                    tracks_wav.setframerate(16000)

                def _sys_callback(in_data, _frame_count, _time_info, _status):
                    if in_data and not self._stop_event.is_set():
                        arr = resample_mono(in_data, sys_rate, sys_ch, TARGET_SAMPLE_RATE)
                        self._sys_ring.push(arr)
                    return (None, pa.paContinue)

                sys_stream = p.open(
                    format=pa.paInt16,
                    channels=sys_ch,
                    rate=sys_rate,
                    frames_per_buffer=max(256, int(sys_rate * 0.02)),
                    input=True,
                    input_device_index=sys_idx,
                    stream_callback=_sys_callback,
                )

                mic_stream = None
                if mic_dev is not None:
                    mic_rate = int(round(float(mic_dev["defaultSampleRate"])))
                    mic_ch = min(2, max(1, int(mic_dev["maxInputChannels"])))
                    mic_idx = int(mic_dev["index"])

                    def _mic_callback(in_data, _frame_count, _time_info, _status):
                        if in_data and not self._stop_event.is_set():
                            arr = resample_mono(in_data, mic_rate, mic_ch, TARGET_SAMPLE_RATE)
                            self._mic_ring.push(arr)
                        return (None, pa.paContinue)

                    mic_stream = p.open(
                        format=pa.paInt16,
                        channels=mic_ch,
                        rate=mic_rate,
                        frames_per_buffer=max(256, int(mic_rate * 0.02)),
                        input=True,
                        input_device_index=mic_idx,
                        stream_callback=_mic_callback,
                    )

                self._pa_instance = p
                self._sys_stream = sys_stream
                self._mic_stream = mic_stream
                self._wav_file = wav_f
                self._wav_path = output_path
                self._tracks_wav = tracks_wav
                self._tracks_path = tracks_path
                self._mode = mode
                self._session_id = session_id
                self._rec_id = rec_id
                self._device_name = str(loop_dev.get("name", "WASAPI Loopback"))
                self._started_at = time.monotonic()
                self._written_samples = 0
                self._level = 0.0
                self._stop_event.clear()
                self._recording = True

                sys_stream.start_stream()
                if mic_stream is not None:
                    mic_stream.start_stream()

                self._writer_thread = threading.Thread(
                    target=self._writer_loop,
                    name="wasapi-writer",
                    daemon=True,
                )
                self._writer_thread.start()
                return self.get_status()
            except Exception:
                self._cleanup_resources()
                try:
                    p.terminate()
                except Exception:
                    pass
                raise

    def _writer_loop(self) -> None:
        while not self._stop_event.wait(TICK_INTERVAL_SEC):
            self._flush_tick()

    def _flush_tick(self) -> None:
        with self._lock:
            if not self._recording or self._wav_file is None:
                return
            elapsed = max(0.0, time.monotonic() - self._started_at)
            target_total = int(elapsed * TARGET_SAMPLE_RATE)
            need = max(0, target_total - self._written_samples)
            if need <= 0:
                return

            sys_chunk = self._sys_ring.pop_exact(need)
            if self._mode == "both":
                mic_chunk = self._mic_ring.pop_exact(need)
                mixed = mix_pcm_frames(sys_chunk, mic_chunk)
                if self._tracks_wav is not None:
                    mic_16k = resample_mono(mic_chunk.tobytes(), TARGET_SAMPLE_RATE, 1, 16000)
                    sys_16k = resample_mono(sys_chunk.tobytes(), TARGET_SAMPLE_RATE, 1, 16000)
                    n_16k = min(len(mic_16k), len(sys_16k))
                    if n_16k > 0:
                        stereo = np.empty((n_16k, 2), dtype=np.int16)
                        stereo[:, 0] = mic_16k[:n_16k]
                        stereo[:, 1] = sys_16k[:n_16k]
                        self._tracks_wav.writeframes(stereo.tobytes())
            else:
                mixed = sys_chunk

            self._wav_file.writeframes(mixed.tobytes())
            self._written_samples += len(mixed)

            rms = float(np.sqrt(np.mean(mixed.astype(np.float32) ** 2))) if len(mixed) else 0.0
            norm = min(1.0, math.sqrt(rms / 9000.0)) if rms > 1.0 else 0.0
            self._level = self._level * 0.45 + norm * 0.55

            with self._live_lock:
                self._live_chunks.append(mixed)
                self._live_samples += len(mixed)
                while self._live_samples > self._max_live_samples and self._live_chunks:
                    old = self._live_chunks.popleft()
                    self._live_samples -= len(old)

    def export_live_fragment(self, output_path: Path) -> bool:
        """Write the rolling recent audio window to a temporary WAV for live ASR."""
        with self._lock:
            if not self._recording:
                return False
        with self._live_lock:
            if self._live_samples < TARGET_SAMPLE_RATE:
                return False
            joined = np.concatenate(list(self._live_chunks))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(TARGET_SAMPLE_RATE)
            wf.writeframes(joined.tobytes())
        return True

    def stop(self) -> dict[str, Any]:
        """Stop capturing, flush remaining audio to WAV, and return capture metadata."""
        writer: threading.Thread | None = None
        with self._lock:
            if not self._recording:
                return {
                    "recording": False,
                    "session_id": None,
                    "rec_id": None,
                    "wav_path": None,
                    "duration_sec": 0.0,
                }
            self._stop_event.set()
            writer = self._writer_thread

        if writer is not None and writer.is_alive():
            writer.join(timeout=2.0)

        with self._lock:
            self._flush_tick()
            meta = {
                "recording": False,
                "session_id": self._session_id,
                "rec_id": self._rec_id,
                "mode": self._mode,
                "wav_path": self._wav_path,
                "tracks_path": self._tracks_path,
                "duration_sec": round(self._written_samples / TARGET_SAMPLE_RATE, 2),
            }
            self._cleanup_resources()
            return meta

    def _cleanup_resources(self) -> None:
        self._recording = False
        self._level = 0.0
        for stream in (self._sys_stream, self._mic_stream):
            if stream is not None:
                try:
                    stream.stop_stream()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass
        self._sys_stream = None
        self._mic_stream = None

        if self._wav_file is not None:
            try:
                self._wav_file.close()
            except Exception:
                pass
            self._wav_file = None

        if self._tracks_wav is not None:
            try:
                self._tracks_wav.close()
            except Exception:
                pass
            self._tracks_wav = None
        self._tracks_path = None

        if self._pa_instance is not None:
            try:
                self._pa_instance.terminate()
            except Exception:
                pass
            self._pa_instance = None


def split_dual_track_wav(tracks_path: Path, out_dir: Path) -> tuple[Path, Path]:
    """Split a stereo 16 kHz call WAV into Ch0 (mic.wav) and Ch1 (system.wav) mono WAVs."""
    tracks_path = Path(tracks_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with wave.open(str(tracks_path), "rb") as wf:
        if wf.getnchannels() != 2:
            raise ValueError(f"Se esperaba audio estéreo de 2 canales en {tracks_path}")
        rate = wf.getframerate()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    interleaved = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
    mic_path = out_dir / "mic.wav"
    sys_path = out_dir / "system.wav"
    for path, col in ((mic_path, 0), (sys_path, 1)):
        with wave.open(str(path), "wb") as out_wf:
            out_wf.setnchannels(1)
            out_wf.setsampwidth(sampwidth)
            out_wf.setframerate(rate)
            out_wf.writeframes(np.ascontiguousarray(interleaved[:, col]).tobytes())
    return mic_path, sys_path


system_audio_recorder = SystemAudioRecorder()
