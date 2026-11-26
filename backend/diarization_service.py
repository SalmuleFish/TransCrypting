"""Local ONNX speaker diarization and call dual-track speaker merging."""

from __future__ import annotations

import difflib
from importlib.util import find_spec
import logging
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import threading
import urllib.request
import wave
from typing import Any

import numpy as np

from backend.config import (
    DIARIZATION_EMB_FALLBACK_FILENAME,
    DIARIZATION_EMB_FALLBACK_URL,
    DIARIZATION_EMB_FILENAME,
    DIARIZATION_EMB_URL,
    DIARIZATION_MODELS_DIR,
    DIARIZATION_SEG_DIR,
    DIARIZATION_SEG_URL,
)
from backend.whisper_service import _find_ffmpeg_exe

logger = logging.getLogger(__name__)

_DOWNLOAD_LOCK = threading.Lock()


def is_diarization_runtime_available() -> bool:
    """Return True if sherpa-onnx is importable in the current environment."""
    return find_spec("sherpa_onnx") is not None


def _resolve_seg_model_file(seg_dir: Path) -> Path | None:
    for name in ("model.onnx", "model.int8.onnx"):
        cand = seg_dir / name
        if cand.is_file() and cand.stat().st_size > 100_000:
            return cand
    return None


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": "TransCrypting/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp, tmp_path.open("wb") as out:
            shutil.copyfileobj(resp, out, length=256 * 1024)
        tmp_path.replace(dest)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _extract_tar_bz2_safe(archive: Path, target_root: Path) -> None:
    root_resolved = target_root.resolve()
    with tarfile.open(archive, "r:bz2") as tf:
        for member in tf.getmembers():
            member_path = (root_resolved / member.name).resolve()
            if root_resolved != member_path and root_resolved not in member_path.parents:
                raise RuntimeError(f"Ruta insegura en archivo de diarización: {member.name}")
        tf.extractall(root_resolved)


def ensure_diarization_models() -> tuple[Path, Path]:
    """Ensure Pyannote segmentation and speaker embedding ONNX files exist locally."""
    with _DOWNLOAD_LOCK:
        DIARIZATION_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        seg_model = _resolve_seg_model_file(DIARIZATION_SEG_DIR)
        if seg_model is None:
            archive_path = DIARIZATION_MODELS_DIR / "pyannote-seg-3-0.tar.bz2"
            logger.info("Descargando modelo de segmentación Pyannote 3.0 ONNX...")
            _download_file(DIARIZATION_SEG_URL, archive_path)
            try:
                _extract_tar_bz2_safe(archive_path, DIARIZATION_MODELS_DIR)
            finally:
                archive_path.unlink(missing_ok=True)
            seg_model = _resolve_seg_model_file(DIARIZATION_SEG_DIR)
            if seg_model is None:
                raise RuntimeError("No se encontró model.onnx tras extraer Pyannote Segmentation 3.0.")

        emb_model = DIARIZATION_MODELS_DIR / DIARIZATION_EMB_FILENAME
        fallback_model = DIARIZATION_MODELS_DIR / DIARIZATION_EMB_FALLBACK_FILENAME
        if emb_model.is_file() and emb_model.stat().st_size > 100_000:
            return seg_model, emb_model
        if fallback_model.is_file() and fallback_model.stat().st_size > 100_000:
            return seg_model, fallback_model

        try:
            logger.info("Descargando modelo de embeddings de voz (%s)...", DIARIZATION_EMB_FILENAME)
            _download_file(DIARIZATION_EMB_URL, emb_model)
            return seg_model, emb_model
        except Exception as primary_exc:
            logger.warning("Fallo descargando %s (%s); probando respaldo...", DIARIZATION_EMB_FILENAME, primary_exc)
            _download_file(DIARIZATION_EMB_FALLBACK_URL, fallback_model)
            return seg_model, fallback_model


def load_audio_mono_16k(audio_path: Path) -> np.ndarray:
    """Decode any supported audio file to 16 kHz mono float32 samples in [-1, 1]."""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio no encontrado: {audio_path}")

    if audio_path.suffix.lower() == ".wav":
        try:
            with wave.open(str(audio_path), "rb") as wf:
                if wf.getframerate() == 16000 and wf.getnchannels() == 1 and wf.getsampwidth() == 2:
                    raw = wf.readframes(wf.getnframes())
                    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception:
            pass

    ffmpeg_exe = _find_ffmpeg_exe()
    with tempfile.TemporaryDirectory(prefix="transcrypting-diar-") as tmpdir:
        wav_out = Path(tmpdir) / "mono16k.wav"
        cmd = [
            ffmpeg_exe,
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(audio_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            str(wav_out),
        ]
        subprocess.run(
            cmd,
            capture_output=True,
            check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        with wave.open(str(wav_out), "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def has_speech_energy(samples: np.ndarray, threshold_rms: float = 0.004) -> bool:
    """Return True if at least two 250ms windows contain non-trivial speech energy."""
    if len(samples) < 4000:
        return False
    win = 4000  # 0.25s at 16 kHz
    usable = (len(samples) // win) * win
    if usable <= 0:
        return float(np.sqrt(np.mean(samples ** 2))) >= threshold_rms
    blocks = samples[:usable].reshape(-1, win)
    rms = np.sqrt(np.mean(blocks ** 2, axis=1))
    return int(np.sum(rms >= threshold_rms)) >= 2


def diarize_audio(
    audio_path: Path,
    num_threads: int = 2,
    threshold: float = 0.52,
) -> list[dict[str, Any]]:
    """Run local sherpa-onnx diarization returning chronological speaker turns."""
    if not is_diarization_runtime_available():
        return []

    samples = load_audio_mono_16k(audio_path)
    if len(samples) < 8000 or not has_speech_energy(samples):
        return []

    import sherpa_onnx  # type: ignore[import-untyped]

    seg_model, emb_model = ensure_diarization_models()
    threads = max(1, min(8, int(num_threads or 2)))

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(seg_model),
            ),
            num_threads=threads,
            debug=False,
            provider="cpu",
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(emb_model),
            num_threads=threads,
            debug=False,
            provider="cpu",
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=-1,
            threshold=float(threshold),
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise RuntimeError("Configuración inválida de diarización sherpa-onnx.")

    diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
    result = diarizer.process(samples)
    raw_turns = result.sort_by_start_time()
    if not raw_turns:
        return []

    id_map: dict[int, int] = {}
    turns: list[dict[str, Any]] = []
    for seg in raw_turns:
        start = round(float(seg.start), 3)
        end = round(float(seg.end), 3)
        if end <= start:
            continue
        raw_id = int(seg.speaker)
        if raw_id not in id_map:
            id_map[raw_id] = len(id_map) + 1
        idx = id_map[raw_id]
        turns.append(
            {
                "start": start,
                "end": end,
                "speaker": f"Hablante {idx}",
                "speaker_key": f"spk_{idx}",
            }
        )
    return turns


def _best_speaker_for_span(
    start: float,
    end: float,
    diar_turns: list[dict[str, Any]],
    fallback_max_dist: float = 1.5,
) -> str | None:
    best_overlap = 0.0
    best_speaker: str | None = None
    for turn in diar_turns:
        ov = min(end, float(turn["end"])) - max(start, float(turn["start"]))
        if ov > best_overlap:
            best_overlap = ov
            best_speaker = str(turn["speaker"])

    if best_speaker is not None:
        return best_speaker

    mid = (start + end) * 0.5
    nearest_dist = fallback_max_dist
    nearest_speaker: str | None = None
    for turn in diar_turns:
        t_start = float(turn["start"])
        t_end = float(turn["end"])
        dist = 0.0 if t_start <= mid <= t_end else min(abs(mid - t_start), abs(mid - t_end))
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_speaker = str(turn["speaker"])
    return nearest_speaker


def _maybe_split_segment_by_words(
    seg: dict[str, Any],
    diar_turns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Split a long ASR segment at a clean speaker boundary when word timings exist."""
    words = seg.get("words")
    text = (seg.get("text") or "").strip()
    if not words or not isinstance(words, list) or len(words) < 6 or not text:
        return []

    tokens = text.split()
    if len(tokens) != len(words):
        return []

    tagged: list[str] = []
    prev_spk: str | None = None
    for w in words:
        w_start = float(w.get("start", seg.get("start", 0.0)))
        w_end = float(w.get("end", w_start + 0.1))
        spk = _best_speaker_for_span(w_start, w_end, diar_turns) or prev_spk or "Hablante 1"
        tagged.append(spk)
        prev_spk = spk

    runs: list[tuple[str, int, int]] = []
    run_start = 0
    for idx in range(1, len(tagged)):
        if tagged[idx] != tagged[run_start]:
            runs.append((tagged[run_start], run_start, idx))
            run_start = idx
    runs.append((tagged[run_start], run_start, len(tagged)))

    if len(runs) <= 1 or len(runs) > 3:
        return []

    # Verify every sub-run is substantial (>= 3 words and >= 1.0s) to avoid choppy fragmentation
    split_segments: list[dict[str, Any]] = []
    for spk, a, b in runs:
        sub_words = words[a:b]
        s_start = round(float(sub_words[0].get("start", seg.get("start", 0.0))), 2)
        s_end = round(float(sub_words[-1].get("end", seg.get("end", s_start))), 2)
        if (b - a) < 3 or (s_end - s_start) < 1.0:
            return []
        split_segments.append(
            {
                "start": s_start,
                "end": s_end,
                "text": " ".join(tokens[a:b]),
                "words": sub_words,
                "speaker": spk,
            }
        )
    return split_segments


def assign_speakers_to_segments(
    asr_segments: list[dict[str, Any]],
    diar_turns: list[dict[str, Any]],
    speaker_prefix: str = "Hablante",
) -> list[dict[str, Any]]:
    """Attach speaker labels to ASR segments using overlap and word-level split when clean."""
    if not asr_segments:
        return []
    if not diar_turns:
        return [dict(seg) for seg in asr_segments]

    assigned: list[dict[str, Any]] = []
    last_speaker = f"{speaker_prefix} 1"

    for seg in asr_segments:
        split_out = _maybe_split_segment_by_words(seg, diar_turns)
        if split_out:
            last_speaker = split_out[-1]["speaker"]
            assigned.extend(split_out)
            continue

        item = dict(seg)
        start = float(item.get("start", 0.0))
        end = max(start + 0.05, float(item.get("end", start + 0.05)))
        spk = _best_speaker_for_span(start, end, diar_turns) or last_speaker
        last_speaker = spk
        item["speaker"] = spk
        assigned.append(item)

    # Renormalize speaker numbers in chronological first-appearance order
    remap: dict[str, int] = {}
    for item in assigned:
        raw_spk = str(item.get("speaker") or "").strip() or f"{speaker_prefix} 1"
        if raw_spk not in remap:
            remap[raw_spk] = len(remap) + 1
        num = remap[raw_spk]
        item["speaker"] = f"{speaker_prefix} {num}"
        item["speaker_key"] = f"spk_{num}"

    return assigned


def _normalize_for_bleed(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in (text or "")).split())


def dedupe_call_bleed(
    mic_segments: list[dict[str, Any]],
    sys_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop mic segments that are clear acoustic echo from speaker playback during a call."""
    if not mic_segments or not sys_segments:
        return [dict(s) for s in mic_segments]

    kept: list[dict[str, Any]] = []
    for m in mic_segments:
        m_start = float(m.get("start", 0.0))
        m_end = max(m_start + 0.05, float(m.get("end", m_start + 0.05)))
        m_dur = m_end - m_start
        m_norm = _normalize_for_bleed(str(m.get("text") or ""))
        is_bleed = False

        if m_norm:
            for s in sys_segments:
                s_start = float(s.get("start", 0.0))
                s_end = max(s_start + 0.05, float(s.get("end", s_start + 0.05)))
                ov = min(m_end, s_end) - max(m_start, s_start)
                if ov / m_dur < 0.60:
                    continue
                s_norm = _normalize_for_bleed(str(s.get("text") or ""))
                if not s_norm:
                    continue
                sim = difflib.SequenceMatcher(None, m_norm, s_norm).ratio()
                if sim >= 0.72:
                    is_bleed = True
                    break

        if not is_bleed:
            kept.append(dict(m))
    return kept


def merge_call_segments(
    mic_segments: list[dict[str, Any]],
    sys_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Combine user mic segments ('Tú') and diarized system audio segments chronologically."""
    clean_mic = dedupe_call_bleed(mic_segments, sys_segments)
    merged: list[dict[str, Any]] = []

    for seg in clean_mic:
        item = dict(seg)
        item["speaker"] = "Tú"
        item["speaker_key"] = "me"
        merged.append(item)

    for seg in sys_segments:
        item = dict(seg)
        if not item.get("speaker"):
            item["speaker"] = "Hablante 1"
            item["speaker_key"] = "spk_1"
        merged.append(item)

    merged.sort(key=lambda s: (float(s.get("start", 0.0)), float(s.get("end", 0.0))))
    return merged


def format_segments_plain_text(segments: list[dict[str, Any]]) -> str:
    """Format segments into readable plain text, grouping speaker turns when present."""
    if not segments:
        return ""
    has_speakers = any(bool((s.get("speaker") or "").strip()) for s in segments)
    if not has_speakers:
        return " ".join((s.get("text") or "").strip() for s in segments if (s.get("text") or "").strip())

    lines: list[str] = []
    cur_spk: str | None = None
    cur_buf: list[str] = []
    for seg in segments:
        txt = (seg.get("text") or "").strip()
        if not txt:
            continue
        spk = (seg.get("speaker") or "").strip()
        if spk == cur_spk:
            cur_buf.append(txt)
        else:
            if cur_buf:
                prefix = f"{cur_spk}: " if cur_spk else ""
                lines.append(f"{prefix}{' '.join(cur_buf)}")
            cur_spk = spk
            cur_buf = [txt]

    if cur_buf:
        prefix = f"{cur_spk}: " if cur_spk else ""
        lines.append(f"{prefix}{' '.join(cur_buf)}")

    return "\n".join(lines)
