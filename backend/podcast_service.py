"""100% local Podcast Studio: grounded two-host script generation & Kokoro-82M / Piper ONNX synthesis."""

from __future__ import annotations

import gc
import json
import re
import subprocess
import tempfile
import threading
import wave
from pathlib import Path
from typing import Any, Callable

import httpx
import numpy as np

from backend.config import (
    KOKORO_MODEL_URL,
    KOKORO_VOICES_URL,
    KOKORO_VOICE_PRESETS,
    PODCAST_VOICE_PRESETS,
    TTS_MODELS_DIR,
)

_HF_PIPER_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class PodcastCancelledError(Exception):
    """Raised when podcast generation is cancelled by the user."""


def sanitize_text_for_tts(text: str) -> str:
    """Clean Markdown formatting and bracket citations so spoken dialogue sounds natural."""
    clean = str(text or "")
    # Remove bracketed citations like [04:12], [12:34], [archivo.pdf · p. 4], [imagen.png]
    clean = re.sub(r"\[\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[-–,]\s*\d{1,2}:\d{2}(?::\d{2})?)*\s*\]", "", clean)
    clean = re.sub(
        r"\[\s*[^\[\]\n]{1,90}?\.(?:pdf|pptx|docx|txt|md|png|jpg|jpeg|webp|bmp)(?:\s*[·|]\s*[^\]\n]+)?\s*\]",
        "",
        clean,
        flags=re.IGNORECASE,
    )
    # Strip Markdown links [label](url) -> label
    clean = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", clean)
    # Strip code fences / inline backticks / headings / emphasis markers
    clean = clean.replace("```", " ").replace("`", "")
    clean = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", clean)
    clean = re.sub(r"\*\*([^*]+)\*\*", r"\1", clean)
    clean = re.sub(r"\*([^*\n]+)\*", r"\1", clean)
    clean = re.sub(r"_([^_\n]+)_", r"\1", clean)
    clean = clean.replace("*", "").replace("#", "")
    # Remove emojis / unusual symbols that confuse phonemizers
    clean = re.sub(r"[▶✦📄🖼•]", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def build_podcast_script_messages(
    recordings: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    focus_prompt: str = "",
    style: str = "conversation",
    length: str = "medium",
    language: str = "es",
    num_ctx: int = 8192,
    indexed_chunks: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Build prompt messages for generating a conversational two-host podcast script."""
    lang = "en" if str(language).lower().startswith("en") else "es"
    presets = PODCAST_VOICE_PRESETS[lang]
    host_a = presets["A"]["name"]
    host_b = presets["B"]["name"]

    turn_ranges = {
        "short": "8 a 10 intervenciones cortas (~2 minutos)",
        "medium": "14 a 16 intervenciones (~4 minutos)",
        "long": "20 a 24 intervenciones (~6 a 7 minutos)",
    }
    length_desc = turn_ranges.get(length, turn_ranges["medium"])

    style_instructions = {
        "conversation": (
            f"Conversación amena, natural y curiosa entre {host_a} (quien guía y conecta ideas) "
            f"y {host_b} (quien profundiza con ejemplos concretos y preguntas agudas)."
        ),
        "exam_review": (
            f"Sesión dinámica de repaso para examen: {host_a} plantea preguntas clave y posibles trampas del tema, "
            f"y {host_b} responde con claridad, trucos mnemotécnicos y puntos evaluables."
        ),
        "debate": (
            f"Debate analítico constructivo: {host_a} y {host_b} contrastan posturas, ventajas, limitaciones y casos reales presentes en las fuentes."
        ),
        "step_by_step": (
            f"Explicación didáctica paso a paso desde lo más básico hasta los detalles técnicos, asegurando que cualquier estudiante siga el hilo sin perderse."
        ),
    }
    style_desc = style_instructions.get(style, style_instructions["conversation"])

    budget = max(4500, int(num_ctx * 1.9))
    context_parts: list[str] = []
    used = 0

    if indexed_chunks:
        for chunk in indexed_chunks:
            header = f"[{chunk.get('filename', 'fuente')} · {chunk.get('locator', '')}]"
            block = f"{header}\n{(chunk.get('content') or '').strip()}"
            if used + len(block) > int(budget * 0.55) and context_parts:
                break
            context_parts.append(block)
            used += len(block) + 2

    remaining = max(1800, budget - used)
    items: list[tuple[str, str]] = []
    for r in recordings:
        summary = (r.get("summary_text") or "").strip()
        transcript = (r.get("transcript_text") or "").strip()
        items.append((f"Audio: {r.get('filename', 'grabación')}", summary or transcript[:1800]))
    for s in sources:
        summary = (s.get("summary_text") or "").strip()
        text = (s.get("extracted_text") or "").strip()
        items.append((f"Fuente: {s.get('filename', 'documento')}", summary or text[:1800]))

    if items:
        per_item = max(700, remaining // len(items))
        for label, body in items:
            snippet = body[:per_item]
            context_parts.append(f"=== {label} ===\n{snippet}")

    context_text = "\n\n".join(context_parts)
    focus_line = (
        f"ENFOQUE PRIORITARIO DEL USUARIO: {focus_prompt.strip()}\n"
        if focus_prompt and focus_prompt.strip()
        else "ENFOQUE: Cubre los conceptos centrales, conexiones entre fuentes y conclusiones más valiosas.\n"
    )

    tag_a = "CLAIRE" if lang == "en" else "SOFIA"
    tag_b = "RYAN" if lang == "en" else "MATEO"

    if lang == "en":
        lang_rule = "Write the entire spoken dialogue in natural, conversational English."
    else:
        lang_rule = "Escribe todo el diálogo hablado en español natural y fluido (evita anglicismos innecesarios salvo términos técnicos propios de la materia)."

    system_prompt = (
        f"Eres un guionista experto en podcasts educativos en formato de diálogo.\n"
        f"Escribe ÚNICAMENTE líneas de diálogo con el formato exacto '{tag_a}: texto' y '{tag_b}: texto', sin introducciones ni cierres fuera del guion."
    )
    user_prompt = (
        f"Crea un guion de podcast entre dos conductores ({host_a} y {host_b}):\n"
        f"Estilo: {style_desc}\n"
        f"Extensión objetivo: {length_desc}.\n"
        f"{focus_line}"
        f"{lang_rule}\n\n"
        "Reglas estrictas de formato para lectura en voz alta (TTS):\n"
        f"1. Escribe cada intervención en una línea separada empezando exactamente por `{tag_a}:` o `{tag_b}:`.\n"
        "Ejemplo:\n"
        f"{tag_a}: ¡Hola! Hoy vamos a conectar las ideas clave de nuestras fuentes de estudio.\n"
        f"{tag_b}: Así es, empecemos por el concepto central y cómo se aplica en la práctica.\n"
        "2. Alterna de forma natural entre ambos conductores; cada línea debe tener 1 a 3 oraciones listas para leerse en voz alta.\n"
        "3. NO devuelvas JSON, listas numeradas, asteriscos (**), títulos Markdown, emojis ni corchetes tipo [04:12] o [p. 3].\n"
        "4. Basa todo el contenido ÚNICAMENTE en las fuentes proporcionadas abajo.\n\n"
        f"FUENTES SELECCIONADAS:\n{context_text}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _normalize_speaker_label(raw_label: str, host_a: str, host_b: str, dynamic_map: dict[str, str]) -> str:
    """Map varied host labels (B, Mateo, Ryan, Host 2, dynamic names) to 'A' or 'B'."""
    norm = re.sub(r"[\*\[\]_\(\)`\"']", "", str(raw_label or "")).strip().lower()
    norm = (
        norm.replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("í", "i")
    )
    norm_clean = re.sub(r"\s+", " ", norm).strip(" .:-—–")
    b_tokens = {"b", "2", "mateo", "ryan", "host b", "host 2", "speaker b", "speaker 2", "locutor b", "locutor 2", "cohost", "co-host"}
    a_tokens = {"a", "1", "sofia", "claire", "host a", "host 1", "speaker a", "speaker 1", "locutor a", "locutor 1", "host"}
    if norm_clean in b_tokens or norm_clean == host_b.lower():
        return "B"
    if norm_clean in a_tokens or norm_clean == host_a.lower():
        return "A"
    if any(tok in norm_clean.split() for tok in ("mateo", "ryan", "2", "b")):
        return "B"
    if any(tok in norm_clean.split() for tok in ("sofia", "claire", "1", "a")):
        return "A"
    if norm_clean in dynamic_map:
        return dynamic_map[norm_clean]
    if not dynamic_map:
        dynamic_map[norm_clean] = "A"
        return "A"
    if len(dynamic_map) == 1:
        dynamic_map[norm_clean] = "B"
        return "B"
    return "B" if (len(dynamic_map) % 2) else "A"


def _strip_code_fences(raw_text: str) -> str:
    text = (raw_text or "").strip()
    text = re.sub(r"^```(?:json|text|markdown|md)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_turns_from_list(items: list[Any], host_a: str, host_b: str) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    dynamic_map: dict[str, str] = {}
    for item in items:
        if isinstance(item, dict):
            raw_sp = str(
                item.get("speaker")
                or item.get("host")
                or item.get("name")
                or item.get("role")
                or "A"
            )
            speaker = _normalize_speaker_label(raw_sp, host_a, host_b, dynamic_map)
            raw_line = (
                item.get("text")
                or item.get("line")
                or item.get("content")
                or item.get("dialogue")
                or item.get("message")
                or ""
            )
            text = sanitize_text_for_tts(str(raw_line))
        elif isinstance(item, str):
            text = sanitize_text_for_tts(item)
            speaker = "B" if (len(turns) % 2) else "A"
        else:
            continue
        if not text:
            continue
        turns.append({"speaker": speaker, "name": host_b if speaker == "B" else host_a, "text": text})
    return turns


def parse_podcast_script_json(raw_text: str, language: str = "es") -> list[dict[str, str]]:
    """Parse LLM-generated two-host dialogue from line format, JSON, broken JSON, or prose."""
    lang = "en" if str(language).lower().startswith("en") else "es"
    presets = PODCAST_VOICE_PRESETS[lang]
    host_a = presets["A"]["name"]
    host_b = presets["B"]["name"]

    cleaned = _strip_code_fences(raw_text)
    if not cleaned:
        return []

    # 1. Try full/embedded JSON array or wrapper object ({"dialogue": [...]})
    json_candidates: list[str] = [cleaned]
    arr_match = re.search(r"\[[\s\S]*\]", cleaned)
    if arr_match:
        json_candidates.append(arr_match.group(0))
    obj_match = re.search(r"\{[\s\S]*\}", cleaned)
    if obj_match:
        json_candidates.append(obj_match.group(0))

    for cand in json_candidates:
        try:
            parsed = json.loads(cand)
        except Exception:
            continue
        if isinstance(parsed, list):
            turns = _extract_turns_from_list(parsed, host_a, host_b)
            if turns:
                return turns
        elif isinstance(parsed, dict):
            for key in ("turns", "dialogue", "script", "podcast", "segments", "conversation", "items"):
                val = parsed.get(key)
                if isinstance(val, list):
                    turns = _extract_turns_from_list(val, host_a, host_b)
                    if turns:
                        return turns

    # 2. Try truncated/broken JSON object-by-object regex extraction
    if '"speaker"' in cleaned or '"text"' in cleaned or '"line"' in cleaned:
        regex_turns: list[dict[str, str]] = []
        dynamic_map: dict[str, str] = {}
        obj_chunks = re.findall(r"\{[^{}]+(?:\}|$)", cleaned)
        for chunk in obj_chunks:
            sp_m = re.search(r'"(?:speaker|host|name|role)"\s*:\s*"([^"]+)"', chunk)
            tx_m = re.search(
                r'"(?:text|line|content|dialogue|message)"\s*:\s*"((?:\\.|[^"\\])*)(?:"|$)',
                chunk,
                flags=re.DOTALL,
            )
            if not tx_m:
                continue
            raw_sp = sp_m.group(1) if sp_m else ("B" if len(regex_turns) % 2 else "A")
            speaker = _normalize_speaker_label(raw_sp, host_a, host_b, dynamic_map)
            raw_spoken = tx_m.group(1).replace('\\"', '"').replace("\\n", " ")
            spoken = sanitize_text_for_tts(raw_spoken)
            if spoken:
                regex_turns.append(
                    {
                        "speaker": speaker,
                        "name": host_b if speaker == "B" else host_a,
                        "text": spoken,
                    }
                )
        if regex_turns:
            return regex_turns

    # 3. Resilient line-by-line dialogue parser:
    # Supports "SOFIA: ...", "**Sofía:** ...", "1. **Mateo**: ...", "[Host B]: ...", "- Locutor 1 — ..."
    line_turns: list[dict[str, str]] = []
    dynamic_map = {}
    speaker_line_re = re.compile(
        r"^\s*(?:[-*•]\s*|\d+[.)]\s*)?"
        r"(?:\*\*|__|\[|\()?"
        r"([A-Za-zÁÉÍÓÚÜáéíóúüñÑ][A-Za-zÁÉÍÓÚÜáéíóúüñÑ0-9 _\-]{0,28}?)"
        r"(?:\*\*|__|\]|\))?"
        r"\s*(?::|\*\*:\s*|[—–])\s*"
        r"(.+?)\s*$"
    )
    non_speaker_prefixes = {"http", "https", "nota", "note", "tema", "resumen", "summary", "titulo", "title", "fuente", "source", "ejemplo", "example"}

    for raw_line in cleaned.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") or re.fullmatch(r"[\[\(].*[\]\)]", stripped):
            continue

        m = speaker_line_re.match(stripped)
        if m:
            raw_who = m.group(1).strip()
            low_who = raw_who.lower()
            if low_who not in non_speaker_prefixes and not m.group(2).lstrip().startswith("//"):
                spoken = sanitize_text_for_tts(m.group(2))
                if spoken:
                    speaker = _normalize_speaker_label(raw_who, host_a, host_b, dynamic_map)
                    line_turns.append(
                        {
                            "speaker": speaker,
                            "name": host_b if speaker == "B" else host_a,
                            "text": spoken,
                        }
                    )
                continue

        # Continuation line for the current active turn
        if line_turns and not stripped.startswith(("{", "}", "[", "]")):
            extra = sanitize_text_for_tts(stripped)
            if extra:
                line_turns[-1]["text"] = f"{line_turns[-1]['text']} {extra}".strip()

    if line_turns:
        return line_turns

    # 4. Final plain-prose fallback: split paragraphs / sentence pairs and alternate hosts
    paragraphs = [sanitize_text_for_tts(p) for p in re.split(r"\n+", cleaned) if p.strip()]
    paragraphs = [p for p in paragraphs if len(p) >= 15]
    if len(paragraphs) == 1 and len(paragraphs[0]) > 140:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraphs[0]) if len(s.strip()) >= 12]
        if len(sentences) >= 2:
            grouped = []
            for idx in range(0, len(sentences), 2):
                grouped.append(" ".join(sentences[idx : idx + 2]))
            paragraphs = grouped

    prose_turns: list[dict[str, str]] = []
    for idx, text in enumerate(paragraphs):
        speaker = "B" if (idx % 2) else "A"
        prose_turns.append(
            {
                "speaker": speaker,
                "name": host_b if speaker == "B" else host_a,
                "text": text,
            }
        )
    return prose_turns


def _check_podcast_cancel(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise PodcastCancelledError("Generación de podcast cancelada por el usuario.")


def ensure_kokoro_models(
    models_dir: Path = TTS_MODELS_DIR,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[Path, Path]:
    """Ensure local Kokoro-82M ONNX files exist inside data/models/tts/kokoro/."""
    _check_podcast_cancel(cancel_event)
    kokoro_dir = models_dir / "kokoro"
    kokoro_dir.mkdir(parents=True, exist_ok=True)

    direct_model = models_dir / "kokoro-v1.0.onnx"
    direct_voices = models_dir / "voices-v1.0.bin"
    if direct_model.is_file() and direct_voices.is_file():
        return direct_model, direct_voices

    model_path = kokoro_dir / "kokoro-v1.0.onnx"
    voices_path = kokoro_dir / "voices-v1.0.bin"
    files = [
        (voices_path, KOKORO_VOICES_URL, "voces Kokoro-82M"),
        (model_path, KOKORO_MODEL_URL, "modelo neuronal Kokoro-82M"),
    ]
    for target, url, label in files:
        if target.is_file() and target.stat().st_size > 1000:
            continue
        _check_podcast_cancel(cancel_event)
        if progress_callback:
            progress_callback(0.43, f"Descargando {label} local (primera vez)...")
        part = kokoro_dir / f"{target.name}.part"
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=300.0) as resp:
                resp.raise_for_status()
                with part.open("wb") as out_f:
                    for chunk in resp.iter_bytes(chunk_size=131072):
                        _check_podcast_cancel(cancel_event)
                        if chunk:
                            out_f.write(chunk)
            part.replace(target)
        except Exception:
            if part.exists():
                part.unlink(missing_ok=True)
            raise
    return model_path, voices_path


def ensure_piper_voice(
    voice_key: str,
    rel_path: str,
    models_dir: Path = TTS_MODELS_DIR,
    cancel_event: threading.Event | None = None,
) -> Path:
    """
    Ensure a Piper ONNX voice (.onnx + .onnx.json) exists locally in data/models/tts/.

    Downloads once from Hugging Face rhasspy/piper-voices on first use if not present.
    """
    _check_podcast_cancel(cancel_event)
    models_dir.mkdir(parents=True, exist_ok=True)
    # Check both direct file in models_dir and nested voice_key subfolder
    direct_onnx = models_dir / f"{voice_key}.onnx"
    direct_json = models_dir / f"{voice_key}.onnx.json"
    nested_onnx = models_dir / voice_key / f"{voice_key}.onnx"
    nested_json = models_dir / voice_key / f"{voice_key}.onnx.json"

    if direct_onnx.is_file() and direct_json.is_file():
        return direct_onnx
    if nested_onnx.is_file() and nested_json.is_file():
        return nested_onnx

    for suffix in (".onnx", ".onnx.json"):
        target = models_dir / f"{voice_key}{suffix}"
        if target.is_file() and target.stat().st_size > 100:
            continue
        part = models_dir / f"{voice_key}{suffix}.part"
        url = f"{_HF_PIPER_BASE}/{rel_path}{suffix}"
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=180.0) as resp:
                resp.raise_for_status()
                with part.open("wb") as out_f:
                    for chunk in resp.iter_bytes(chunk_size=65536):
                        _check_podcast_cancel(cancel_event)
                        if chunk:
                            out_f.write(chunk)
            part.replace(target)
        except Exception:
            if part.exists():
                part.unlink(missing_ok=True)
            raise

    return direct_onnx


def _resample_mono_float(pcm: np.ndarray, src_rate: int, target_rate: int) -> np.ndarray:
    arr = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if src_rate != target_rate and len(arr) > 1:
        new_len = max(1, int(round(len(arr) * target_rate / float(src_rate))))
        old_x = np.linspace(0.0, 1.0, len(arr), endpoint=False)
        new_x = np.linspace(0.0, 1.0, new_len, endpoint=False)
        arr = np.interp(new_x, old_x, arr).astype(np.float32)
    return arr


def _read_wav_as_float_mono(wav_path: Path, target_rate: int = 24000) -> np.ndarray:
    with wave.open(str(wav_path), "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if sampwidth != 2 or not frames:
        return np.zeros(0, dtype=np.float32)

    pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)

    return _resample_mono_float(pcm, rate, target_rate)


def _write_pcm16_wav(samples: np.ndarray, output_path: Path, sample_rate: int = 24000) -> float:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    if peak > 0.96:
        samples = samples * (0.94 / peak)
    clipped = np.clip(samples, -1.0, 1.0)
    int16_data = (clipped * 32767.0).astype(np.int16)

    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_data.tobytes())

    return round(len(int16_data) / float(sample_rate), 2)


def _synthesize_turn_sapi_wav(text: str, out_wav: Path, voice_idx: int) -> None:
    """Offline Windows System.Speech fallback if Kokoro/Piper voices cannot be loaded."""
    ps_script = r"""
param([string]$Txt, [string]$OutPath, [int]$VoiceIdx)
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voices = $synth.GetInstalledVoices() | Where-Object { $_.Enabled }
if ($voices.Count -gt 0) {
    $chosen = $voices[$VoiceIdx % $voices.Count].VoiceInfo.Name
    $synth.SelectVoice($chosen)
}
$synth.Rate = 1
$synth.SetOutputToWaveFile($OutPath)
$synth.Speak($Txt)
$synth.Dispose()
"""
    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            ps_script,
            "-Txt",
            text,
            "-OutPath",
            str(out_wav),
            "-VoiceIdx",
            str(voice_idx),
        ],
        capture_output=True,
        text=True,
        timeout=45,
        creationflags=_NO_WINDOW,
    )
    if completed.returncode != 0 or not out_wav.is_file():
        raise RuntimeError(f"SAPI TTS fallback failed: {completed.stderr.strip()}")


def synthesize_podcast_audio(
    turns: list[dict[str, str]],
    output_path: Path,
    language: str = "es",
    models_dir: Path = TTS_MODELS_DIR,
    progress_callback: Callable[[float, str], None] | None = None,
    cancel_event: threading.Event | None = None,
    cpu_threads: int = 0,
) -> tuple[float, str]:
    """
    Synthesize all dialogue turns into one local 24000 Hz mono WAV file.

    Uses Kokoro-82M ONNX primarily (100% offline/local natural neural voices),
    falling back to Piper ONNX or Windows SpeechSynthesizer if needed.
    """
    _check_podcast_cancel(cancel_event)
    if not turns:
        raise ValueError("El guion del podcast no contiene intervenciones.")

    lang = "en" if str(language).lower().startswith("en") else "es"
    kokoro_cfg = KOKORO_VOICE_PRESETS[lang]
    piper_presets = PODCAST_VOICE_PRESETS[lang]
    target_rate = 24000

    kokoro_engine = None
    voice_a = None
    voice_b = None
    engine_label = "kokoro-82m-onnx"

    if progress_callback:
        progress_callback(0.42, "Cargando voces neuronales locales (Kokoro-82M ONNX)...")

    try:
        from kokoro_onnx import Kokoro

        model_path, voices_path = ensure_kokoro_models(
            models_dir=models_dir,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
        )
        _check_podcast_cancel(cancel_event)
        if cpu_threads > 0 and hasattr(Kokoro, "from_session"):
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = max(1, int(cpu_threads))
            opts.inter_op_num_threads = 1
            sess = ort.InferenceSession(
                str(model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            kokoro_engine = Kokoro.from_session(sess, str(voices_path))
        else:
            kokoro_engine = Kokoro(str(model_path), str(voices_path))
    except PodcastCancelledError:
        raise
    except Exception:
        kokoro_engine = None

    if kokoro_engine is None:
        engine_label = "piper-onnx"
        if progress_callback:
            progress_callback(0.45, "Cargando voces neuronales locales (Piper ONNX)...")
        try:
            from piper.voice import PiperVoice

            onnx_a = ensure_piper_voice(
                piper_presets["A"]["key"],
                piper_presets["A"]["rel_path"],
                models_dir,
                cancel_event=cancel_event,
            )
            onnx_b = ensure_piper_voice(
                piper_presets["B"]["key"],
                piper_presets["B"]["rel_path"],
                models_dir,
                cancel_event=cancel_event,
            )
            voice_a = PiperVoice.load(str(onnx_a), config_path=f"{onnx_a}.json", use_cuda=False)
            voice_b = PiperVoice.load(str(onnx_b), config_path=f"{onnx_b}.json", use_cuda=False)
        except PodcastCancelledError:
            raise
        except Exception:
            voice_a = None
            voice_b = None
            engine_label = "windows-sapi"

    audio_segments: list[np.ndarray] = []
    prev_speaker = ""

    try:
        with tempfile.TemporaryDirectory(prefix="podcast_tts_") as tmp_dir:
            tmp_root = Path(tmp_dir)
            total = len(turns)
            for idx, turn in enumerate(turns, start=1):
                _check_podcast_cancel(cancel_event)
                spoken = sanitize_text_for_tts(turn.get("text", ""))
                if not spoken:
                    continue
                speaker = "B" if turn.get("speaker") == "B" else "A"
                if progress_callback:
                    pct = 0.48 + (idx / float(max(1, total))) * 0.42
                    progress_callback(
                        pct,
                        f"Sintetizando diálogo ({idx}/{total} · {turn.get('name', speaker)})...",
                    )

                if kokoro_engine is not None:
                    v_spec = kokoro_cfg[speaker]
                    raw_pcm, sr = kokoro_engine.create(
                        spoken,
                        voice=v_spec["voice"],
                        speed=float(v_spec.get("speed", 1.0)),
                        lang=kokoro_cfg["lang"],
                    )
                    pcm = _resample_mono_float(raw_pcm, int(sr), target_rate)
                else:
                    turn_wav = tmp_root / f"turn_{idx:03d}.wav"
                    if voice_a is not None and voice_b is not None:
                        voice = voice_b if speaker == "B" else voice_a
                        with wave.open(str(turn_wav), "wb") as wf:
                            voice.synthesize_wav(spoken, wf)
                    else:
                        _synthesize_turn_sapi_wav(spoken, turn_wav, 1 if speaker == "B" else 0)
                    pcm = _read_wav_as_float_mono(turn_wav, target_rate=target_rate)

                _check_podcast_cancel(cancel_event)
                if len(pcm) > 0:
                    if audio_segments:
                        gap_sec = 0.34 if speaker != prev_speaker else 0.16
                        audio_segments.append(np.zeros(int(target_rate * gap_sec), dtype=np.float32))
                    audio_segments.append(pcm)
                    prev_speaker = speaker

        _check_podcast_cancel(cancel_event)
        if progress_callback:
            progress_callback(0.94, "Ensamblando episodio final...")

        if not audio_segments:
            raise RuntimeError("No se pudo sintetizar audio para ninguna intervención.")

        combined = np.concatenate(audio_segments)
        duration_sec = _write_pcm16_wav(combined, output_path, sample_rate=target_rate)
        return duration_sec, engine_label
    finally:
        del kokoro_engine
        del voice_a
        del voice_b
        gc.collect()
