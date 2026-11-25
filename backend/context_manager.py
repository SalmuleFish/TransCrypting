"""Prompt construction and local BM25 context management for multi-recording sessions."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from typing import Any

_STOPWORDS_ES_EN = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al",
    "a", "en", "con", "por", "para", "y", "o", "que", "es", "son", "fue", "se",
    "su", "sus", "lo", "le", "les", "me", "te", "nos", "como", "mas",
    "pero", "si", "no", "ya", "muy", "este", "esta", "estos", "estas", "ese",
    "esa", "eso", "sobre", "entre", "hasta", "desde", "hay", "ser", "estar",
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "have", "has", "had", "not", "but", "what", "when", "where", "which", "who",
}


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_transcript_with_timestamps(rec: dict[str, Any]) -> str:
    """Render a recording's transcript with [MM:SS] markers when segments exist."""
    segments = rec.get("segments") or []
    if segments:
        lines = []
        for seg in segments:
            ts = format_timestamp(float(seg.get("start", 0.0)))
            text = (seg.get("text") or "").strip()
            if text:
                lines.append(f"[{ts}] {text}")
        if lines:
            return "\n".join(lines)
    return (rec.get("transcript_text") or "").strip()


def format_transcript_for_llm(
    rec: dict[str, Any],
    max_block_chars: int = 380,
    max_block_sec: float = 25.0,
) -> str:
    """Group short Whisper segments into readable timestamped paragraphs to aid LLM recall."""
    segments = rec.get("segments") or []
    if not segments:
        return (rec.get("transcript_text") or "").strip()

    paragraphs: list[str] = []
    buf: list[str] = []
    buf_len = 0
    block_start = float(segments[0].get("start", 0.0))

    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        seg_start = float(seg.get("start", 0.0))
        if buf and (
            buf_len + len(text) + 1 > max_block_chars
            or (seg_start - block_start) >= max_block_sec
        ):
            paragraphs.append(f"[{format_timestamp(block_start)}] {' '.join(buf)}")
            buf = [text]
            buf_len = len(text)
            block_start = seg_start
        else:
            if not buf:
                block_start = seg_start
            buf.append(text)
            buf_len += len(text) + 1

    if buf:
        paragraphs.append(f"[{format_timestamp(block_start)}] {' '.join(buf)}")

    return "\n".join(paragraphs) if paragraphs else (rec.get("transcript_text") or "").strip()


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def _tokenize(text: str) -> list[str]:
    normalized = _strip_accents(text.lower())
    tokens = re.findall(r"[a-z0-9]+", normalized)
    return [t for t in tokens if len(t) > 1 and t not in _STOPWORDS_ES_EN]


def _chunk_recording(rec: dict[str, Any], chunk_chars: int = 1100) -> list[dict[str, Any]]:
    """Split a recording into timestamped text chunks for BM25 retrieval."""
    filename = rec.get("filename", "audio")
    segments = rec.get("segments") or []
    chunks: list[dict[str, Any]] = []

    if segments:
        buf: list[str] = []
        buf_len = 0
        start_ts = format_timestamp(float(segments[0].get("start", 0.0)))
        for seg in segments:
            ts = format_timestamp(float(seg.get("start", 0.0)))
            line = f"[{ts}] {(seg.get('text') or '').strip()}"
            if buf and buf_len + len(line) > chunk_chars:
                chunks.append(
                    {
                        "filename": filename,
                        "start_ts": start_ts,
                        "text": "\n".join(buf),
                    }
                )
                last_line = buf[-1]
                buf = [last_line, line]
                buf_len = len(last_line) + len(line)
                start_ts = ts
            else:
                if not buf:
                    start_ts = ts
                buf.append(line)
                buf_len += len(line) + 1
        if buf:
            chunks.append(
                {
                    "filename": filename,
                    "start_ts": start_ts,
                    "text": "\n".join(buf),
                }
            )
        return chunks

    raw_text = (rec.get("transcript_text") or "").strip()
    if not raw_text:
        return []
    for idx in range(0, len(raw_text), chunk_chars):
        chunks.append(
            {
                "filename": filename,
                "start_ts": "00:00",
                "text": raw_text[idx : idx + chunk_chars],
            }
        )
    return chunks


def _bm25_select_chunks(
    chunks: list[dict[str, Any]],
    query: str,
    max_chars: int,
    max_chunks: int | None = None,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[dict[str, Any]]:
    """Rank transcript chunks locally with BM25 and select top matches within budget."""
    if not chunks:
        return []

    tokenized_docs = [_tokenize(c["text"] + " " + c["filename"]) for c in chunks]
    doc_lens = [max(1, len(d)) for d in tokenized_docs]
    avgdl = sum(doc_lens) / len(doc_lens)

    df: Counter[str] = Counter()
    tfs: list[Counter[str]] = []
    for doc in tokenized_docs:
        tf = Counter(doc)
        tfs.append(tf)
        for term in tf:
            df[term] += 1

    query_terms = _tokenize(query)
    n_docs = len(chunks)
    scored: list[tuple[float, int]] = []
    for idx, tf in enumerate(tfs):
        score = 0.0
        dl = doc_lens[idx]
        for term in query_terms:
            if term not in tf:
                continue
            n_q = df[term]
            idf = math.log((n_docs - n_q + 0.5) / (n_q + 0.5) + 1.0)
            freq = tf[term]
            denom = freq + k1 * (1.0 - b + b * (dl / avgdl))
            score += idf * ((freq * (k1 + 1.0)) / denom)
        scored.append((score, idx))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected_indices: list[int] = []
    used_chars = 0
    for score, idx in scored:
        if score <= 0 and selected_indices:
            break
        if max_chunks is not None and len(selected_indices) >= max_chunks:
            break
        chunk_len = len(chunks[idx]["text"]) + 60
        if used_chars + chunk_len > max_chars and selected_indices:
            continue
        selected_indices.append(idx)
        used_chars += chunk_len
        if used_chars >= max_chars:
            break

    selected_indices.sort()
    return [chunks[i] for i in selected_indices]


def build_recording_summary_messages(
    filename: str,
    transcript_text: str,
    duration_sec: float,
    max_input_chars: int = 18000,
) -> list[dict[str, str]]:
    """Build prompt messages to summarize a single audio transcript."""
    clean_text = transcript_text.strip()
    truncated_note = ""
    if len(clean_text) > max_input_chars:
        half = max_input_chars // 2
        clean_text = (
            clean_text[:half]
            + "\n\n[... parte intermedia omitida por límite de ventana de contexto ...]\n\n"
            + clean_text[-half:]
        )
        truncated_note = "Nota: la transcripción es extensa y fue condensada a sus secciones principales.\n"

    duration_str = format_timestamp(duration_sec) if duration_sec > 0 else "desconocida"
    system_prompt = (
        "Eres un asistente experto en análisis de grabaciones de audio y clases. "
        "Responde siempre en español claro y estructurado. "
        "Basa tu resumen EXCLUSIVAMENTE en la transcripción proporcionada; "
        "no inventes datos, cifras ni nombres que no aparezcan en el texto."
    )
    user_prompt = (
        f"Archivo de audio: {filename} (duración: {duration_str})\n"
        f"{truncated_note}\n"
        "Genera un resumen útil y bien organizado en Markdown con estas secciones:\n"
        "### Resumen general\n"
        "Párrafo breve explicando de qué trata la grabación.\n\n"
        "### Puntos clave\n"
        "- Lista con las ideas, hechos, fechas o datos más relevantes.\n\n"
        "### Acuerdos, tareas o conclusiones (si aplica)\n"
        "- Lista de acciones o decisiones mencionadas (omite esta sección si no hay ninguna).\n\n"
        f"Transcripción:\n\"\"\"\n{clean_text}\n\"\"\""
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_combined_summary_messages(
    recordings: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build prompt messages to synthesize multiple recording summaries in one conversation."""
    parts = []
    for idx, rec in enumerate(recordings, start=1):
        duration_str = format_timestamp(float(rec.get("duration_sec") or 0.0))
        summary = (rec.get("summary_text") or "").strip()
        if not summary:
            summary = (rec.get("transcript_text") or "")[:2000].strip()
        parts.append(
            f"=== Grabación {idx}: {rec['filename']} (duración {duration_str}) ===\n{summary}"
        )

    joined = "\n\n".join(parts)
    system_prompt = (
        "Eres un asistente experto en sintetizar múltiples grabaciones de una misma conversación. "
        "Responde en español, integrando la información de todos los audios sin inventar datos."
    )
    user_prompt = (
        f"Esta conversación contiene {len(recordings)} grabaciones. "
        "Crea un resumen unificado en Markdown que conecte toda la información:\n"
        "### Síntesis global\n"
        "Panorama general considerando todas las grabaciones.\n\n"
        "### Qué aporta cada grabación y cambios/novedades\n"
        "- Menciona explícitamente los puntos clave o actualizaciones por archivo.\n\n"
        "### Conclusiones y pendientes consolidados\n"
        "- Lista final de acuerdos, decisiones o pendientes vigentes.\n\n"
        f"Resúmenes individuales:\n{joined}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_chat_messages(
    recordings: list[dict[str, Any]],
    combined_summary: str,
    chat_history: list[dict[str, Any]],
    user_question: str,
    num_ctx: int = 8192,
) -> tuple[list[dict[str, str]], str]:
    """
    Build LLM messages for contextual Q&A across all ready recordings in the session.

    Uses paragraph grouping and hybrid BM25 spotlighting when transcripts are medium/long
    so compact 4B models do not miss details buried in the middle of a class recording.
    """
    ready_recs = [r for r in recordings if r.get("status") == "ready" and r.get("transcript_text")]

    transcript_char_budget = max(4500, int(num_ctx * 2.2))
    history_char_budget = max(2000, int(num_ctx * 0.8))

    formatted_full_blocks: list[str] = []
    all_chunks: list[dict[str, Any]] = []
    total_full_chars = 0
    for idx, rec in enumerate(ready_recs, start=1):
        duration_str = format_timestamp(float(rec.get("duration_sec") or 0.0))
        grouped_text = format_transcript_for_llm(rec)
        block = (
            f"=== Grabación {idx}: {rec['filename']} (duración {duration_str}) ===\n"
            f"{grouped_text}"
        )
        formatted_full_blocks.append(block)
        total_full_chars += len(block)
        all_chunks.extend(_chunk_recording(rec))

    recent_user_turns = " ".join(
        m["content"] for m in chat_history[-4:] if m.get("role") == "user"
    )
    search_query = f"{user_question} {recent_user_turns}".strip()

    if total_full_chars <= transcript_char_budget:
        context_mode = "full"
        full_text = "\n\n".join(formatted_full_blocks)
        if total_full_chars > 3500 and all_chunks:
            spotlight = _bm25_select_chunks(
                all_chunks,
                search_query,
                max_chars=2600,
                max_chunks=5,
            )
            if spotlight:
                spotlight_text = "\n\n".join(
                    f"[{c['filename']} @ {c['start_ts']}]\n{c['text']}"
                    for c in spotlight
                )
                context_body = (
                    "FRAGMENTOS MÁS RELEVANTES DETECTADOS PARA ESTA PREGUNTA:\n"
                    f"{spotlight_text}\n\n"
                    "TRANSCRIPCIÓN COMPLETA:\n"
                    f"{full_text}"
                )
            else:
                context_body = full_text
        else:
            context_body = full_text
    else:
        context_mode = "retrieval"
        summary_blocks = []
        for idx, rec in enumerate(ready_recs, start=1):
            summary = (rec.get("summary_text") or "").strip()
            summary_blocks.append(f"- Grabación {idx} ({rec['filename']}): {summary}")

        summaries_text = "RESÚMENES DE LAS GRABACIONES:\n" + "\n".join(summary_blocks)
        if combined_summary.strip():
            summaries_text += f"\n\nRESUMEN INTEGRADO:\n{combined_summary.strip()}"

        # Hard cap summaries to 35% of budget so BM25 transcript excerpts always have room
        max_summary_chars = max(1600, int(transcript_char_budget * 0.35))
        if len(summaries_text) > max_summary_chars:
            summaries_text = (
                summaries_text[: max_summary_chars - 60]
                + "\n[... resúmenes condensados para priorizar fragmentos relevantes ...]"
            )

        remaining_budget = max(2500, transcript_char_budget - len(summaries_text))
        selected_chunks = _bm25_select_chunks(all_chunks, search_query, remaining_budget)

        chunk_blocks = [
            f"[{c['filename']} @ {c['start_ts']}]\n{c['text']}"
            for c in selected_chunks
        ]
        context_body = (
            f"{summaries_text}\n\n"
            "FRAGMENTOS RELEVANTES DE LAS TRANSCRIPCIONES:\n"
            + "\n\n".join(chunk_blocks)
        )

    system_prompt = (
        "Eres TransCrypting, un asistente local que responde preguntas sobre grabaciones de audio y clases.\n"
        "Reglas estrictas:\n"
        "1. Responde siempre en español claro y directo.\n"
        "2. Las transcripciones provienen de audio real (por ejemplo clases con ruido de fondo, "
        "voces superpuestas o mezcla de español e inglés). Revisa todo el texto buscando también "
        "equivalencias fonéticas o conceptuales antes de concluir que algo no aparece.\n"
        "3. Basa tus respuestas ÚNICAMENTE en el contexto de las grabaciones proporcionadas abajo.\n"
        "4. Cita el archivo y la marca de tiempo [MM:SS] cuando sea útil.\n"
        "5. Si tras revisar todo el contexto el dato realmente no aparece, dilo claramente sin inventar.\n\n"
        f"CONTEXTO DE LA CONVERSACIÓN ({len(ready_recs)} grabación(es)):\n"
        f"{context_body}"
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    selected_history: list[dict[str, str]] = []
    hist_chars = 0
    for msg in reversed(chat_history):
        role = msg.get("role")
        content = msg.get("content", "")
        if role not in ("user", "assistant") or not content:
            continue
        if hist_chars + len(content) > history_char_budget and selected_history:
            break
        selected_history.append({"role": role, "content": content})
        hist_chars += len(content)

    messages.extend(reversed(selected_history))
    messages.append({"role": "user", "content": user_question})
    return messages, context_mode


def estimate_session_context_usage(
    recordings: list[dict[str, Any]],
    combined_summary: str,
    messages: list[dict[str, Any]],
    num_ctx: int = 8192,
) -> dict[str, Any]:
    """Estimate current token usage vs num_ctx and report protection mode."""
    ready_recs = [r for r in recordings if r.get("status") == "ready" and r.get("transcript_text")]
    transcript_char_budget = max(4500, int(num_ctx * 2.2))

    transcript_chars = sum(len(format_transcript_for_llm(r)) + 80 for r in ready_recs)
    chat_chars = sum(len(m.get("content") or "") for m in messages[-12:])

    if transcript_chars <= transcript_char_budget:
        mode = "full"
        used_transcript_chars = transcript_chars
        if transcript_chars > 3500:
            used_transcript_chars += min(2600, transcript_chars // 2)
    else:
        mode = "retrieval"
        # Protected mode caps injected context inside transcript_char_budget
        used_transcript_chars = transcript_char_budget

    system_overhead_tokens = 260
    transcript_tokens = int(round(used_transcript_chars / 3.4))
    chat_tokens = int(round(min(chat_chars, int(num_ctx * 0.8)) / 3.4))
    estimated_tokens = min(
        num_ctx,
        system_overhead_tokens + transcript_tokens + chat_tokens,
    )
    usage_percent = round(min(100.0, (estimated_tokens / max(1, num_ctx)) * 100.0), 1)

    if mode == "retrieval":
        health = "protected"
    elif usage_percent >= 72.0:
        health = "high"
    else:
        health = "optimal"

    return {
        "estimated_tokens": estimated_tokens,
        "num_ctx": int(num_ctx),
        "usage_percent": usage_percent,
        "mode": mode,
        "health": health,
        "transcript_tokens": transcript_tokens,
        "chat_tokens": chat_tokens,
    }


def build_suggestions_messages(
    recordings: list[dict[str, Any]],
    combined_summary: str = "",
) -> list[dict[str, str]]:
    """Build prompt asking Gemma 4 for 3 tailored follow-up suggestions based on audio content."""
    ready_recs = [r for r in recordings if r.get("status") == "ready"]
    snippets: list[str] = []
    if combined_summary.strip():
        snippets.append(f"Resumen integrado:\n{combined_summary.strip()[:1800]}")
    for idx, rec in enumerate(ready_recs[:4], start=1):
        summary = (rec.get("summary_text") or "").strip()[:1200]
        sample = (rec.get("transcript_text") or "").strip()[:900]
        snippets.append(
            f"Grabación {idx} ({rec.get('filename', 'audio')}):\n"
            f"Resumen: {summary}\nMuestra: {sample}"
        )

    context = "\n\n".join(snippets)
    system_prompt = (
        "Eres un asistente que sugiere preguntas útiles según el tipo de grabación "
        "(clase escolar, examen, entrevista, reunión, conferencia, etc.). "
        "Responde ÚNICAMENTE con un arreglo JSON válido de 3 objetos, sin texto adicional."
    )
    user_prompt = (
        "Basándote en esta grabación, crea 3 sugerencias cortas y específicas para el usuario.\n"
        "- Si es una clase o repaso de examen: incluye opciones como evaluar al estudiante con preguntas tipo examen, explicar conceptos difíciles de la clase, o listar temas/fechas clave.\n"
        "- Si es una reunión o charla: incluye acuerdos, tareas o detalles específicos mencionados.\n"
        "- Cada objeto debe tener exactamente: \"label\" (máximo 4 palabras) y \"prompt\" (pregunta directa en español).\n"
        "- Formato exacto: [{\"label\": \"...\", \"prompt\": \"...\"}, ...]\n\n"
        f"{context}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_suggestions_json(
    raw_text: str,
    recording_count: int = 1,
) -> list[dict[str, str]]:
    """Extract up to 3 {label, prompt} suggestions from LLM output with clean fallback."""
    cleaned = (raw_text or "").strip()
    match = re.search(r"\[[\s\S]*\]", cleaned)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                valid: list[dict[str, str]] = []
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    label = str(item.get("label") or "").strip()[:42]
                    prompt = str(item.get("prompt") or "").strip()[:260]
                    if label and prompt:
                        valid.append({"label": label, "prompt": prompt})
                    if len(valid) == 3:
                        break
                if valid:
                    return valid
        except Exception:
            pass

    fallback = [
        {
            "label": "Puntos clave",
            "prompt": "¿Cuáles son los puntos más importantes y conceptos clave de la grabación?",
        },
        {
            "label": "Ponme a prueba",
            "prompt": "Hazme 5 preguntas tipo examen basadas únicamente en lo que se dijo en esta grabación para repasar.",
        },
        {
            "label": "Fechas y pendientes" if recording_count < 2 else "Comparar grabaciones",
            "prompt": (
                "¿Qué fechas, tareas o indicaciones importantes se mencionaron?"
                if recording_count < 2
                else "¿Qué cambios, temas nuevos o diferencias hay entre las grabaciones?"
            ),
        },
    ]
    return fallback


def _extract_timestamp(raw_ts: Any) -> str:
    m = re.search(r"\b(\d{1,2}:\d{2}(?::\d{2})?)\b", str(raw_ts or ""))
    return m.group(1) if m else ""


def build_quiz_generation_messages(
    recordings: list[dict[str, Any]],
    combined_summary: str = "",
    num_questions: int = 5,
    mode: str = "mixed",
    num_ctx: int = 8192,
) -> list[dict[str, str]]:
    """Build prompt asking Gemma 4 / Ollama for structured multiple-choice & open study questions."""
    ready_recs = [r for r in recordings if r.get("status") == "ready" and r.get("transcript_text")]
    budget = max(4500, int(num_ctx * 1.9))

    full_blocks = []
    total_chars = 0
    for idx, rec in enumerate(ready_recs, start=1):
        block = f"=== Grabación {idx}: {rec['filename']} ===\n{format_transcript_for_llm(rec)}"
        full_blocks.append(block)
        total_chars += len(block)

    if total_chars <= budget:
        context = "\n\n".join(full_blocks)
    else:
        parts = []
        if combined_summary.strip():
            parts.append(f"Resumen integrado:\n{combined_summary.strip()[:2200]}")
        per_rec = max(1200, (budget - 2200) // max(1, len(ready_recs)))
        for idx, rec in enumerate(ready_recs, start=1):
            text = format_transcript_for_llm(rec)
            if len(text) > per_rec:
                half = per_rec // 2
                text = text[:half] + "\n[...]\n" + text[-half:]
            parts.append(f"=== Grabación {idx}: {rec['filename']} ===\n{text}")
        context = "\n\n".join(parts)

    clean_count = max(2, min(10, int(num_questions)))
    if mode == "mcq":
        mode_rule = f"TODAS las {clean_count} preguntas deben ser de tipo \"mcq\" (opción múltiple con 4 opciones)."
    elif mode == "open":
        mode_rule = f"TODAS las {clean_count} preguntas deben ser de tipo \"open\" (respuesta abierta para redactar)."
    else:
        mode_rule = (
            f"Alterna preguntas de tipo \"mcq\" (opción múltiple con 4 opciones) y \"open\" (respuesta abierta) "
            f"hasta completar exactamente {clean_count} preguntas."
        )

    system_prompt = (
        "Eres un profesor creando un examen interactivo de repaso basado EXCLUSIVAMENTE en grabaciones de clase.\n"
        "Devuelve ÚNICAMENTE un arreglo JSON válido sin texto ni explicaciones fuera del JSON."
    )
    user_prompt = (
        f"Genera exactamente {clean_count} preguntas de estudio en español basadas únicamente en la clase.\n"
        f"{mode_rule}\n"
        "Reglas de formato JSON estricto:\n"
        "- Para \"mcq\": {\"id\": 1, \"type\": \"mcq\", \"question\": \"...\", \"options\": [\"op1\", \"op2\", \"op3\", \"op4\"], \"correct_index\": 0, \"explanation\": \"breve explicación\", \"timestamp\": \"MM:SS\"}\n"
        "- Para \"open\": {\"id\": 2, \"type\": \"open\", \"question\": \"...\", \"expected_answer\": \"respuesta modelo breve según la clase\", \"timestamp\": \"MM:SS\"}\n"
        "- No pongas la letra A/B/C/D dentro de los strings de options.\n"
        "- Incluye el \"timestamp\" [MM:SS] aproximado donde se menciona en el texto.\n"
        "- No inventes datos que no estén en la grabación.\n\n"
        f"CONTENIDO DE LA CLASE:\n{context}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_quiz_json(raw_text: str, mode: str = "mixed") -> list[dict[str, Any]]:
    """Parse and sanitize generated quiz questions JSON."""
    cleaned = (raw_text or "").strip()
    match = re.search(r"\[[\s\S]*\]", cleaned)
    questions: list[dict[str, Any]] = []

    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    q_text = str(item.get("question") or "").strip()
                    if not q_text:
                        continue
                    raw_type = str(item.get("type") or "").strip().lower()
                    if mode in ("mcq", "open"):
                        q_type = mode
                    else:
                        q_type = "mcq" if raw_type == "mcq" or isinstance(item.get("options"), list) else "open"

                    ts = _extract_timestamp(item.get("timestamp"))
                    if q_type == "mcq":
                        raw_opts = item.get("options") or []
                        opts = [str(o).strip() for o in raw_opts if str(o).strip()][:4]
                        if len(opts) < 2:
                            if mode == "mcq":
                                continue
                            q_type = "open"
                        else:
                            try:
                                c_idx = int(item.get("correct_index", 0))
                            except (TypeError, ValueError):
                                c_idx = 0
                            if c_idx < 0 or c_idx >= len(opts):
                                c_idx = 0
                            questions.append(
                                {
                                    "id": len(questions) + 1,
                                    "type": "mcq",
                                    "question": q_text,
                                    "options": opts,
                                    "correct_index": c_idx,
                                    "explanation": str(item.get("explanation") or "").strip(),
                                    "timestamp": ts,
                                }
                            )
                            continue

                    questions.append(
                        {
                            "id": len(questions) + 1,
                            "type": "open",
                            "question": q_text,
                            "expected_answer": str(
                                item.get("expected_answer") or item.get("explanation") or ""
                            ).strip(),
                            "timestamp": ts,
                        }
                    )
        except Exception:
            questions = []

    return questions


def extract_relevant_excerpt(
    recordings: list[dict[str, Any]],
    query: str,
    max_chars: int = 3800,
) -> str:
    """Return top BM25 transcript excerpts for grading an open-ended quiz answer."""
    ready_recs = [r for r in recordings if r.get("status") == "ready" and r.get("transcript_text")]
    all_chunks: list[dict[str, Any]] = []
    for rec in ready_recs:
        all_chunks.extend(_chunk_recording(rec))
    selected = _bm25_select_chunks(all_chunks, query, max_chars=max_chars, max_chunks=4)
    if selected:
        return "\n\n".join(f"[{c['filename']} @ {c['start_ts']}]\n{c['text']}" for c in selected)
    return "\n\n".join(format_transcript_for_llm(r)[:1500] for r in ready_recs[:2])


def build_open_answer_grading_messages(
    question: str,
    expected_answer: str,
    timestamp: str,
    user_answer: str,
    relevant_excerpt: str,
) -> list[dict[str, str]]:
    """Build prompt to evaluate a student's open-ended answer against class recordings."""
    system_prompt = (
        "Eres un profesor amable pero preciso evaluando la respuesta de un estudiante según su grabación de clase.\n"
        "Devuelve ÚNICAMENTE un objeto JSON válido sin texto extra."
    )
    user_prompt = (
        f"Pregunta: {question}\n"
        f"Respuesta esperada de referencia: {expected_answer or 'Basada en la clase'}\n"
        f"Marca de tiempo de referencia: {timestamp or 'N/A'}\n"
        f"Respuesta del estudiante: {user_answer}\n\n"
        f"Fragmentos relevantes de la clase:\n{relevant_excerpt}\n\n"
        "Devuelve exactamente este JSON en español:\n"
        "{\"verdict\": \"correct\" | \"partial\" | \"incorrect\", \"score\": 0-100, "
        "\"feedback\": \"qué estuvo bien o por qué\", \"missing_points\": \"qué faltó mencionar (vacío si está completa)\", "
        "\"timestamp\": \"MM:SS\"}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_grade_json(raw_text: str, fallback_timestamp: str = "") -> dict[str, Any]:
    """Parse open-answer grading JSON safely with clean fallback."""
    cleaned = (raw_text or "").strip()
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                verdict = str(data.get("verdict") or "partial").lower()
                if verdict not in ("correct", "partial", "incorrect"):
                    verdict = "partial"
                try:
                    score = max(0, min(100, int(round(float(data.get("score", 60))))))
                except (TypeError, ValueError):
                    score = 100 if verdict == "correct" else 60 if verdict == "partial" else 20
                ts = _extract_timestamp(data.get("timestamp")) or _extract_timestamp(fallback_timestamp)
                return {
                    "verdict": verdict,
                    "score": score,
                    "feedback": str(data.get("feedback") or "").strip(),
                    "missing_points": str(data.get("missing_points") or "").strip(),
                    "timestamp": ts,
                }
        except Exception:
            pass

    return {
        "verdict": "partial",
        "score": 60,
        "feedback": cleaned or "Evaluación completada.",
        "missing_points": "",
        "timestamp": _extract_timestamp(fallback_timestamp),
    }
