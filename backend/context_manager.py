"""Prompt construction and local BM25 context management for multi-recording sessions."""

from __future__ import annotations

import json
import math
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from backend.source_service import is_meaningful_page_text

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
            spk = (seg.get("speaker") or "").strip()
            if text:
                prefix = f"{spk}: " if spk else ""
                lines.append(f"[{ts}] {prefix}{text}")
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
    block_speaker = (segments[0].get("speaker") or "").strip()

    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        seg_start = float(seg.get("start", 0.0))
        seg_speaker = (seg.get("speaker") or "").strip()
        speaker_changed = bool(buf) and seg_speaker != block_speaker
        if buf and (
            speaker_changed
            or buf_len + len(text) + 1 > max_block_chars
            or (seg_start - block_start) >= max_block_sec
        ):
            spk_prefix = f"{block_speaker}: " if block_speaker else ""
            paragraphs.append(f"[{format_timestamp(block_start)}] {spk_prefix}{' '.join(buf)}")
            buf = [text]
            buf_len = len(text)
            block_start = seg_start
            block_speaker = seg_speaker
        else:
            if not buf:
                block_start = seg_start
                block_speaker = seg_speaker
            buf.append(text)
            buf_len += len(text) + 1

    if buf:
        spk_prefix = f"{block_speaker}: " if block_speaker else ""
        paragraphs.append(f"[{format_timestamp(block_start)}] {spk_prefix}{' '.join(buf)}")

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
            spk = (seg.get("speaker") or "").strip()
            spk_prefix = f"{spk}: " if spk else ""
            line = f"[{ts}] {spk_prefix}{(seg.get('text') or '').strip()}"
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


def _chunk_source(src: dict[str, Any], chunk_chars: int = 1100) -> list[dict[str, Any]]:
    """Split an indexed document/image source into located chunks for BM25 fallback."""
    filename = src.get("filename", "fuente")
    pages = src.get("pages") or []
    chunks: list[dict[str, Any]] = []
    if pages:
        for page in pages:
            loc = (page.get("locator") or "").strip() or "p. 1"
            txt = (page.get("text") or "").strip()
            if not txt:
                continue
            for idx in range(0, len(txt), chunk_chars):
                piece = txt[idx : idx + chunk_chars].strip()
                if piece:
                    chunks.append({"filename": filename, "start_ts": loc, "text": piece})
        if chunks:
            return chunks

    raw = (src.get("extracted_text") or "").strip()
    if not raw:
        return []
    loc = filename if src.get("source_type") == "image" else "sec. 1"
    for idx in range(0, len(raw), chunk_chars):
        piece = raw[idx : idx + chunk_chars].strip()
        if piece:
            chunks.append({"filename": filename, "start_ts": loc, "text": piece})
    return chunks


def _format_source_for_llm(src: dict[str, Any]) -> str:
    pages = src.get("pages") or []
    if pages:
        blocks = []
        for p in pages:
            loc = (p.get("locator") or "").strip()
            txt = (p.get("text") or "").strip()
            if is_meaningful_page_text(txt):
                blocks.append(f"[{loc}]\n{txt}" if loc else txt)
        if blocks:
            return "\n\n".join(blocks)
    raw = (src.get("extracted_text") or "").strip()
    return raw if is_meaningful_page_text(raw) else ""


def _humanize_filename(filename: str) -> str:
    stem = Path(filename or "fuente").stem
    stem = re.sub(r"([a-z])([A-Z0-9])", r"\1 \2", stem)
    stem = re.sub(r"([A-Za-z])([0-9])", r"\1 \2", stem)
    stem = re.sub(r"[_\-\./]+", " ", stem)
    return re.sub(r"\s+", " ", stem).strip()


def _build_source_catalog(
    ready_recs: list[dict[str, Any]],
    ready_srcs: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    for idx, rec in enumerate(ready_recs, start=1):
        fname = rec.get("filename", f"grabacion_{idx}.m4a")
        alias = _humanize_filename(fname)
        dur = format_timestamp(float(rec.get("duration_sec") or 0.0))
        lines.append(f'- Audio {idx}: "{fname}" (alias: "{alias}") · duración {dur}')
    for idx, src in enumerate(ready_srcs, start=1):
        fname = src.get("filename", f"fuente_{idx}")
        alias = _humanize_filename(fname)
        kind = str(src.get("source_type") or "doc").upper()
        cnt = int(src.get("page_count") or 0)
        unit = "imagen" if kind == "IMAGE" else f"{cnt or 1} pág./secc."
        lines.append(f'- Fuente {idx} ({kind}): "{fname}" (alias: "{alias}") · {unit}')
    return "\n".join(lines)


def _format_indexed_chunk_header(chunk: dict[str, Any]) -> str:
    filename = chunk.get("filename") or "fuente"
    locator = (chunk.get("locator") or "").strip()
    if chunk.get("item_kind") == "recording":
        return f"[{filename} @ {locator or '00:00'}]"
    if locator:
        return f"[{filename} · {locator}]"
    return f"[{filename}]"


def _select_chunks_within_budget(
    chunks: list[dict[str, Any]],
    max_chars: int,
    max_chunks: int | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used = 0
    for c in chunks:
        text = c.get("content") or c.get("text") or ""
        if not text:
            continue
        est = len(text) + 70
        if max_chunks is not None and len(selected) >= max_chunks:
            break
        if used + est > max_chars and selected:
            break
        selected.append(c)
        used += est
    return selected


def build_source_summary_messages(
    filename: str,
    source_type: str,
    extracted_text: str,
    page_count: int = 0,
    max_input_chars: int = 18000,
) -> list[dict[str, str]]:
    """Build prompt messages to summarize a single document or indexed image source."""
    clean_text = (extracted_text or "").strip()
    truncated_note = ""
    if len(clean_text) > max_input_chars:
        half = max_input_chars // 2
        clean_text = (
            clean_text[:half]
            + "\n\n[... contenido intermedio resumido por límite de ventana de contexto ...]\n\n"
            + clean_text[-half:]
        )
        truncated_note = "Nota: el documento es extenso y fue condensado a sus secciones principales.\n"

    type_labels = {
        "pdf": "Documento PDF",
        "pptx": "Presentación / Diapositivas",
        "docx": "Documento Word",
        "text": "Texto / Apuntes",
        "image": "Imagen analizada",
    }
    kind_label = type_labels.get((source_type or "").lower(), "Fuente documental")
    meta_detail = f"{page_count} página(s)/sección(es)" if page_count else "contenido único"

    system_prompt = (
        "Eres un asistente experto en analizar documentos de estudio, presentaciones, apuntes e imágenes. "
        "Responde siempre en español claro y estructurado. "
        "Basa tu resumen EXCLUSIVAMENTE en el texto extraído proporcionado; cita ubicaciones como [p. X] o [diap. X] cuando aparezcan."
    )
    user_prompt = (
        f"Fuente: {filename} ({kind_label}, {meta_detail})\n"
        f"{truncated_note}\n"
        "Genera un resumen útil y bien organizado en Markdown:\n"
        "### Resumen general\n"
        "Párrafo breve explicando el tema y propósito de esta fuente.\n\n"
        "### Puntos clave y conceptos\n"
        "- Lista con definiciones, fórmulas, fechas o ideas centrales citando su página o diapositiva cuando aplique.\n\n"
        "### Datos o indicaciones destacadas (si aplica)\n"
        "- Lista breve de conclusiones, tareas o detalles importantes (omite si no aplica).\n\n"
        f"Contenido extraído:\n\"\"\"\n{clean_text}\n\"\"\""
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


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
    max_words = max(100, min(400, len(clean_text) // 40))
    system_prompt = (
        "Eres un asistente experto en análisis de grabaciones de audio y clases. "
        "Responde siempre en español claro y estructurado. "
        "Basa tu resumen EXCLUSIVAMENTE en la transcripción proporcionada; "
        "no inventes datos, cifras ni nombres que no aparezcan en el texto."
    )
    user_prompt = (
        f"Archivo de audio: {filename} (duración: {duration_str})\n"
        f"{truncated_note}\n"
        f"Resume en un máximo de {max_words} palabras. Evita repetir información entre secciones.\n"
        "Usa estas secciones en Markdown:\n"
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
    sources: list[dict[str, Any]] | None = None,
    *,
    max_input_chars: int = 12000,
) -> list[dict[str, str]]:
    """Build prompt messages to synthesize multiple enabled recordings and sources in one conversation."""
    active_recs = [r for r in recordings if r.get("enabled", True) and (r.get("summary_text") or r.get("transcript_text"))]
    active_srcs = [s for s in (sources or []) if s.get("enabled", True) and (s.get("summary_text") or s.get("extracted_text"))]
    total_items = len(active_recs) + len(active_srcs)
    chars_per_item = min(2500, max(1, max_input_chars // max(1, total_items)))

    def excerpt(summary: str) -> str:
        if len(summary) <= chars_per_item:
            return summary
        marker = "\n[…]\n"
        if chars_per_item < len(marker) + 2:
            return summary[:chars_per_item]
        remaining = chars_per_item - len(marker)
        head = remaining // 2
        return summary[:head] + marker + summary[-(remaining - head):]

    parts = []
    for idx, rec in enumerate(active_recs, start=1):
        duration_str = format_timestamp(float(rec.get("duration_sec") or 0.0))
        summary = (rec.get("summary_text") or "").strip()
        if not summary:
            summary = (rec.get("transcript_text") or "")[:2000].strip()
        parts.append(
            f"=== Grabación {idx}: {rec['filename']} (duración {duration_str}) ===\n{excerpt(summary)}"
        )

    for idx, src in enumerate(active_srcs, start=1):
        summary = (src.get("summary_text") or "").strip()
        if not summary:
            summary = (src.get("extracted_text") or "")[:2000].strip()
        kind = str(src.get("source_type") or "documento").upper()
        parts.append(
            f"=== Fuente {idx} ({kind}): {src['filename']} ===\n{excerpt(summary)}"
        )

    joined = "\n\n".join(parts)
    system_prompt = (
        "Eres un asistente experto en sintetizar múltiples fuentes de estudio (grabaciones, PDFs, diapositivas e imágenes) de una misma conversación. "
        "Responde en español, integrando la información de todas las fuentes activas sin inventar datos."
    )
    user_prompt = (
        f"Esta conversación contiene {total_items} fuente(s) activa(s) "
        f"({len(active_recs)} grabación(es) y {len(active_srcs)} documento(s)/imagen(es)). "
        "Escribe un resumen unificado en Markdown de hasta 400 palabras. "
        "Evita repetir información entre secciones.\n"
        "### Síntesis global\n"
        "Panorama general considerando todas las fuentes activas.\n\n"
        "### Qué aporta cada fuente y cómo se complementan\n"
        "- Usa una viñeta breve por archivo para explicar su aporte.\n\n"
        "### Conclusiones y puntos clave consolidados\n"
        "- Lista final de conceptos esenciales, acuerdos o pendientes vigentes.\n\n"
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
    sources: list[dict[str, Any]] | None = None,
    indexed_chunks: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, str]], str]:
    """
    Build LLM messages for contextual Q&A across all enabled recordings & sources.

    Uses full text when it safely fits num_ctx, and switches automatically to
    indexed SQLite FTS5 / BM25 retrieval when total sources exceed budget.
    """
    ready_recs = [
        r for r in recordings
        if r.get("status") == "ready" and r.get("enabled", True) and r.get("transcript_text")
    ]
    ready_srcs = [
        s for s in (sources or [])
        if s.get("status") == "ready" and s.get("enabled", True) and s.get("extracted_text")
    ]

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

    for idx, src in enumerate(ready_srcs, start=1):
        src_kind = str(src.get("source_type") or "documento").upper()
        src_text = _format_source_for_llm(src)
        block = (
            f"=== Fuente {idx} ({src_kind}): {src['filename']} ===\n"
            f"{src_text}"
        )
        formatted_full_blocks.append(block)
        total_full_chars += len(block)
        all_chunks.extend(_chunk_source(src))

    recent_user_turns = " ".join(
        m["content"] for m in chat_history[-4:] if m.get("role") == "user"
    )
    search_query = f"{user_question} {recent_user_turns}".strip()

    if total_full_chars <= transcript_char_budget:
        context_mode = "full"
        full_text = "\n\n".join(formatted_full_blocks)
        if total_full_chars > 3500:
            if indexed_chunks:
                picked = _select_chunks_within_budget(indexed_chunks, max_chars=2600, max_chunks=5)
                spotlight_text = "\n\n".join(
                    f"{_format_indexed_chunk_header(c)}\n{c['content']}" for c in picked
                )
            elif all_chunks:
                spotlight = _bm25_select_chunks(
                    all_chunks,
                    search_query,
                    max_chars=2600,
                    max_chunks=5,
                )
                spotlight_text = "\n\n".join(
                    f"[{c['filename']} @ {c['start_ts']}]\n{c['text']}"
                    for c in spotlight
                )
            else:
                spotlight_text = ""

            if spotlight_text:
                context_body = (
                    "FRAGMENTOS MÁS RELEVANTES DETECTADOS PARA ESTA PREGUNTA:\n"
                    f"{spotlight_text}\n\n"
                    "CONTENIDO COMPLETO DE LAS FUENTES ACTIVAS:\n"
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
        for idx, src in enumerate(ready_srcs, start=1):
            summary = (src.get("summary_text") or "").strip()
            summary_blocks.append(f"- Fuente {idx} ({src['filename']}): {summary}")

        summaries_text = "RESÚMENES DE LAS FUENTES ACTIVAS:\n" + "\n".join(summary_blocks)
        if combined_summary.strip():
            summaries_text += f"\n\nRESUMEN INTEGRADO:\n{combined_summary.strip()}"

        max_summary_chars = max(1600, int(transcript_char_budget * 0.35))
        if len(summaries_text) > max_summary_chars:
            summaries_text = (
                summaries_text[: max_summary_chars - 60]
                + "\n[... resúmenes condensados para priorizar fragmentos relevantes ...]"
            )

        remaining_budget = max(2500, transcript_char_budget - len(summaries_text))
        if indexed_chunks:
            selected = _select_chunks_within_budget(indexed_chunks, max_chars=remaining_budget)
            chunk_blocks = [
                f"{_format_indexed_chunk_header(c)}\n{c['content']}" for c in selected
            ]
        else:
            selected_chunks = _bm25_select_chunks(all_chunks, search_query, remaining_budget)
            chunk_blocks = [
                f"[{c['filename']} @ {c['start_ts']}]\n{c['text']}"
                for c in selected_chunks
            ]

        context_body = (
            f"{summaries_text}\n\n"
            "FRAGMENTOS MÁS RELEVANTES DEL ÍNDICE LOCAL:\n"
            + "\n\n".join(chunk_blocks)
        )

    audio_cite_fmt = "[nombre_audio · MM:SS]" if len(ready_recs) > 1 else "[MM:SS]"
    total_sources = len(ready_recs) + len(ready_srcs)
    catalog_text = _build_source_catalog(ready_recs, ready_srcs)
    system_prompt = (
        "Eres TransCrypting, un asistente local estilo cuaderno de investigación que responde preguntas sobre grabaciones, PDFs, diapositivas, apuntes e imágenes.\n"
        "Reglas estrictas:\n"
        "1. Responde siempre en español claro y directo.\n"
        "2. Si hay transcripciones de audio real (clases con ruido o mezcla español/inglés), busca equivalencias fonéticas o conceptuales antes de descartar un dato.\n"
        "3. Basa tus respuestas ÚNICAMENTE en las fuentes activas proporcionadas abajo.\n"
        "4. Identificación flexible de nombres de fuente: si el usuario menciona una fuente de forma abreviada, sin guiones bajos o traducida (por ejemplo \"chapter 11\" o \"capítulo 11\" para referirse a \"Chapter_11_Persuasive_Messages_Summary.pdf\"), relaciónala automáticamente con su archivo correspondiente del catálogo antes de responder.\n"
        "5. Cita siempre de forma precisa usando estos formatos exactos entre corchetes:\n"
        f"   - Para audios: {audio_cite_fmt}\n"
        "   - Para documentos/diapositivas: [nombre_archivo.pdf · p. 4] o [presentacion.pptx · diap. 2]\n"
        "   - Para imágenes: [nombre_imagen.png]\n"
        "6. Si tras revisar todo el contexto el dato realmente no aparece, dilo claramente sin inventar.\n\n"
        f"CATÁLOGO COMPLETO DE FUENTES ACTIVAS ({total_sources} total: {len(ready_recs)} audio(s), {len(ready_srcs)} documento(s)/imagen(es)):\n"
        f"{catalog_text}\n\n"
        f"CONTENIDO DE LAS FUENTES ACTIVAS:\n"
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
    sources: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Estimate current token usage vs num_ctx across active recordings and sources."""
    ready_recs = [
        r for r in recordings
        if r.get("status") == "ready" and r.get("enabled", True) and r.get("transcript_text")
    ]
    ready_srcs = [
        s for s in (sources or [])
        if s.get("status") == "ready" and s.get("enabled", True) and s.get("extracted_text")
    ]
    transcript_char_budget = max(4500, int(num_ctx * 2.2))

    transcript_chars = sum(len(format_transcript_for_llm(r)) + 80 for r in ready_recs)
    source_chars = sum(len(_format_source_for_llm(s)) + 80 for s in ready_srcs)
    total_source_chars = transcript_chars + source_chars
    chat_chars = sum(len(m.get("content") or "") for m in messages[-12:])

    if total_source_chars <= transcript_char_budget:
        mode = "full"
        used_transcript_chars = total_source_chars
        if total_source_chars > 3500:
            used_transcript_chars += min(2600, total_source_chars // 2)
    else:
        mode = "retrieval"
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
        "active_recording_count": len(ready_recs),
        "active_source_count": len(ready_srcs),
    }


def build_suggestions_messages(
    recordings: list[dict[str, Any]],
    combined_summary: str = "",
    sources: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Build prompt asking the local LLM for 3 tailored follow-up suggestions across active sources."""
    ready_recs = [
        r for r in recordings if r.get("status") == "ready" and r.get("enabled", True)
    ]
    ready_srcs = [
        s for s in (sources or []) if s.get("status") == "ready" and s.get("enabled", True)
    ]
    snippets: list[str] = []
    if combined_summary.strip():
        snippets.append(f"Resumen integrado:\n{combined_summary.strip()[:1800]}")
    for idx, rec in enumerate(ready_recs[:3], start=1):
        summary = (rec.get("summary_text") or "").strip()[:1000]
        sample = (rec.get("transcript_text") or "").strip()[:700]
        snippets.append(
            f"Grabación {idx} ({rec.get('filename', 'audio')}):\n"
            f"Resumen: {summary}\nMuestra: {sample}"
        )
    for idx, src in enumerate(ready_srcs[:3], start=1):
        summary = (src.get("summary_text") or "").strip()[:1000]
        sample = (src.get("extracted_text") or "").strip()[:700]
        snippets.append(
            f"Fuente {idx} ({src.get('filename', 'documento')} · {src.get('source_type', 'doc')}):\n"
            f"Resumen: {summary}\nMuestra: {sample}"
        )

    context = "\n\n".join(snippets)
    system_prompt = (
        "Eres un asistente que sugiere preguntas útiles según las fuentes de estudio cargadas "
        "(clases en audio, PDFs, diapositivas, apuntes o imágenes). "
        "Responde ÚNICAMENTE con un arreglo JSON válido de 3 objetos, sin texto adicional."
    )
    user_prompt = (
        "Basándote en estas fuentes activas, crea 3 sugerencias cortas y específicas para el usuario.\n"
        "- Si son apuntes, clases o PDFs de estudio: incluye opciones como evaluar al estudiante con preguntas tipo examen, explicar conceptos difíciles o cruzar ideas entre fuentes.\n"
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
    raw = str(raw_ts or "").strip()
    if not raw or raw.upper() == "N/A":
        return ""
    m = re.search(r"\b(\d{1,2}:\d{2}(?::\d{2})?)\b", raw)
    if m:
        return m.group(1)
    return raw.strip("[]")[:80]


def build_quiz_generation_messages(
    recordings: list[dict[str, Any]],
    combined_summary: str = "",
    num_questions: int = 5,
    mode: str = "mixed",
    num_ctx: int = 8192,
    sources: list[dict[str, Any]] | None = None,
    indexed_chunks: list[dict[str, Any]] | None = None,
    difficulty: str = "medium",
    question_style: str = "conceptual",
    focus_prompt: str = "",
    include_hints: bool = True,
) -> list[dict[str, str]]:
    """Build prompt asking the local LLM for structured customizable study questions."""
    ready_recs = [
        r for r in recordings
        if r.get("status") == "ready" and r.get("transcript_text")
    ]
    ready_srcs = [
        s for s in (sources or [])
        if s.get("status") == "ready" and s.get("extracted_text")
    ]
    budget = max(4500, int(num_ctx * 1.9))

    full_blocks = []
    total_chars = 0
    for idx, rec in enumerate(ready_recs, start=1):
        block = f"=== Grabación {idx}: {rec['filename']} ===\n{format_transcript_for_llm(rec)}"
        full_blocks.append(block)
        total_chars += len(block)

    for idx, src in enumerate(ready_srcs, start=1):
        kind = str(src.get("source_type") or "documento").upper()
        block = f"=== Fuente {idx} ({kind}): {src['filename']} ===\n{_format_source_for_llm(src)}"
        full_blocks.append(block)
        total_chars += len(block)

    if total_chars <= budget:
        context = "\n\n".join(full_blocks)
    else:
        parts = []
        if combined_summary.strip():
            parts.append(f"Resumen integrado:\n{combined_summary.strip()[:2200]}")
        if indexed_chunks:
            picked = _select_chunks_within_budget(indexed_chunks, max_chars=max(2000, budget - 2200))
            parts.extend(f"{_format_indexed_chunk_header(c)}\n{c['content']}" for c in picked)
        else:
            count = max(1, len(ready_recs) + len(ready_srcs))
            per_item = max(1000, (budget - 2200) // count)
            for idx, rec in enumerate(ready_recs, start=1):
                text = format_transcript_for_llm(rec)
                if len(text) > per_item:
                    half = per_item // 2
                    text = text[:half] + "\n[...]\n" + text[-half:]
                parts.append(f"=== Grabación {idx}: {rec['filename']} ===\n{text}")
            for idx, src in enumerate(ready_srcs, start=1):
                text = _format_source_for_llm(src)
                if len(text) > per_item:
                    half = per_item // 2
                    text = text[:half] + "\n[...]\n" + text[-half:]
                parts.append(f"=== Fuente {idx}: {src['filename']} ===\n{text}")
        context = "\n\n".join(parts)

    clean_count = max(2, min(15, int(num_questions)))
    if mode == "mcq":
        mode_rule = (
            f"TODAS las {clean_count} preguntas deben ser de tipo \"mcq\" (opción múltiple con 4 opciones).\n"
            "- Distribuye el índice correcto (`correct_index`) entre 0, 1, 2 y 3 (NUNCA pongas todas las respuestas correctas en el mismo índice)."
        )
    elif mode == "true_false":
        half_low = clean_count // 2
        true_target = half_low if (clean_count % 2 == 0 or random.choice([True, False])) else half_low + 1
        false_target = clean_count - true_target
        tf_plan = [0] * true_target + [1] * false_target
        random.shuffle(tf_plan)
        plan_desc = ", ".join(
            f"P{idx}: {'Verdadero (correct_index: 0)' if val == 0 else 'Falso (correct_index: 1)'}"
            for idx, val in enumerate(tf_plan, start=1)
        )
        mode_rule = (
            f"TODAS las {clean_count} preguntas deben ser de tipo \"true_false\" (afirmación para responder Verdadero o Falso).\n"
            f"- VARIACIÓN OBLIGATORIA ({true_target} Verdaderas y {false_target} Falsas): NUNCA hagas que todas sean Falsas ni todas Verdaderas.\n"
            f"- Sigue exactamente esta distribución mezclada: {plan_desc}.\n"
            "- Las afirmaciones marcadas como Verdadero DEBEN ser 100% fieles a las fuentes (sin alterar hechos); las marcadas como Falso DEBEN introducir una contradicción clara pero sutil frente a las fuentes."
        )
    elif mode == "open":
        mode_rule = f"TODAS las {clean_count} preguntas deben ser de tipo \"open\" (respuesta abierta para redactar)."
    else:
        mode_rule = (
            f"Combina de forma variada preguntas \"mcq\" (4 opciones), \"true_false\" (Verdadero/Falso) y \"open\" (respuesta abierta) "
            f"hasta completar exactamente {clean_count} preguntas.\n"
            "- En las preguntas \"true_false\" alterna entre Verdadero (correct_index: 0) y Falso (correct_index: 1) (nunca todas iguales), y en las \"mcq\" varía `correct_index` entre 0, 1, 2 y 3."
        )

    diff_map = {
        "easy": "Nivel BÁSICO: evalúa definiciones esenciales y conceptos centrales de forma clara.",
        "medium": "Nivel INTERMEDIO: evalúa comprensión, relaciones entre ideas y ejemplos concretos de las fuentes.",
        "hard": "Nivel DIFÍCIL / EXAMEN PARCIAL: redacta preguntas exigentes con distractores plausibles, análisis profundo, comparaciones y aplicación.",
    }
    style_map = {
        "conceptual": "Estilo CONCEPTUAL: prioriza fundamentos teóricos, definiciones, causas y comparaciones.",
        "practical": "Estilo PRÁCTICO: prioriza casos aplicados, ejercicios, escenarios reales y resolución de problemas.",
        "tricky": "Estilo TRAMPAS DE EXAMEN: prioriza matices sutiles, excepciones, condiciones límite y errores frecuentes.",
    }
    diff_rule = diff_map.get(difficulty, diff_map["medium"])
    style_rule = style_map.get(question_style, style_map["conceptual"])
    focus_rule = (
        f"ENFOQUE ESPECÍFICO DEL USUARIO (PRIORITARIO): {focus_prompt.strip()}\n"
        if focus_prompt and focus_prompt.strip()
        else ""
    )
    hint_rule = (
        '- Incluye en cada objeto el campo `"hint": "pista breve sin revelar la respuesta"`.\n'
        if include_hints
        else '- Deja `"hint": ""` vacío en cada objeto.\n'
    )

    system_prompt = (
        "Eres un profesor creando un examen interactivo de repaso basado EXCLUSIVAMENTE en las fuentes cargadas (audios, PDFs, diapositivas, apuntes o imágenes).\n"
        "Devuelve ÚNICAMENTE un arreglo JSON válido sin texto ni explicaciones fuera del JSON."
    )
    user_prompt = (
        f"Genera exactamente {clean_count} preguntas de estudio en español basadas únicamente en las fuentes seleccionadas.\n"
        f"{mode_rule}\n"
        f"{diff_rule}\n"
        f"{style_rule}\n"
        f"{focus_rule}"
        "Reglas de formato JSON estricto:\n"
        "- Para \"mcq\": {\"id\": 1, \"type\": \"mcq\", \"question\": \"...\", \"options\": [\"op1\", \"op2\", \"op3\", \"op4\"], \"correct_index\": 2, \"explanation\": \"breve explicación\", \"hint\": \"pista corta\", \"timestamp\": \"MM:SS o fuente.pdf · p. X\"}\n"
        "- Para \"true_false\": {\"id\": 2, \"type\": \"true_false\", \"question\": \"Afirmación clara...\", \"options\": [\"Verdadero\", \"Falso\"], \"correct_index\": 0 o 1, \"explanation\": \"breve explicación\", \"hint\": \"pista corta\", \"timestamp\": \"MM:SS o fuente.pdf · p. X\"} (usa 0 cuando la afirmación sea Verdadera y 1 cuando sea Falsa)\n"
        "- Para \"open\": {\"id\": 3, \"type\": \"open\", \"question\": \"...\", \"expected_answer\": \"respuesta modelo breve\", \"hint\": \"pista corta\", \"timestamp\": \"MM:SS o fuente.pdf · p. X\"}\n"
        "- No pongas la letra A/B/C/D dentro de los strings de options.\n"
        f"{hint_rule}"
        "- Incluye en \"timestamp\" la marca [MM:SS] si proviene de audio, o \"archivo.pdf · p. X\" / \"imagen.png\" si proviene de documento/imagen.\n"
        "- No inventes datos que no estén en las fuentes.\n\n"
        f"CONTENIDO DE ESTUDIO:\n{context}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_quiz_json(raw_text: str, mode: str = "mixed") -> list[dict[str, Any]]:
    """Parse and sanitize generated quiz questions JSON (supports mcq, true_false, open, hints)."""
    cleaned = (raw_text or "").strip()
    questions: list[dict[str, Any]] = []

    parsed_list: list[Any] | None = None
    candidates = [cleaned]
    arr_m = re.search(r"\[[\s\S]*\]", cleaned)
    if arr_m:
        candidates.append(arr_m.group(0))
    obj_m = re.search(r"\{[\s\S]*\}", cleaned)
    if obj_m:
        candidates.append(obj_m.group(0))

    for cand in candidates:
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if isinstance(obj, list):
            parsed_list = obj
            break
        if isinstance(obj, dict):
            for key in ("questions", "quiz", "items"):
                if isinstance(obj.get(key), list):
                    parsed_list = obj[key]
                    break
            if parsed_list is not None:
                break

    if isinstance(parsed_list, list):
        for item in parsed_list:
            if not isinstance(item, dict):
                continue
            q_text = str(item.get("question") or "").strip()
            if not q_text:
                continue
            raw_type = str(item.get("type") or "").strip().lower()
            raw_opts = item.get("options") if isinstance(item.get("options"), list) else []

            if mode in ("mcq", "true_false", "open"):
                q_type = mode
            elif raw_type in ("true_false", "tf", "boolean", "v_f"):
                q_type = "true_false"
            elif raw_type == "mcq" or len(raw_opts) >= 3:
                q_type = "mcq"
            elif len(raw_opts) == 2 and any("verdadero" in str(o).lower() or "true" in str(o).lower() for o in raw_opts):
                q_type = "true_false"
            else:
                q_type = "open"

            ts = _extract_timestamp(item.get("timestamp"))
            hint = str(item.get("hint") or "").strip()

            if q_type == "true_false":
                c_idx = 0
                raw_bool = item.get("correct_answer")
                if raw_bool is None and "is_true" in item:
                    raw_bool = item.get("is_true")
                if raw_bool is None and "answer" in item:
                    raw_bool = item.get("answer")

                if isinstance(raw_bool, bool):
                    c_idx = 0 if raw_bool else 1
                else:
                    raw_ca = str(raw_bool if raw_bool is not None else "").strip().lower()
                    if raw_ca in ("false", "falso", "f", "no", "incorrecto"):
                        c_idx = 1
                    elif raw_ca in ("true", "verdadero", "v", "si", "sí", "correcto"):
                        c_idx = 0
                    else:
                        try:
                            parsed_idx = int(item.get("correct_index", 0))
                        except (TypeError, ValueError):
                            parsed_idx = 0
                        if len(raw_opts) >= 2 and 0 <= parsed_idx < len(raw_opts):
                            opt_text = str(raw_opts[parsed_idx]).strip().lower()
                            if "falso" in opt_text or "false" in opt_text:
                                c_idx = 1
                            elif "verdadero" in opt_text or "true" in opt_text:
                                c_idx = 0
                            else:
                                c_idx = 1 if parsed_idx == 1 else 0
                        else:
                            c_idx = 1 if parsed_idx == 1 else 0
                questions.append(
                    {
                        "id": len(questions) + 1,
                        "type": "true_false",
                        "question": q_text,
                        "options": ["Verdadero", "Falso"],
                        "correct_index": c_idx,
                        "explanation": str(item.get("explanation") or "").strip(),
                        "hint": hint,
                        "timestamp": ts,
                    }
                )
                continue

            if q_type == "mcq":
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
                            "hint": hint,
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
                    "hint": hint,
                    "timestamp": ts,
                }
            )

    return questions


def is_true_false_unbalanced(questions: list[dict[str, Any]]) -> bool:
    """Return True if a quiz's True/False items collapsed into all-True, all-False, or extreme skew."""
    tf_items = [q for q in questions if q.get("type") == "true_false"]
    total = len(tf_items)
    if total < 2:
        return False
    true_count = sum(1 for q in tf_items if int(q.get("correct_index", 0)) == 0)
    false_count = total - true_count
    if true_count == 0 or false_count == 0:
        return True
    if total >= 5:
        min_side = min(true_count, false_count)
        if min_side < max(2, total // 4):
            return True
    return False


def build_quiz_rebalance_messages(
    original_messages: list[dict[str, str]],
    raw_json_response: str,
    questions: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build a corrective follow-up prompt so skewed True/False exams contain balanced Verdadero & Falso items."""
    tf_items = [q for q in questions if q.get("type") == "true_false"]
    total_tf = len(tf_items)
    true_items = [q for q in tf_items if int(q.get("correct_index", 0)) == 0]
    false_items = [q for q in tf_items if int(q.get("correct_index", 0)) == 1]

    desired_true = max(1, total_tf // 2)
    need_more_true = len(true_items) < len(false_items)

    if need_more_true:
        flip_needed = max(1, desired_true - len(true_items))
        candidates = [q["id"] for q in false_items[:flip_needed]]
        instruction = (
            f"ATENCIÓN: Tu examen de Verdadero/Falso quedó desbalanceado ({len(true_items)} Verdaderas y {len(false_items)} Falsas).\n"
            f"Reescribe ÚNICAMENTE las preguntas con id {candidates} para convertirlas en afirmaciones GENUINAMENTE VERDADERAS "
            "según el CONTENIDO DE ESTUDIO original (cambiando su `question` para que sea fiel a las fuentes, "
            "poniendo `correct_index`: 0 y actualizando `explanation`). Mantén todas las demás preguntas tal como están y devuelve "
            "el arreglo JSON completo con exactamente las mismas claves."
        )
    else:
        desired_false = max(1, total_tf - desired_true)
        flip_needed = max(1, desired_false - len(false_items))
        candidates = [q["id"] for q in true_items[:flip_needed]]
        instruction = (
            f"ATENCIÓN: Tu examen de Verdadero/Falso quedó desbalanceado ({len(true_items)} Verdaderas y {len(false_items)} Falsas).\n"
            f"Reescribe ÚNICAMENTE las preguntas con id {candidates} para convertirlas en afirmaciones GENUINAMENTE FALSAS "
            "mediante una contradicción sutil frente al CONTENIDO DE ESTUDIO original (cambiando su `question`, "
            "poniendo `correct_index`: 1 y actualizando `explanation`). Mantén todas las demás preguntas tal como están y devuelve "
            "el arreglo JSON completo con exactamente las mismas claves."
        )

    return [
        *original_messages,
        {"role": "assistant", "content": raw_json_response},
        {"role": "user", "content": instruction},
    ]


def shuffle_quiz_mcq_options(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Randomize MCQ option order so the correct option isn't stuck at index 0, keeping positional options intact."""
    out: list[dict[str, Any]] = []
    positional_re = re.compile(r"\b(?:todas|ninguna)\s+las\s+anteriores\b|\ball\s+of\s+the\s+above\b|\bnone\s+of\s+the\s+above\b", re.IGNORECASE)
    for q in questions:
        item = dict(q)
        if item.get("type") == "mcq" and isinstance(item.get("options"), list) and len(item["options"]) >= 2:
            opts = list(item["options"])
            c_idx = int(item.get("correct_index", 0))
            if 0 <= c_idx < len(opts) and not any(positional_re.search(str(o)) for o in opts):
                indexed = list(enumerate(opts))
                random.shuffle(indexed)
                new_opts = [opt for _old_i, opt in indexed]
                new_c_idx = next(new_i for new_i, (old_i, _opt) in enumerate(indexed) if old_i == c_idx)
                item["options"] = new_opts
                item["correct_index"] = new_c_idx
        out.append(item)
    return out


def extract_relevant_excerpt(
    recordings: list[dict[str, Any]],
    query: str,
    max_chars: int = 3800,
    sources: list[dict[str, Any]] | None = None,
    indexed_chunks: list[dict[str, Any]] | None = None,
) -> str:
    """Return top indexed/BM25 excerpts across active recordings and sources for grading an open-ended answer."""
    if indexed_chunks:
        picked = _select_chunks_within_budget(indexed_chunks, max_chars=max_chars, max_chunks=5)
        if picked:
            return "\n\n".join(
                f"{_format_indexed_chunk_header(c)}\n{c['content']}" for c in picked
            )

    ready_recs = [
        r for r in recordings
        if r.get("status") == "ready" and r.get("enabled", True) and r.get("transcript_text")
    ]
    ready_srcs = [
        s for s in (sources or [])
        if s.get("status") == "ready" and s.get("enabled", True) and s.get("extracted_text")
    ]
    all_chunks: list[dict[str, Any]] = []
    for rec in ready_recs:
        all_chunks.extend(_chunk_recording(rec))
    for src in ready_srcs:
        all_chunks.extend(_chunk_source(src))
    selected = _bm25_select_chunks(all_chunks, query, max_chars=max_chars, max_chunks=4)
    if selected:
        return "\n\n".join(f"[{c['filename']} @ {c['start_ts']}]\n{c['text']}" for c in selected)
    fallback_parts = [format_transcript_for_llm(r)[:1500] for r in ready_recs[:2]]
    fallback_parts.extend(_format_source_for_llm(s)[:1500] for s in ready_srcs[:2])
    return "\n\n".join(fallback_parts)


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
