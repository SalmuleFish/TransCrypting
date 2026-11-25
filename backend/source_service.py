"""Local CPU document extraction, chunking, image preparation, and WinRT OCR fallback."""

from __future__ import annotations

import base64
import io
import re
import subprocess
from pathlib import Path
from typing import Any

from backend.config import (
    SUPPORTED_DOCUMENT_EXTENSIONS,
    SUPPORTED_IMAGE_EXTENSIONS,
)

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _format_ts(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def detect_source_type(filename: str) -> str:
    """Return canonical source type: pdf | pptx | docx | text | image."""
    ext = Path(filename or "").suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext == ".pptx":
        return "pptx"
    if ext == ".docx":
        return "docx"
    if ext in {".txt", ".md"}:
        return "text"
    if ext in SUPPORTED_IMAGE_EXTENSIONS:
        return "image"
    if ext in SUPPORTED_DOCUMENT_EXTENSIONS:
        return "text"
    return "text"


def _clean_text_block(raw: str) -> str:
    text = (raw or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_document_pages(file_path: Path, source_type: str) -> list[dict[str, str]]:
    """
    Extract structured pages/slides/sections from local documents on CPU.

    Returns a list of dicts: {"locator": "p. 1", "label": "Página 1", "text": "..."}.
    """
    src_type = (source_type or detect_source_type(file_path.name)).lower()
    pages: list[dict[str, str]] = []

    if src_type == "pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))
        for idx, page in enumerate(reader.pages, start=1):
            raw_text = ""
            try:
                raw_text = page.extract_text() or ""
            except Exception:
                raw_text = ""
            cleaned = _clean_text_block(raw_text)
            if not cleaned:
                cleaned = "[Página sin texto seleccionable o escaneada]"
            pages.append(
                {
                    "locator": f"p. {idx}",
                    "label": f"Página {idx}",
                    "text": cleaned,
                }
            )
        return pages

    if src_type == "pptx":
        from pptx import Presentation

        prs = Presentation(str(file_path))
        for idx, slide in enumerate(prs.slides, start=1):
            parts: list[str] = []
            for shape in slide.shapes:
                if getattr(shape, "has_text_frame", False) and shape.text_frame is not None:
                    txt = (shape.text_frame.text or "").strip()
                    if txt:
                        parts.append(txt)
                if getattr(shape, "has_table", False) and shape.table is not None:
                    for row in shape.table.rows:
                        cells = [
                            (cell.text or "").strip().replace("\n", " ")
                            for cell in row.cells
                        ]
                        row_line = " | ".join(c for c in cells if c)
                        if row_line:
                            parts.append(row_line)
            try:
                if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
                    notes = (slide.notes_slide.notes_text_frame.text or "").strip()
                    if notes:
                        parts.append(f"Notas del presentador: {notes}")
            except Exception:
                pass
            cleaned = _clean_text_block("\n".join(parts)) or "[Diapositiva sin texto]"
            pages.append(
                {
                    "locator": f"diap. {idx}",
                    "label": f"Diapositiva {idx}",
                    "text": cleaned,
                }
            )
        return pages

    if src_type == "docx":
        from docx import Document

        doc = Document(str(file_path))
        blocks: list[str] = []
        for para in doc.paragraphs:
            t = (para.text or "").strip()
            if t:
                blocks.append(t)
        for table in doc.tables:
            for row in table.rows:
                cells = [
                    (cell.text or "").strip().replace("\n", " ")
                    for cell in row.cells
                ]
                line = " | ".join(c for c in cells if c)
                if line:
                    blocks.append(line)
        return _group_blocks_into_sections(blocks, max_chars=1800)

    # Plain text / Markdown
    raw_bytes = file_path.read_bytes()
    decoded = ""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            decoded = raw_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if not decoded:
        decoded = raw_bytes.decode("utf-8", errors="replace")

    cleaned = _clean_text_block(decoded)
    if not cleaned:
        return [{"locator": "sec. 1", "label": "Sección 1", "text": "[Archivo vacío]"}]

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", cleaned) if p.strip()]
    return _group_blocks_into_sections(paragraphs, max_chars=1800)


def _group_blocks_into_sections(
    blocks: list[str], max_chars: int = 1800
) -> list[dict[str, str]]:
    if not blocks:
        return [{"locator": "sec. 1", "label": "Sección 1", "text": "[Documento sin texto]"}]

    sections: list[dict[str, str]] = []
    cur: list[str] = []
    cur_len = 0

    for block in blocks:
        if cur and cur_len + len(block) + 2 > max_chars:
            idx = len(sections) + 1
            sections.append(
                {
                    "locator": f"sec. {idx}",
                    "label": f"Sección {idx}",
                    "text": "\n\n".join(cur),
                }
            )
            cur = [block]
            cur_len = len(block)
        else:
            cur.append(block)
            cur_len += len(block) + 2

    if cur:
        idx = len(sections) + 1
        sections.append(
            {
                "locator": f"sec. {idx}",
                "label": f"Sección {idx}",
                "text": "\n\n".join(cur),
            }
        )
    return sections


def pages_to_full_text(pages: list[dict[str, str]]) -> str:
    """Render extracted pages into a single structured text document with locators."""
    parts = []
    for page in pages:
        loc = (page.get("locator") or "").strip()
        txt = (page.get("text") or "").strip()
        if not txt:
            continue
        if loc:
            parts.append(f"[{loc}]\n{txt}")
        else:
            parts.append(txt)
    return "\n\n".join(parts).strip()


def chunk_pages(
    pages: list[dict[str, str]],
    max_chars: int = 1100,
    overlap_chars: int = 140,
) -> list[dict[str, str]]:
    """Split extracted document pages into overlapping search chunks for FTS5."""
    chunks: list[dict[str, str]] = []
    for page in pages:
        locator = (page.get("locator") or "").strip() or "p. 1"
        text = (page.get("text") or "").strip()
        if not text:
            continue
        if len(text) <= max_chars:
            chunks.append({"locator": locator, "content": text})
            continue

        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(text_len, start + max_chars)
            if end < text_len:
                split_at = text.rfind(" ", start + max_chars // 2, end)
                if split_at > start:
                    end = split_at
            piece = text[start:end].strip()
            if piece:
                chunks.append({"locator": locator, "content": piece})
            if end >= text_len:
                break
            next_start = max(start + 1, end - overlap_chars)
            start = next_start
    return chunks


def chunk_recording_for_index(
    rec: dict[str, Any],
    max_chars: int = 1100,
) -> list[dict[str, str]]:
    """Convert a recording's transcript segments into [MM:SS]-located FTS5 chunks."""
    segments = rec.get("segments") or []
    if not segments:
        raw = (rec.get("transcript_text") or "").strip()
        if not raw:
            return []
        return chunk_pages([{"locator": "00:00", "text": raw}], max_chars=max_chars)

    chunks: list[dict[str, str]] = []
    buf: list[str] = []
    buf_len = 0
    chunk_start_ts = _format_ts(float(segments[0].get("start", 0.0)))

    for seg in segments:
        seg_text = (seg.get("text") or "").strip()
        if not seg_text:
            continue
        ts = _format_ts(float(seg.get("start", 0.0)))
        line = f"[{ts}] {seg_text}"
        if buf and buf_len + len(line) + 1 > max_chars:
            chunks.append({"locator": chunk_start_ts, "content": "\n".join(buf)})
            buf = [line]
            buf_len = len(line)
            chunk_start_ts = ts
        else:
            if not buf:
                chunk_start_ts = ts
            buf.append(line)
            buf_len += len(line) + 1

    if buf:
        chunks.append({"locator": chunk_start_ts, "content": "\n".join(buf)})
    return chunks


def prepare_image_base64(image_path: Path, max_dim: int = 1536) -> str:
    """Normalize an image to RGB JPEG (<= max_dim px) and encode as base64."""
    from PIL import Image, ImageOps

    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode not in ("RGB", "L"):
            rgba = img.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[3])
            rgb = bg
        else:
            rgb = img.convert("RGB")

        w, h = rgb.size
        longest = max(w, h)
        if longest > max_dim:
            scale = max_dim / float(longest)
            new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
            resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)
            rgb = rgb.resize(new_size, resampling)

        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=85, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")


def build_image_analysis_messages(filename: str, image_b64: str) -> list[dict[str, Any]]:
    """Build a single-pass multimodal prompt to transcribe and index an image once."""
    return [
        {
            "role": "system",
            "content": (
                "Eres un asistente académico experto en transcribir y describir imágenes de estudio "
                "(pizarrones, diapositivas, apuntes, diagramas, fórmulas o tablas) en español claro."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Analiza esta imagen ('{filename}') para indexarla y poder responder preguntas sobre ella después.\n"
                "Incluye en formato Markdown:\n"
                "1. **Texto visible**: transcribe todo el texto, títulos, viñetas, números y fórmulas tal como aparecen.\n"
                "2. **Diagrama o estructura visual**: explica qué relaciones, gráficas, flechas o conceptos muestra la imagen.\n"
                "No inventes texto ilegible; si algo no se distingue con claridad, indícalo."
            ),
            "images": [image_b64],
        },
    ]


def extract_image_text_windows_ocr(image_path: Path) -> str:
    """Safe CPU OCR fallback using built-in Windows WinRT OcrEngine via PowerShell."""
    ps_script = r"""
param([string]$ImgPath)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime
[void][Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]
[void][Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics, ContentType = WindowsRuntime]
[void][Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime]
$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
function Await($WinRtTask, $ResultType) {
    $netTask = $asTaskGeneric.MakeGenericMethod($ResultType).Invoke($null, @($WinRtTask))
    $netTask.Wait(-1) | Out-Null
    return $netTask.Result
}
$file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($ImgPath)) ([Windows.Storage.StorageFile])
$stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
$decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
$bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
if ($null -eq $engine) {
    $lang = New-Object Windows.Globalization.Language("en-US")
    $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($lang)
}
if ($null -ne $engine) {
    $ocrResult = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    Write-Output $ocrResult.Text
}
"""
    try:
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                ps_script,
                "-ImgPath",
                str(image_path.resolve()),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            creationflags=_NO_WINDOW,
        )
        if completed.returncode == 0:
            return _clean_text_block(completed.stdout or "")
    except Exception:
        pass
    return ""
