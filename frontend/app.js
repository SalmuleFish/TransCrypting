/* TransCrypting Frontend — Bento Lime/Obsidian/Porcelain (100% local) */

const AUDIO_EXTENSIONS = new Set([
  ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".aac", ".wma", ".mp4", ".mkv", ".mov", ".opus",
]);
const SOURCE_EXTENSIONS = new Set([
  ".pdf", ".pptx", ".docx", ".txt", ".md", ".png", ".jpg", ".jpeg", ".webp", ".bmp",
]);
const TAB_ORDER = ["summary", "transcript", "chat", "quiz", "podcast"];
const PARAKEET_MODEL_ID = "parakeet-tdt-0.6b-v3";

const state = {
  sessions: [],
  activeSessionId: null,
  activeSession: null,
  activeRecordingId: null,
  systemStatus: null,
  activeTab: "summary",
  isStreaming: false,
  isResummarizing: false,
  chatAbortController: null,
  resummarizeAbortController: null,
  quizAbortController: null,
  quizGradeAbortController: null,
  stoppingAI: false,
  lastContextMode: "full",
  theme: "light",
  accentColor: "#c8ee26",
  fontScale: 1.0,
  appearancePreviewBackup: null,
  confirmCallback: null,
  pollTimer: null,
  fetchingSuggestionsFor: new Set(),
  attemptedSuggestionsKeys: new Set(),
  podcastBusy: false,
  podcastStartedAt: null,
  podcastElapsedTimer: null,
  cancellingPodcastId: null,
  transcriptRecFilter: null,
  transcriptSpeakerFilter: null,
  captureMode: "mic",
};

const audioPlayers = new Map();
const podcastAudioPlayers = new Map();
const podcastFormState = new Map();
const sessionQuizzes = new Map();
const renderMemo = new Map();
const customSelectControllers = new Map();
let activeDropdownMenu = null;
let activeDropdownWrapper = null;
let microphoneCapture = null;
let systemCaptureState = null;
let microphonePreview = null;
let transcriptPlayback = null;

function memoChanged(key, sig) {
  if (renderMemo.get(key) === sig) return false;
  renderMemo.set(key, sig);
  return true;
}

function invalidateMemo(prefix = "") {
  if (!prefix) {
    renderMemo.clear();
    return;
  }
  for (const k of Array.from(renderMemo.keys())) {
    if (k.startsWith(prefix)) renderMemo.delete(k);
  }
}

// --- Helpers ---

async function api(url, options = {}) {
  const resp = await fetch(url, options);
  if (!resp.ok) {
    let detail = `Error HTTP ${resp.status}`;
    try {
      const data = await resp.json();
      if (data.detail) detail = data.detail;
    } catch (_) {}
    throw new Error(detail);
  }
  const ct = resp.headers.get("content-type") || "";
  if (ct.includes("application/json")) return resp.json();
  return resp;
}

function showToast(message, isError = false) {
  const container = document.getElementById("toast-container");
  const el = document.createElement("div");
  el.className = `toast ${isError ? "error" : ""}`;
  el.textContent = message;
  container.appendChild(el);
  setTimeout(() => {
    el.classList.add("leaving");
    setTimeout(() => el.remove(), 200);
  }, 3600);
}

function getActiveModelLabel() {
  return state.systemStatus?.settings?.ollama_model || "Modelo local";
}

function escapeHtml(str) {
  return String(str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatDuration(sec) {
  const total = Math.max(0, Math.round(sec || 0));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const pad = (n) => String(n).padStart(2, "0");
  return h > 0 ? `${pad(h)}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
}

function parseTimestampToSeconds(tsStr) {
  const parts = String(tsStr || "")
    .trim()
    .split(":")
    .map((n) => Number(n));
  if (parts.some((n) => Number.isNaN(n) || n < 0)) return 0;
  if (parts.length === 2) return parts[0] * 60 + parts[1];
  if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
  return 0;
}

function normalizeRefTokens(str) {
  return String(str || "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/\.(?:mp3|wav|m4a|ogg|flac|webm|aac|wma|mp4|mkv|mov|opus|pdf|pptx|docx|txt|md|png|jpg|jpeg|webp|bmp)\b/gi, "")
    .replace(/([a-z])([A-Z0-9])/g, "$1 $2")
    .replace(/([a-zA-Z])([0-9])/g, "$1 $2")
    .replace(/[_\-./]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function findMatchingRecordingByRef(refText) {
  const recordings = (state.activeSession?.recordings || []).filter(
    (r) => r.status === "ready" || r.duration_sec > 0 || r.transcript_text
  );
  if (!recordings.length) return null;

  const raw = String(refText || "").trim().replace(/^\[|\]$/g, "");
  if (!raw) return null;

  const rawLower = raw.toLowerCase();
  const exactFile = recordings.find((r) => (r.filename || "").toLowerCase() === rawLower);
  if (exactFile) return exactFile;

  const normRef = normalizeRefTokens(raw);
  if (!normRef) return null;

  // Match "audio 1", "grabacion 2", etc.
  const idxMatch = normRef.match(/^(?:audio|grabacion|recording|pista)\s*(\d+)$/i);
  if (idxMatch) {
    const idx = Number(idxMatch[1]) - 1;
    if (idx >= 0 && idx < recordings.length) return recordings[idx];
  }

  for (const r of recordings) {
    if (normalizeRefTokens(r.filename) === normRef) return r;
  }

  // Substring containment when name fragment is reasonably specific (>= 4 chars)
  if (normRef.length >= 4) {
    const subMatch = recordings.find((r) => {
      const nr = normalizeRefTokens(r.filename);
      return nr && (nr.includes(normRef) || normRef.includes(nr));
    });
    if (subMatch) return subMatch;
  }

  // Token overlap match (handles slight rewording or dropped suffixes)
  const refTokens = normRef.split(" ").filter((t) => t.length >= 2);
  if (refTokens.length >= 2) {
    let bestRec = null;
    let bestScore = 0;
    for (const r of recordings) {
      const recTokens = new Set(normalizeRefTokens(r.filename).split(" ").filter((t) => t.length >= 2));
      if (recTokens.size === 0) continue;
      let hits = 0;
      for (const t of refTokens) {
        if (recTokens.has(t)) hits += 1;
      }
      const ratio = hits / Math.min(refTokens.length, recTokens.size);
      if (hits >= 2 && ratio >= 0.6 && hits > bestScore) {
        bestScore = hits;
        bestRec = r;
      }
    }
    if (bestRec) return bestRec;
  }

  return null;
}

function makeTimestampPillHtml(tsStr, rec = null) {
  const clean = tsStr.trim();
  const seconds = parseTimestampToSeconds(clean);
  const readyCount = (state.activeSession?.recordings || []).filter(
    (r) => r.status === "ready" || r.duration_sec > 0 || r.transcript_text
  ).length;
  let displayText = clean;
  let titleText = `Ir al momento ${clean}`;
  const recAttr = rec?.id ? ` data-rec-id="${escapeHtml(rec.id)}"` : "";
  if (rec?.filename) {
    titleText = `Ir a ${rec.filename} en ${clean}`;
    if (readyCount >= 2) {
      const stem = String(rec.filename).replace(/\.[^.]+$/, "");
      const shortStem = stem.length > 22 ? `${stem.slice(0, 20).trim()}…` : stem;
      displayText = `${shortStem} · ${clean}`;
    }
  }
  return `<button type="button" class="chat-ts-pill"${recAttr} data-seconds="${seconds}" data-label="${escapeHtml(clean)}" title="${escapeHtml(titleText)}"><span class="ts-play-tri">▶</span>${escapeHtml(displayText)}</button>`;
}

function findMatchingSourceByRef(refText) {
  const sources = state.activeSession?.sources || [];
  if (!sources.length) return { source: null, locator: "" };
  const raw = String(refText || "").trim().replace(/^\[|\]$/g, "");
  const parts = raw.split(/\s*[·|]\s*/);
  const namePart = (parts[0] || "").trim().toLowerCase();
  const locator = (parts[1] || "").trim();
  const normName = normalizeRefTokens(namePart);
  const matched =
    sources.find((s) => (s.filename || "").toLowerCase() === namePart) ||
    sources.find((s) => namePart && (s.filename || "").toLowerCase().includes(namePart)) ||
    (normName.length >= 4
      ? sources.find((s) => {
          const ns = normalizeRefTokens(s.filename);
          return ns === normName || ns.includes(normName) || normName.includes(ns);
        })
      : null);
  if (matched) return { source: matched, locator };
  return { source: null, locator };
}

function makeSourcePillHtml(source, label, locator = "") {
  const icon = source?.source_type === "image" ? "🖼" : "📄";
  return `<button type="button" class="chat-src-pill" data-source-id="${escapeHtml(source?.id || "")}" data-locator="${escapeHtml(locator)}" title="Abrir fuente: ${escapeHtml(label)}"><span>${icon}</span><span>${escapeHtml(label)}</span></button>`;
}

function parseAudioCitationInner(inner) {
  const text = String(inner || "").trim();
  if (!text) return null;
  if (/(?:^|[\s·|])(?:p\.|diap\.|sec\.|p[aá]gina)\s*\d+/i.test(text)) return null;
  if (/^https?:\/\//i.test(text)) return null;

  const tsRe = /\b\d{1,2}:\d{2}(?::\d{2})?\b/g;
  const timestamps = text.match(tsRe);
  if (!timestamps || timestamps.length === 0) return null;

  const stripped = text
    .replace(tsRe, " ")
    .replace(/[·|@—–\-,]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  if (!stripped) {
    return { rec: null, timestamps };
  }

  const matchedRec = findMatchingRecordingByRef(stripped);
  if (matchedRec) {
    return { rec: matchedRec, timestamps };
  }

  // Fallback for slight LLM spelling differences if it isn't a doc source and looks like a compact ref
  const hasDocExt = /\.(?:pdf|pptx|docx|txt|md|png|jpg|jpeg|webp|bmp)\b/i.test(stripped);
  if (!hasDocExt && !/[?!]/.test(stripped) && stripped.split(/\s+/).length <= 8) {
    const srcCheck = findMatchingSourceByRef(stripped);
    if (!srcCheck.source) {
      return { rec: null, timestamps };
    }
  }
  return null;
}

function makeCitationPillHtml(refValue) {
  const clean = String(refValue || "").trim();
  if (!clean) return "";
  const audioParsed = parseAudioCitationInner(clean.replace(/^\[|\]$/g, ""));
  if (audioParsed) {
    return audioParsed.timestamps.map((t) => makeTimestampPillHtml(t, audioParsed.rec)).join(" ");
  }
  const { source, locator } = findMatchingSourceByRef(clean);
  if (source) {
    return makeSourcePillHtml(source, clean, locator);
  }
  return `<span class="chat-src-pill" style="cursor:default;">📄 ${escapeHtml(clean)}</span>`;
}

function enhanceTimestampsInHtml(html) {
  const protectedCode = [];
  const masked = html.replace(/<pre[\s\S]*?<\/pre>|<code[\s\S]*?<\/code>/gi, (match) => {
    const idx = protectedCode.length;
    protectedCode.push(match);
    return `@@PROTECTED_CODE_${idx}@@`;
  });

  const bracketAnyRe = /\[([^\[\]\n]{3,140})\]/g;
  const sources = state.activeSession?.sources || [];
  const docRefRe = /\[([^\[\]\n]{3,90}?\.(?:pdf|pptx|docx|txt|md|png|jpg|jpeg|webp|bmp)(?:\s*[·|]\s*(?:p\.|diap\.|sec\.)\s*\d+)?)\]/gi;
  const docAliasWithLocRe = /\[([^\[\]\n]{3,90}?\s*[·|]\s*(?:p\.|diap\.|sec\.)\s*\d+)\]/gi;

  const parts = masked.split(/(<[^>]+>)/g);
  for (let i = 0; i < parts.length; i += 2) {
    parts[i] = parts[i].replace(bracketAnyRe, (full, inner) => {
      const parsed = parseAudioCitationInner(inner);
      if (!parsed) return full;
      return parsed.timestamps.map((t) => makeTimestampPillHtml(t, parsed.rec)).join(" ");
    });
    if (sources.length > 0) {
      parts[i] = parts[i].replace(docRefRe, (full, inner) => {
        const { source, locator } = findMatchingSourceByRef(inner);
        if (!source) return full;
        return makeSourcePillHtml(source, inner.trim(), locator);
      });
      parts[i] = parts[i].replace(docAliasWithLocRe, (full, inner) => {
        const { source, locator } = findMatchingSourceByRef(inner);
        if (!source) return full;
        return makeSourcePillHtml(source, inner.trim(), locator);
      });
    }
  }

  return parts.join("").replace(/@@PROTECTED_CODE_(\d+)@@/g, (_, idx) => protectedCode[Number(idx)] || "");
}

function renderMarkdown(raw, withTimestampPills = false) {
  const escaped = escapeHtml(raw || "");
  const codeBlocks = [];
  const text = escaped.replace(/```([\s\S]*?)```/g, (_, code) => {
    const idx = codeBlocks.length;
    codeBlocks.push(`<pre><code>${code.trim()}</code></pre>`);
    return `@@CODEBLOCK_${idx}@@`;
  });

  const lines = text.split("\n");
  const out = [];
  let inList = false;

  const formatInline = (line) =>
    line
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s.,;:!?)]|$)/g, "$1<em>$2</em>")
      .replace(/(^|[\s(])_([^_\n]+)_(?=[\s.,;:!?)]|$)/g, "$1<em>$2</em>");

  for (const rawLine of lines) {
    const line = rawLine.trimEnd();
    if (/^@@CODEBLOCK_\d+@@$/.test(line.trim())) {
      if (inList) {
        out.push("</ul>");
        inList = false;
      }
      out.push(line.trim());
      continue;
    }

    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) {
      if (inList) {
        out.push("</ul>");
        inList = false;
      }
      const lvl = heading[1].length;
      out.push(`<h${lvl}>${formatInline(heading[2])}</h${lvl}>`);
      continue;
    }

    const quote = line.match(/^&gt;\s?(.*)$/);
    if (quote) {
      if (inList) {
        out.push("</ul>");
        inList = false;
      }
      out.push(`<blockquote>${formatInline(quote[1])}</blockquote>`);
      continue;
    }

    const bullet = line.match(/^\s*[-*•]\s+(.*)$/) || line.match(/^\s*\d+\.\s+(.*)$/);
    if (bullet) {
      if (!inList) {
        out.push("<ul>");
        inList = true;
      }
      out.push(`<li>${formatInline(bullet[1])}</li>`);
      continue;
    }

    if (inList) {
      out.push("</ul>");
      inList = false;
    }

    if (line.trim() === "") continue;
    out.push(`<p>${formatInline(line)}</p>`);
  }

  if (inList) out.push("</ul>");
  let html = out.join("").replace(/@@CODEBLOCK_(\d+)@@/g, (_, idx) => codeBlocks[Number(idx)] || "");
  if (withTimestampPills) {
    html = enhanceTimestampsInHtml(html);
  }
  return html;
}

// --- SVG Icons ---

const ICON_PLAY = `<svg viewBox="0 0 24 24" width="13" height="13" fill="currentColor"><polygon points="6 4 20 12 6 20 6 4"/></svg>`;
const ICON_PAUSE = `<svg viewBox="0 0 24 24" width="13" height="13" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>`;
const ICON_TRASH = `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/></svg>`;
const ICON_RETRANSCRIBE = `<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>`;
const ICON_CHEVRON = `<svg class="bento-select-chevron" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>`;
const ICON_CHECK = `<svg class="bento-select-check" viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>`;
const ICON_SPARKLE = `<svg class="chip-sparkle" viewBox="0 0 24 24" width="14" height="14" fill="currentColor" aria-hidden="true"><path d="M12 2.5L14.4 9.6L21.5 12L14.4 14.4L12 21.5L9.6 14.4L2.5 12L9.6 9.6Z"/></svg>`;
const ICON_SEND_ARROW = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/></svg>`;
const ICON_STOP_SQUARE = `<svg viewBox="0 0 24 24" width="13" height="13" fill="currentColor" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="2.5"/></svg>`;

// --- Theme (Light / Soft Graphite Dark) ---

function applyTheme(theme, persist = false) {
  const normalized = theme === "dark" ? "dark" : "light";
  state.theme = normalized;
  document.documentElement.dataset.theme = normalized;

  const moon = document.getElementById("theme-icon-moon");
  const sun = document.getElementById("theme-icon-sun");
  if (moon && sun) {
    moon.classList.toggle("hidden", normalized === "dark");
    sun.classList.toggle("hidden", normalized !== "dark");
  }

  try {
    localStorage.setItem("transcript_theme", normalized);
  } catch (_) {}

  if (window.pywebview?.api?.set_titlebar_theme) {
    window.pywebview.api.set_titlebar_theme(normalized).catch(() => {});
  }

  if (persist) {
    api("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ theme: normalized }),
    }).catch(() => {});
  }
}

function normalizeHexColor(hex) {
  const clean = String(hex || "").trim().toLowerCase();
  return /^#[0-9a-f]{6}$/.test(clean) ? clean : "#c8ee26";
}

function hexToRgb(hex) {
  const clean = normalizeHexColor(hex).slice(1);
  return {
    r: parseInt(clean.slice(0, 2), 16),
    g: parseInt(clean.slice(2, 4), 16),
    b: parseInt(clean.slice(4, 6), 16),
  };
}

function rgbToHex(r, g, b) {
  const clamp = (n) => Math.max(0, Math.min(255, Math.round(n))).toString(16).padStart(2, "0");
  return `#${clamp(r)}${clamp(g)}${clamp(b)}`;
}

function computeAccentInk(r, g, b) {
  const toLinear = (v) => {
    const c = v / 255;
    return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  };
  const lum = 0.2126 * toLinear(r) + 0.7152 * toLinear(g) + 0.0722 * toLinear(b);
  return lum > 0.42 ? "#141716" : "#ffffff";
}

function normalizeFontScale(val) {
  const num = Number(val);
  if (!Number.isFinite(num)) return 1.0;
  return Math.round(Math.max(0.85, Math.min(1.25, num)) * 100) / 100;
}

function getFontScaleLabel(scale) {
  if (scale <= 0.92) return "Pequeña";
  if (scale <= 1.03) return "Normal";
  if (scale <= 1.14) return "Grande";
  return "Extra grande";
}

function syncAppearanceUI(accentColor, fontScale) {
  const hex = normalizeHexColor(accentColor);
  const scale = normalizeFontScale(fontScale);
  const customInput = document.getElementById("set-accent-custom");
  const hexLabel = document.getElementById("accent-hex-label");
  if (customInput && customInput.value.toLowerCase() !== hex) customInput.value = hex;
  if (hexLabel) hexLabel.textContent = hex.toUpperCase();

  document.querySelectorAll("#accent-swatches .accent-swatch").forEach((btn) => {
    const on = btn.dataset.accent?.toLowerCase() === hex;
    btn.classList.toggle("active", on);
    btn.setAttribute("aria-checked", String(on));
  });

  const slider = document.getElementById("set-font-scale");
  const badge = document.getElementById("font-scale-badge");
  if (slider && Number(slider.value) !== scale) slider.value = String(scale);
  if (badge) badge.textContent = `${Math.round(scale * 100)}% · ${getFontScaleLabel(scale)}`;

  document.querySelectorAll("#font-scale-group .seg-control-item").forEach((btn) => {
    const on = Math.abs(Number(btn.dataset.fontScale) - scale) < 0.025;
    btn.classList.toggle("active", on);
    btn.setAttribute("aria-pressed", String(on));
  });
}

function applyAppearance({ accentColor, fontScale }, persistLocal = true) {
  const hex = normalizeHexColor(accentColor ?? state.accentColor);
  const scale = normalizeFontScale(fontScale ?? state.fontScale);
  state.accentColor = hex;
  state.fontScale = scale;

  const { r, g, b } = hexToRgb(hex);
  const hoverHex = rgbToHex(r * 0.88, g * 0.88, b * 0.88);
  const tintHex = rgbToHex(r + (255 - r) * 0.8, g + (255 - g) * 0.8, b + (255 - b) * 0.8);
  const ink = computeAccentInk(r, g, b);

  const root = document.documentElement;
  root.style.setProperty("--lime", hex);
  root.style.setProperty("--lime-hover", hoverHex);
  root.style.setProperty("--lime-tint", tintHex);
  root.style.setProperty("--accent-rgb", `${r}, ${g}, ${b}`);
  root.style.setProperty("--accent-ink", ink);
  root.style.setProperty("--font-scale", String(scale));

  if (persistLocal) {
    try {
      localStorage.setItem("transcript_accent_color", hex);
      localStorage.setItem("transcript_font_scale", String(scale));
    } catch (_) {}
  }

  syncAppearanceUI(hex, scale);
  requestAnimationFrame(updateTabIndicator);
}

// --- Modal Open/Close Helpers ---

function openModal(modalId) {
  const el = document.getElementById(modalId);
  if (el) el.classList.add("is-open");
}

function closeModal(modalId) {
  const el = document.getElementById(modalId);
  if (el) el.classList.remove("is-open");
}

function isModalOpen(modalId) {
  const el = document.getElementById(modalId);
  return Boolean(el && el.classList.contains("is-open"));
}

// --- Custom Bento Select Component ---

function closeActiveDropdown() {
  if (activeDropdownMenu) {
    const closingMenu = activeDropdownMenu;
    activeDropdownMenu = null;
    closingMenu.classList.add("closing");
    setTimeout(() => closingMenu.remove(), 140);
  }
  if (activeDropdownWrapper) {
    activeDropdownWrapper.classList.remove("open");
    activeDropdownWrapper = null;
  }
}

function initCustomSelect(selectEl) {
  if (!selectEl || customSelectControllers.has(selectEl.id)) return;

  selectEl.classList.add("native-select-hidden");
  selectEl.tabIndex = -1;

  const wrapper = document.createElement("div");
  wrapper.className = "bento-select";
  wrapper.dataset.for = selectEl.id;

  const trigger = document.createElement("button");
  trigger.type = "button";
  trigger.className = "bento-select-trigger";

  const valueSpan = document.createElement("span");
  valueSpan.className = "bento-select-value";

  trigger.appendChild(valueSpan);
  trigger.insertAdjacentHTML("beforeend", ICON_CHEVRON);

  selectEl.parentNode.insertBefore(wrapper, selectEl.nextSibling);
  wrapper.appendChild(trigger);

  const syncLabel = () => {
    const selectedOpt = selectEl.options[selectEl.selectedIndex];
    valueSpan.textContent = selectedOpt ? selectedOpt.textContent : "";
  };

  const openOrToggleMenu = (e) => {
    e.stopPropagation();
    const isSame = activeDropdownWrapper === wrapper;
    closeActiveDropdown();
    if (isSame) return;

    wrapper.classList.add("open");
    activeDropdownWrapper = wrapper;

    const menu = document.createElement("div");
    menu.className = "bento-select-menu";

    for (const opt of Array.from(selectEl.options)) {
      const item = document.createElement("button");
      item.type = "button";
      item.disabled = opt.disabled;
      item.className = `bento-select-option ${opt.value === selectEl.value ? "selected" : ""}`;

      const lbl = document.createElement("span");
      lbl.textContent = opt.textContent;
      item.appendChild(lbl);
      item.insertAdjacentHTML("beforeend", ICON_CHECK);

      item.onclick = (ev) => {
        ev.stopPropagation();
        selectEl.value = opt.value;
        selectEl.dispatchEvent(new Event("change", { bubbles: true }));
        syncLabel();
        closeActiveDropdown();
      };
      menu.appendChild(item);
    }

    document.body.appendChild(menu);
    activeDropdownMenu = menu;

    const rect = trigger.getBoundingClientRect();
    const menuRect = menu.getBoundingClientRect();
    const spaceBelow = window.innerHeight - rect.bottom - 12;
    const openUp = spaceBelow < Math.min(240, menuRect.height + 8) && rect.top > spaceBelow;

    menu.style.width = `${rect.width}px`;
    menu.style.left = `${rect.left}px`;
    if (openUp) {
      menu.style.top = `${Math.max(10, rect.top - menuRect.height - 6)}px`;
    } else {
      menu.style.top = `${rect.bottom + 6}px`;
    }
  };

  trigger.onclick = openOrToggleMenu;

  const fieldLabel = selectEl.closest(".field")?.querySelector(".field-label");
  if (fieldLabel) {
    fieldLabel.addEventListener("click", (e) => {
      e.preventDefault();
      openOrToggleMenu(e);
    });
  }

  selectEl.addEventListener("change", syncLabel);
  syncLabel();
  customSelectControllers.set(selectEl.id, { syncLabel });
}

function initAllCustomSelects() {
  document.querySelectorAll("#settings-form select").forEach(initCustomSelect);
}

function refreshCustomSelect(selectId) {
  const ctrl = customSelectControllers.get(selectId);
  if (ctrl) ctrl.syncLabel();
}

// --- Confirmation & Rename Modals ---

function openConfirm(title, message, onConfirm) {
  state.confirmCallback = onConfirm;
  document.getElementById("confirm-title").textContent = title;
  document.getElementById("confirm-message").textContent = message;
  openModal("confirm-modal");
}

function closeConfirm() {
  state.confirmCallback = null;
  closeModal("confirm-modal");
}

function openRenameModal() {
  if (!state.activeSession) return;
  const input = document.getElementById("rename-input");
  input.value = state.activeSession.title || "";
  openModal("rename-modal");
  setTimeout(() => {
    input.focus();
    input.select();
  }, 60);
}

function closeRenameModal() {
  closeModal("rename-modal");
}

function isAbortError(err) {
  if (!err) return false;
  if (err.name === "AbortError") return true;
  const msg = String(err.message || "");
  return /abort|detenid|cancel/i.test(msg);
}

function isAnyAIWorkActive() {
  if (state.isStreaming || state.isResummarizing || state.podcastBusy) return true;
  const res = state.systemStatus?.resources;
  if (res?.active_operation) return true;
  if (res?.phase && ["llm", "whisper", "podcast"].includes(res.phase)) return true;

  if (state.activeSessionId) {
    const qs = sessionQuizzes.get(state.activeSessionId);
    if (qs) {
      if (qs.loading || qs.submittingExam) return true;
      if (Object.values(qs.answers || {}).some((a) => a && a.grading)) return true;
    }
  }

  const sess = state.activeSession;
  if (sess) {
    const workingRec = (sess.recordings || []).some((r) =>
      ["queued", "preparing", "transcribing", "summarizing"].includes(r.status)
    );
    const workingSrc = (sess.sources || []).some((s) =>
      ["queued", "extracting", "indexing", "summarizing"].includes(s.status)
    );
    if (workingRec || workingSrc) return true;
  }
  return false;
}

function syncGlobalStopButton() {
  const btn = document.getElementById("btn-stop-ai");
  if (!btn) return;
  const operation = state.systemStatus?.resources?.active_operation;
  const active = isAnyAIWorkActive();
  const task = {
    audio: "transcripción de audio",
    text: "generación de texto",
    podcast: "generación del podcast",
  }[operation?.kind];
  btn.classList.toggle("hidden", !active);
  btn.disabled = Boolean(state.stoppingAI || operation?.stopping);
  btn.title = task ? `Detener ${task}` : "Detener la tarea activa";
  btn.setAttribute("aria-label", btn.title);
}

async function stopActiveAIOperation() {
  const operation = state.systemStatus?.resources?.active_operation;
  if (operation?.stopping || state.stoppingAI || !isAnyAIWorkActive()) return;
  state.stoppingAI = true;
  syncGlobalStopButton();
  try {
    if (state.chatAbortController) state.chatAbortController.abort();
    if (state.resummarizeAbortController) state.resummarizeAbortController.abort();
    if (state.quizAbortController) state.quizAbortController.abort();
    if (state.quizGradeAbortController) state.quizGradeAbortController.abort();

    if (operation) {
      const result = await api(`/api/system/stop-active/${operation.id}`, { method: "POST" });
      showToast(result.stopped ? "Deteniendo la tarea activa…" : "Esa tarea ya terminó.");
    } else {
      await stopBackendAIGeneration();
      showToast("Deteniendo la tarea activa…");
    }
    schedulePollingIfProcessing();
  } catch (err) {
    showToast(err.message, true);
  } finally {
    await refreshSystemStatus();
    state.stoppingAI = false;
    syncGlobalStopButton();
  }
}

async function stopBackendAIGeneration() {
  try {
    await fetch("/api/system/stop", { method: "POST" });
  } catch (_) {}
}

async function forceStopAllAI(options = {}) {
  if (state.stoppingAI) return;
  state.stoppingAI = true;
  syncGlobalStopButton();

  try {
    if (state.chatAbortController) {
      state.chatAbortController.abort();
    }
    if (state.resummarizeAbortController) {
      state.resummarizeAbortController.abort();
    }
    if (state.quizAbortController) {
      state.quizAbortController.abort();
    }
    if (state.quizGradeAbortController) {
      state.quizGradeAbortController.abort();
    }

    await stopBackendAIGeneration();
    await refreshSystemStatus();
    if (state.activeSessionId) {
      invalidateMemo("sources-structure");
      invalidateMemo("summary-pane");
      invalidateMemo("podcast-pane");
      try {
        const fresh = await api(`/api/sessions/${state.activeSessionId}`);
        state.activeSession = fresh;
        renderActiveSession(fresh);
      } catch (_) {}
    }
    if (!options.silentToast) {
      showToast("Generación de IA detenida.");
    }
  } finally {
    state.stoppingAI = false;
    syncGlobalStopButton();
  }
}

async function stopActiveChatGeneration() {
  if (state.chatAbortController) {
    state.chatAbortController.abort();
  }
  await stopBackendAIGeneration();
}

async function stopResummarizeGeneration() {
  if (state.resummarizeAbortController) {
    state.resummarizeAbortController.abort();
  }
  await stopBackendAIGeneration();
}

async function stopQuizGeneration() {
  if (state.quizAbortController) {
    state.quizAbortController.abort();
  }
  await stopBackendAIGeneration();
}

async function stopQuizGrading() {
  if (state.quizGradeAbortController) {
    state.quizGradeAbortController.abort();
  }
  await stopBackendAIGeneration();
}

function setChatSendButtonMode(streaming) {
  const btn = document.getElementById("btn-send-chat");
  if (!btn) return;
  if (streaming) {
    btn.type = "button";
    btn.disabled = false;
    btn.classList.add("is-stop-btn");
    btn.innerHTML = ICON_STOP_SQUARE;
    btn.title = "Detener respuesta (forzar paro)";
  } else {
    btn.type = "submit";
    btn.classList.remove("is-stop-btn");
    btn.innerHTML = ICON_SEND_ARROW;
    btn.title = "Enviar pregunta (Enter)";
    btn.disabled = Boolean(state.podcastBusy);
  }
}

// --- Telemetry, Context Health & Hero Status ---

async function refreshSystemStatus(options = {}) {
  try {
    const data = await api("/api/system/status");
    state.systemStatus = data;
    renderTelemetry(data);
    syncCaptureModeBar();
    if (data.settings?.theme && !localStorage.getItem("transcript_theme")) {
      applyTheme(data.settings.theme, false);
    }
    if (data.settings && !state.appearancePreviewBackup) {
      const hasLocalAccent = Boolean(localStorage.getItem("transcript_accent_color"));
      const hasLocalScale = Boolean(localStorage.getItem("transcript_font_scale"));
      applyAppearance(
        {
          accentColor: hasLocalAccent ? state.accentColor : (data.settings.accent_color || "#c8ee26"),
          fontScale: hasLocalScale ? state.fontScale : (data.settings.font_scale || 1.0),
        },
        false
      );
    }
    if (options.syncSettingsModal && isModalOpen("settings-modal") && !activeDropdownMenu) {
      populateSettingsFromStatus(data);
    }
  } catch (_) {}
}

function renderTelemetry(data) {
  const hw = data.hardware || {};
  const res = data.resources || {};
  const hasMetal = data.whisper_metal_supported;
  const settings = data.settings || {};
  const runningOllama = data.running_ollama_models || [];

  const gpuLabel = document.getElementById("gpu-label");
  const vramText = document.getElementById("vram-text");
  const vramBar = document.getElementById("vram-bar");
  const gpuTemp = document.getElementById("gpu-temp-text");

  if (hw.gpu_available) {
    gpuLabel.textContent = `${(hw.gpu_name || "GPU").replace("NVIDIA GeForce ", "")} VRAM`;
    vramText.textContent = `${hw.vram_used_mb} / ${hw.vram_total_mb} MB`;
    vramBar.style.width = `${Math.min(100, hw.vram_percent || 0)}%`;
    gpuTemp.textContent = `GPU ${hw.gpu_temp_c ?? "--"}°C · Uso ${hw.gpu_util_percent ?? 0}%`;
  } else if (hasMetal) {
    gpuLabel.textContent = "GPU Apple";
    vramText.textContent = "Memoria unificada";
    vramBar.style.width = "0%";
    gpuTemp.textContent = "Metal";
  } else {
    gpuLabel.textContent = "GPU";
    vramText.textContent = "Modo CPU";
    vramBar.style.width = "0%";
    gpuTemp.textContent = "CPU Local";
  }

  document.getElementById("ram-text").textContent = `${hw.ram_used_gb} / ${hw.ram_total_gb} GB`;
  document.getElementById("ram-bar").style.width = `${Math.min(100, hw.ram_percent || 0)}%`;
  const activeLabel = getActiveModelLabel();
  document.getElementById("active-model-name").textContent = res.active_model || runningOllama[0]?.name || "—";

  const pill = document.getElementById("model-state-pill");
  const pillText = document.getElementById("model-state-text");
  const vramChip = document.getElementById("topbar-vram-pill");
  const telTrigger = document.getElementById("btn-telemetry-popover");

  const vramPct = Math.round(hw.vram_percent || 0);
  if (vramChip) {
    vramChip.textContent = hw.gpu_available ? `VRAM ${vramPct}%` : hasMetal ? "Metal" : "CPU";
  }

  if (res.active_operation?.kind === "audio" || res.phase === "whisper") {
    pill.className = "mini-status active";
    const engine = data.whisper_models?.find((model) => model.id === res.active_model)?.engine || "Whisper";
    pillText.textContent = `Audio · ${engine}`;
  } else if (res.phase === "llm") {
    pill.className = "mini-status active";
    pillText.textContent = `Texto · ${res.active_operation?.label || res.active_model || activeLabel}`;
  } else if (res.phase === "podcast" || res.podcast_active) {
    pill.className = "mini-status active";
    const podPct = Math.max(8, Math.round((res.podcast_progress || 0) * 100));
    pillText.textContent = `Podcast ${podPct}%`;
  } else if (res.phase === "unloading") {
    pill.className = "mini-status active";
    pillText.textContent = "Liberando...";
  } else if (runningOllama.length > 0) {
    pill.className = "mini-status active";
    pillText.textContent = hasMetal ? "Modelo cargado" : "En VRAM";
  } else {
    pill.className = "mini-status idle";
    pillText.textContent = hasMetal ? "Listo" : "VRAM libre";
  }

  if (telTrigger) {
    telTrigger.classList.toggle("podcast-live", Boolean(res.podcast_active || res.phase === "podcast"));
  }

  syncPodcastBusyUI(Boolean(res.podcast_active || res.phase === "podcast"));
  syncGlobalStopButton();
}

function formatElapsedClock(ms) {
  const sec = Math.max(0, Math.floor((Number(ms) || 0) / 1000));
  const mm = String(Math.floor(sec / 60)).padStart(2, "0");
  const ss = String(sec % 60).padStart(2, "0");
  return `${mm}:${ss}`;
}

function applyPodcastLockToDOM(isBusy) {
  const lockHint = "En pausa mientras se genera el podcast en modo exclusivo";

  document.querySelectorAll(".seg-btn").forEach((btn) => {
    const lockTab = isBusy && btn.dataset.tab !== "podcast";
    btn.disabled = lockTab;
    btn.classList.toggle("is-podcast-locked", lockTab);
    btn.title = lockTab ? "Pestaña bloqueada mientras se genera el podcast para evitar saturar tu equipo" : "";
  });

  document.querySelectorAll(".session-item").forEach((item) => {
    const lockSess = isBusy && !item.classList.contains("active");
    item.classList.toggle("is-podcast-locked", lockSess);
    item.title = lockSess ? lockHint : "";
  });

  const idsToDisable = [
    "btn-new-session",
    "btn-empty-new",
    "session-search",
    "btn-open-settings",
    "btn-unload-vram",
    "chat-input",
    "btn-quick-attach",
    "btn-rename-session",
  ];
  for (const id of idsToDisable) {
    const el = document.getElementById(id);
    if (el) {
      el.disabled = isBusy;
      if (id !== "chat-input" && id !== "session-search") {
        el.title = isBusy ? lockHint : el.dataset.origTitle || el.title || "";
      }
    }
  }

  const sendBtn = document.getElementById("btn-send-chat");
  const resummBtn = document.getElementById("btn-resummarize");
  const genQuizBtn = document.querySelector("#tab-quiz #btn-generate-quiz");
  const uploadCta = document.querySelector('label[for="audio-file-input"]');
  const dropArea = document.getElementById("upload-drop-area");
  const telTrigger = document.getElementById("btn-telemetry-popover");

  if (sendBtn && !state.isStreaming) sendBtn.disabled = isBusy;
  if (resummBtn && !state.isResummarizing) resummBtn.disabled = isBusy;
  if (genQuizBtn) genQuizBtn.disabled = isBusy;
  if (uploadCta) {
    uploadCta.classList.toggle("is-podcast-locked", isBusy);
    uploadCta.title = isBusy ? lockHint : "";
  }
  if (dropArea) {
    dropArea.classList.toggle("is-podcast-locked", isBusy);
  }
  syncMicrophoneControls();
  if (telTrigger) {
    telTrigger.classList.toggle("podcast-live", isBusy);
  }

  document.querySelectorAll("#recordings-list .source-check, #recordings-list .retranscribe-btn, #recordings-list .rec-remove-btn").forEach((el) => {
    el.disabled = isBusy;
    el.classList.toggle("is-podcast-locked", isBusy);
  });
}

function updatePodcastElapsedDOM() {
  const el = document.querySelector("#tab-podcast .podcast-elapsed");
  if (!el || !state.podcastStartedAt) return;
  el.textContent = formatElapsedClock(Date.now() - state.podcastStartedAt);
}

function syncPodcastBusyUI(isBusy) {
  const prevBusy = state.podcastBusy;
  state.podcastBusy = isBusy;

  if (isBusy) {
    if (!state.podcastStartedAt) {
      state.podcastStartedAt = Date.now();
    }
    if (!state.podcastElapsedTimer) {
      state.podcastElapsedTimer = setInterval(updatePodcastElapsedDOM, 1000);
    }
    if (state.activeTab !== "podcast") {
      switchTab("podcast", { force: true });
    }
  } else {
    if (state.podcastElapsedTimer) {
      clearInterval(state.podcastElapsedTimer);
      state.podcastElapsedTimer = null;
    }
    state.podcastStartedAt = null;
  }

  applyPodcastLockToDOM(isBusy);

  if (prevBusy !== isBusy && state.activeSession && state.activeTab === "podcast") {
    renderPodcastTab(state.activeSession);
  }
}

function renderContextMetrics(metrics) {
  const ctxText = document.getElementById("ctx-text");
  const ctxBar = document.getElementById("ctx-bar");
  const ctxBadge = document.getElementById("ctx-badge");
  const ctxRow = document.getElementById("ctx-row");

  if (!metrics) {
    ctxText.textContent = "0%";
    ctxBar.style.width = "0%";
    ctxBar.className = "tel-fill lime";
    ctxBadge.className = "ctx-badge optimal";
    ctxBadge.textContent = "Óptimo";
    return;
  }

  const pct = Math.min(100, Math.round(metrics.usage_percent || 0));
  ctxText.textContent = `${pct}% (${metrics.estimated_tokens}/${metrics.num_ctx})`;
  ctxBar.style.width = `${Math.max(4, pct)}%`;

  if (metrics.health === "protected" || metrics.mode === "retrieval") {
    ctxBadge.className = "ctx-badge protected";
    ctxBadge.textContent = "RAG Protegido";
    ctxBar.className = "tel-fill protected";
    ctxRow.title = "Hay mucho audio adjunto. TransCrypting selecciona automáticamente resúmenes + fragmentos relevantes para evitar saturar la ventana de contexto.";
  } else if (metrics.health === "high") {
    ctxBadge.className = "ctx-badge high";
    ctxBadge.textContent = "Alto";
    ctxBar.className = "tel-fill amber";
    ctxRow.title = "El uso de contexto es alto. Puedes usar 'Borrar historial' en el chat para liberar tokens sin perder tus grabaciones.";
  } else {
    ctxBadge.className = "ctx-badge optimal";
    ctxBadge.textContent = "Óptimo";
    ctxBar.className = "tel-fill lime";
    ctxRow.title = "Toda la transcripción cabe holgadamente en la ventana del modelo.";
  }
}

// --- Tabs & Sliding Indicator ---

function updateTabIndicator() {
  const activeBtn = document.querySelector(".seg-btn.active");
  const indicator = document.getElementById("seg-indicator");
  if (!activeBtn || !indicator) return;
  indicator.style.width = `${activeBtn.offsetWidth}px`;
  indicator.style.transform = `translateX(${activeBtn.offsetLeft}px)`;
}

function switchTab(tabName, options = {}) {
  if (state.podcastBusy && !options.force && tabName !== "podcast") {
    showToast("Las pestañas están bloqueadas mientras se genera el podcast para evitar saturar el equipo.", true);
    return;
  }
  const prevIdx = TAB_ORDER.indexOf(state.activeTab);
  const nextIdx = TAB_ORDER.indexOf(tabName);
  const dirClass = nextIdx > prevIdx ? "slide-from-right" : nextIdx < prevIdx ? "slide-from-left" : "";

  state.activeTab = tabName;
  document.querySelectorAll(".seg-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === tabName);
  });

  for (const name of TAB_ORDER) {
    const pane = document.getElementById(`tab-${name}`);
    pane.classList.remove("slide-from-right", "slide-from-left");
    if (name === tabName) {
      pane.classList.remove("hidden");
      if (dirClass) {
        void pane.offsetWidth;
        pane.classList.add(dirClass);
      }
    } else {
      pane.classList.add("hidden");
    }
  }

  document.getElementById("actions-summary").classList.toggle("hidden", tabName !== "summary");
  document.getElementById("actions-transcript").classList.toggle("hidden", tabName !== "transcript");
  document.getElementById("actions-chat").classList.toggle("hidden", tabName !== "chat");
  document.getElementById("actions-quiz").classList.toggle("hidden", tabName !== "quiz");
  document.getElementById("actions-podcast").classList.toggle("hidden", tabName !== "podcast");
  const composerDock = document.getElementById("composer-dock");
  if (composerDock) {
    composerDock.classList.toggle("hidden", tabName !== "chat");
  }

  requestAnimationFrame(updateTabIndicator);

  if (tabName === "chat") {
    const scroll = document.getElementById("center-scroll");
    scroll.scrollTop = scroll.scrollHeight;
    if (state.activeSession) {
      ensureDynamicSuggestions(state.activeSession);
    }
  } else if (tabName === "transcript" && transcriptPlayback) {
    syncTranscriptPlayback(audioPlayers.get(transcriptPlayback.recId), true);
  } else if (tabName === "quiz" && state.activeSession) {
    renderQuizTab(state.activeSession);
  } else if (tabName === "podcast" && state.activeSession) {
    renderPodcastTab(state.activeSession);
  }
}

// --- Sessions ---

async function loadSessions(selectFirstIfNone = true) {
  state.sessions = await api("/api/sessions");
  renderSessionList();

  if (state.activeSessionId) {
    const exists = state.sessions.some((s) => s.id === state.activeSessionId);
    if (!exists) state.activeSessionId = null;
  }

  if (!state.activeSessionId && selectFirstIfNone && state.sessions.length > 0) {
    await selectSession(state.sessions[0].id);
  } else if (!state.activeSessionId) {
    renderActiveSession(null);
  }
}

function renderSessionList() {
  const container = document.getElementById("session-list");
  const query = (document.getElementById("session-search").value || "").trim().toLowerCase();
  const sig = `${state.activeSessionId || ""}|${query}|${state.sessions
    .map((s) => `${s.id}:${s.title}:${s.recording_count || 0}:${s.source_count || 0}:${s.message_count || 0}`)
    .join("|")}`;
  if (!memoChanged("session-list", sig)) return;

  container.innerHTML = "";

  const filtered = state.sessions.filter((s) =>
    (s.title || "").toLowerCase().includes(query)
  );

  if (filtered.length === 0) {
    const empty = document.createElement("div");
    empty.className = "rec-empty";
    empty.textContent = state.sessions.length === 0
      ? "Aún no hay conversaciones."
      : "Sin coincidencias.";
    container.appendChild(empty);
    return;
  }

  for (const s of filtered) {
    const item = document.createElement("div");
    item.className = `session-item ${s.id === state.activeSessionId ? "active" : ""}`;
    item.onclick = () => selectSession(s.id);

    const info = document.createElement("div");
    info.className = "session-item-info";

    const title = document.createElement("div");
    title.className = "session-item-title";
    title.textContent = s.title;

    const meta = document.createElement("div");
    meta.className = "session-item-meta";
    const totalSources = (s.recording_count || 0) + (s.source_count || 0);
    meta.textContent = `${totalSources} fuente(s) · ${s.message_count || 0} msg`;

    info.appendChild(title);
    info.appendChild(meta);

    const right = document.createElement("div");
    right.className = "session-item-right";

    const delBtn = document.createElement("button");
    delBtn.className = "session-item-delete";
    delBtn.title = "Eliminar conversación completa";
    delBtn.innerHTML = ICON_TRASH;
    delBtn.onclick = (e) => {
      e.stopPropagation();
      openConfirm(
        "Eliminar conversación",
        `¿Eliminar "${s.title}" junto con sus grabaciones, transcripciones y chat?`,
        async () => {
          await api(`/api/sessions/${s.id}`, { method: "DELETE" });
          if (state.activeSessionId === s.id) {
            state.activeSessionId = null;
            state.activeSession = null;
            state.activeRecordingId = null;
          }
          await loadSessions(true);
          showToast("Conversación eliminada.");
        }
      );
    };

    const dot = document.createElement("span");
    dot.className = "session-active-dot";

    right.appendChild(delBtn);
    right.appendChild(dot);

    item.appendChild(info);
    item.appendChild(right);
    container.appendChild(item);
  }
  if (state.podcastBusy) {
    applyPodcastLockToDOM(true);
  }
}

async function createNewSession() {
  if (state.podcastBusy) {
    showToast("Espera a que termine la producción del podcast antes de crear otra conversación.", true);
    return;
  }
  const created = await api("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: "Nueva conversación" }),
  });
  await loadSessions(false);
  await selectSession(created.id);
}

function stopAllOtherAudio(exceptId = null) {
  for (const [id, audio] of audioPlayers.entries()) {
    if (id !== exceptId && !audio.paused) {
      audio.pause();
    }
  }
  for (const [id, audio] of podcastAudioPlayers.entries()) {
    if (id !== exceptId && !audio.paused) {
      audio.pause();
    }
  }
}

async function selectSession(sessionId) {
  if (state.podcastBusy && state.activeSessionId && sessionId !== state.activeSessionId) {
    showToast("Espera a que termine el podcast actual antes de cambiar de conversación.", true);
    return;
  }
  if (state.activeSessionId !== sessionId) {
    if (state.activeSessionId) {
      stopQuizTimer(getQuizState(state.activeSessionId));
    }
    stopAllOtherAudio(null);
    state.activeRecordingId = null;
    state.transcriptRecFilter = null;
    state.transcriptSpeakerFilter = null;
  }
  state.activeSessionId = sessionId;
  renderSessionList();
  const detail = await api(`/api/sessions/${sessionId}`);
  state.activeSession = detail;
  if (!state.activeRecordingId && detail.recordings?.length) {
    state.activeRecordingId = detail.recordings[0].id;
  }
  renderActiveSession(detail);
  schedulePollingIfProcessing();
}

function schedulePollingIfProcessing() {
  if (state.pollTimer) {
    clearTimeout(state.pollTimer);
    state.pollTimer = null;
  }
  const session = state.activeSession;
  if (!session) return;

  const recProcessing = (session.recordings || []).some((r) =>
    ["queued", "preparing", "transcribing", "summarizing"].includes(r.status)
  );
  const srcProcessing = (session.sources || []).some((s) =>
    ["queued", "extracting", "indexing", "summarizing"].includes(s.status)
  );
  const podProcessing = (session.podcasts || []).some((p) =>
    ["queued", "scripting", "synthesizing", "mixing"].includes(p.status)
  );
  if (recProcessing || srcProcessing || podProcessing || state.podcastBusy) {
    state.pollTimer = setTimeout(async () => {
      if (state.activeSessionId === session.id) {
        const prevReady =
          (session.recordings || []).filter((r) => r.status === "ready").length +
          (session.sources || []).filter((s) => s.status === "ready").length +
          (session.podcasts || []).filter((p) => p.status === "ready").length;
        const fresh = await api(`/api/sessions/${session.id}`);
        state.activeSession = fresh;
        renderRecordings(fresh.recordings || [], fresh.sources || []);
        renderStatsTiles(fresh.recordings || [], fresh.sources || []);
        renderContextMetrics(fresh.context_metrics);
        if (!state.isResummarizing) {
          renderSummaryTab(fresh);
        }
        renderTranscriptTab(fresh);
        if (!getQuizState(fresh.id).questions) {
          renderQuizTab(fresh);
        }
        renderPodcastTab(fresh);
        renderWorkspaceSubtitle(fresh);
        updateChatControls(fresh);

        const nextReady =
          (fresh.recordings || []).filter((r) => r.status === "ready").length +
          (fresh.sources || []).filter((s) => s.status === "ready").length +
          (fresh.podcasts || []).filter((p) => p.status === "ready").length;
        if (nextReady !== prevReady) {
          const prevPodsReady = (session.podcasts || []).filter((p) => p.status === "ready").length;
          const nextPodsReady = (fresh.podcasts || []).filter((p) => p.status === "ready").length;
          if (nextPodsReady > prevPodsReady) {
            showToast("¡Podcast listo! Ya puedes escucharlo en Podcast Studio.");
          }
          loadSessions(false).catch(() => {});
        }
        schedulePollingIfProcessing();
      }
    }, 1500);
  }
}

function renderActiveSession(session) {
  const emptyEl = document.getElementById("empty-session-state");
  const activeEl = document.getElementById("active-session-view");

  if (!session) {
    emptyEl.classList.remove("hidden");
    activeEl.classList.add("hidden");
    invalidateMemo();
    renderStatsTiles([], []);
    renderRecordings([], []);
    renderContextMetrics(null);
    return;
  }

  emptyEl.classList.add("hidden");
  activeEl.classList.remove("hidden");

  document.getElementById("session-title").textContent = session.title;
  renderRecordings(session.recordings || [], session.sources || []);
  renderStatsTiles(session.recordings || [], session.sources || []);
  renderContextMetrics(session.context_metrics);
  if (!state.isResummarizing) {
    renderSummaryTab(session);
  }
  renderTranscriptTab(session);
  renderQuizTab(session);
  renderPodcastTab(session);
  renderWorkspaceSubtitle(session);
  renderChatMessages(session.messages || []);
  updateChatControls(session);
  syncGlobalStopButton();
  requestAnimationFrame(updateTabIndicator);
}

function renderStatsTiles(recordings, sources = []) {
  const count = recordings.length + sources.length;
  const activeCount =
    recordings.filter((r) => r.enabled !== false && r.status === "ready").length +
    sources.filter((s) => s.enabled !== false && s.status === "ready").length;
  const totalSec = recordings.reduce((acc, r) => acc + (Number(r.duration_sec) || 0), 0);
  document.getElementById("stat-audio-count").textContent = String(count);
  document.getElementById("stat-audio-duration").textContent =
    recordings.length > 0 && totalSec > 0
      ? `${activeCount} activas · ${formatDuration(totalSec)}`
      : `${activeCount} activa(s)`;
}

function renderWorkspaceSubtitle(session) {
  const sub = document.getElementById("session-subtitle");
  const recs = session.recordings || [];
  const srcs = session.sources || [];
  const activeRecs = recs.filter((r) => r.status === "ready" && r.enabled !== false).length;
  const activeSrcs = srcs.filter((s) => s.status === "ready" && s.enabled !== false).length;
  const totalCount = recs.length + srcs.length;
  const modelName = getActiveModelLabel();
  const mode = session.context_metrics?.mode || state.lastContextMode;
  const modeLabel = mode === "retrieval" ? "RAG Protegido (FTS5)" : "Contexto completo";

  if (totalCount === 0) {
    sub.textContent = "Adjunta audios, PDFs, diapositivas o imágenes para comenzar";
  } else if (activeRecs + activeSrcs === 0) {
    const anyReady =
      recs.some((r) => r.status === "ready") || srcs.some((s) => s.status === "ready");
    sub.textContent = anyReady
      ? "Todas las fuentes están desactivadas · Activa al menos una en Fuentes"
      : `Procesando ${totalCount} fuente(s)...`;
  } else {
    sub.textContent = `${modelName} · ${activeRecs + activeSrcs} fuente(s) activa(s) · ${modeLabel}`;
  }
}

// --- Custom Audio Player & Recordings ---

function getOrCreateAudio(rec) {
  let audio = audioPlayers.get(rec.id);
  if (!audio) {
    audio = new Audio(`/api/recordings/${rec.id}/audio`);
    audio.preload = "metadata";
    audio.dataset.recId = rec.id;

    const syncPlayback = () => {
      syncPlayerCardDOM(rec.id);
      syncTranscriptPlayback(audio);
    };
    let playbackFrame = 0;
    const followWords = () => {
      playbackFrame = 0;
      if (audio.paused || audio.ended) return;
      syncTranscriptPlayback(audio);
      playbackFrame = requestAnimationFrame(followWords);
    };
    audio.addEventListener("timeupdate", syncPlayback);
    audio.addEventListener("play", () => {
      syncPlayback();
      if (!playbackFrame) playbackFrame = requestAnimationFrame(followWords);
    });
    audio.addEventListener("seeked", syncPlayback);
    audio.addEventListener("pause", syncPlayback);
    audio.addEventListener("ended", syncPlayback);
    audio.addEventListener("loadedmetadata", () => syncPlayerCardDOM(rec.id));

    audioPlayers.set(rec.id, audio);
  }
  return audio;
}

function syncPlayerCardDOM(recId) {
  const card = document.querySelector(`.rec-card[data-rec-id="${recId}"]`);
  const audio = audioPlayers.get(recId);
  if (!card || !audio) return;

  const playBtn = card.querySelector(".play-circle");
  if (playBtn) {
    playBtn.innerHTML = audio.paused ? ICON_PLAY : ICON_PAUSE;
  }

  const eq = card.querySelector(".eq-bars");
  if (eq) {
    eq.classList.toggle("playing", !audio.paused);
  }

  const dur = Number.isFinite(audio.duration) && audio.duration > 0
    ? audio.duration
    : Number(card.dataset.durationSec || 0);
  const cur = audio.currentTime || 0;
  const pct = dur > 0 ? Math.min(100, (cur / dur) * 100) : 0;

  const fill = card.querySelector(".scrubber-fill");
  if (fill) fill.style.width = `${pct}%`;

  const timeEl = card.querySelector(".rec-time");
  if (timeEl) {
    timeEl.textContent = `${formatDuration(cur)} / ${formatDuration(dur)}`;
  }

  const speedBtn = card.querySelector(".speed-btn");
  if (speedBtn) {
    speedBtn.textContent = `${audio.playbackRate || 1}x`;
  }
}

function syncTranscriptPlayback(audio, reveal = false) {
  const playback = transcriptPlayback;
  if (!audio || !playback || playback.recId !== audio.dataset.recId) return;

  const time = audio.currentTime;
  const segment = playback.rows.find((item) => time >= item.start && time < item.end);
  const row = segment?.element || null;
  const word = segment?.words.find((item) => time >= item.start && time < item.end)?.element || null;
  const rowChanged = row !== playback.currentRow;
  if (!rowChanged && word === playback.currentWord && !reveal) return;
  if (rowChanged) {
    playback.currentRow?.removeAttribute("aria-current");
    row?.setAttribute("aria-current", "true");
    playback.currentRow = row;
  }
  playback.currentWord?.classList.remove("is-speaking");
  word?.classList.add("is-speaking");
  playback.currentWord = word;
  const target = word || ((rowChanged || reveal) ? row : null);
  if (!target || state.activeTab !== "transcript") return;

  const scroll = document.getElementById("center-scroll");
  const viewport = scroll.getBoundingClientRect();
  const line = target.getBoundingClientRect();
  if (line.top < viewport.top || line.bottom > viewport.bottom) {
    scroll.scrollTop += line.top - viewport.top - scroll.clientHeight / 3;
  }
}

function showRecordingTranscript(recId) {
  const session = state.activeSession;
  if (!session || session.id !== state.activeSessionId || state.podcastBusy) return;

  if (state.transcriptRecFilter !== recId) {
    document.getElementById("center-scroll").scrollTop = 0;
  }
  state.transcriptRecFilter = recId;
  document.getElementById("transcript-search").value = "";
  renderTranscriptTab(session);
  if (state.activeTab === "transcript") {
    syncTranscriptPlayback(audioPlayers.get(recId), true);
  } else {
    switchTab("transcript");
  }
}

async function setSourceEnabledState(kind, itemId, enabled) {
  if (!state.activeSessionId) return;
  const endpoint =
    kind === "source"
      ? `/api/sessions/${state.activeSessionId}/sources/${itemId}`
      : `/api/sessions/${state.activeSessionId}/recordings/${itemId}`;
  try {
    const fresh = await api(endpoint, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled }),
    });
    state.activeSession = fresh;
    renderActiveSession(fresh);
  } catch (err) {
    showToast(err.message, true);
  }
}

function openSourceViewer(src, locator = "") {
  if (!src) return;
  const titleEl = document.getElementById("source-viewer-title");
  const metaEl = document.getElementById("source-viewer-meta");
  const imgWrap = document.getElementById("source-viewer-image-wrap");
  const imgEl = document.getElementById("source-viewer-img");
  const pagesBar = document.getElementById("source-viewer-pages");
  const bodyEl = document.getElementById("source-viewer-text");

  titleEl.textContent = src.filename || "Vista de fuente";
  const typeLabel = String(src.source_type || "doc").toUpperCase();
  const pageCount = Number(src.page_count || (src.pages || []).length || 1);
  metaEl.textContent = src.source_type === "image" ? "IMAGEN INDEXADA" : `${typeLabel} · ${pageCount} sección(es)`;

  if (src.source_type === "image") {
    imgWrap.classList.remove("hidden");
    imgEl.src = `/api/sources/${src.id}/file`;
  } else {
    imgWrap.classList.add("hidden");
    imgEl.removeAttribute("src");
  }

  const pages = Array.isArray(src.pages) && src.pages.length > 0
    ? src.pages
    : [{ locator: "", label: "Contenido", text: src.extracted_text || src.summary_text || "Sin texto extraído." }];

  let activeIdx = 0;
  if (locator) {
    const cleanLoc = locator.trim().toLowerCase();
    const foundIdx = pages.findIndex(
      (p) => (p.locator || "").toLowerCase() === cleanLoc || (p.label || "").toLowerCase() === cleanLoc
    );
    if (foundIdx >= 0) activeIdx = foundIdx;
  }

  const renderPageContent = (idx) => {
    const page = pages[idx] || pages[0];
    bodyEl.textContent = page?.text || src.extracted_text || "Sin contenido disponible.";
    bodyEl.scrollTop = 0;
    pagesBar.querySelectorAll(".source-page-pill").forEach((btn, i) => {
      btn.classList.toggle("active", i === idx);
    });
  };

  pagesBar.innerHTML = "";
  if (pages.length > 1) {
    pagesBar.classList.remove("hidden");
    pages.forEach((p, idx) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `source-page-pill ${idx === activeIdx ? "active" : ""}`;
      btn.textContent = p.locator || p.label || `Sección ${idx + 1}`;
      btn.onclick = () => renderPageContent(idx);
      pagesBar.appendChild(btn);
    });
  } else {
    pagesBar.classList.add("hidden");
  }

  renderPageContent(activeIdx);
  openModal("source-viewer-modal");
}

function renderRecordings(recordings, sources = state.activeSession?.sources || []) {
  const list = document.getElementById("recordings-list");
  const hasRecs = Boolean(recordings && recordings.length > 0);
  const hasSrcs = Boolean(sources && sources.length > 0);

  if (hasRecs && !recordings.some((r) => r.id === state.activeRecordingId)) {
    state.activeRecordingId = recordings[0].id;
  }

  const structSig = [
    state.activeSessionId || "",
    state.activeRecordingId || "",
    (recordings || [])
      .map((r) => `${r.id}:${r.filename}:${r.status}:${r.enabled !== false}:${r.duration_sec || 0}:${r.language || ""}:${r.error_message || ""}`)
      .join(","),
    (sources || [])
      .map((s) => `${s.id}:${s.filename}:${s.status}:${s.enabled !== false}:${s.page_count || 0}:${s.error_message || ""}`)
      .join(","),
  ].join("|");

  if (!memoChanged("sources-structure", structSig)) {
    // React-like surgical patch: update only progress text/bars without recreating DOM nodes
    for (const rec of (recordings || [])) {
      const card = list.querySelector(`.rec-card[data-rec-id="${rec.id}"]`);
      if (!card) continue;
      const isWorking = ["queued", "preparing", "transcribing", "summarizing"].includes(rec.status);
      if (isWorking) {
        const pct = `${Math.round((rec.progress || 0) * 100)}%`;
        const pill = card.querySelector(".rec-status-pill");
        if (pill && pill.textContent !== pct) pill.textContent = pct;
        const stage = card.querySelector(".rec-stage-line");
        if (stage && stage.textContent !== (rec.stage || "Procesando...")) {
          stage.textContent = rec.stage || "Procesando...";
        }
        const fill = card.querySelector(".working-fill");
        if (fill) fill.style.width = `${Math.max(5, Math.round((rec.progress || 0) * 100))}%`;
      }
    }
    for (const src of (sources || [])) {
      const card = list.querySelector(`.source-card[data-source-id="${src.id}"]`);
      if (!card) continue;
      const isWorking = ["queued", "extracting", "indexing", "summarizing"].includes(src.status);
      if (isWorking) {
        const pct = `${Math.round((src.progress || 0) * 100)}%`;
        const pill = card.querySelector(".rec-status-pill");
        if (pill && pill.textContent !== pct) pill.textContent = pct;
        const stage = card.querySelector(".rec-stage-line");
        if (stage && stage.textContent !== (src.stage || "Indexando fuente...")) {
          stage.textContent = src.stage || "Indexando fuente...";
        }
        const fill = card.querySelector(".working-fill");
        if (fill) fill.style.width = `${Math.max(8, Math.round((src.progress || 0) * 100))}%`;
      }
    }
    return;
  }

  list.innerHTML = "";

  if (!hasRecs && !hasSrcs) {
    const empty = document.createElement("div");
    empty.className = "rec-empty";
    empty.textContent = "Aún no hay fuentes en esta conversación.";
    list.appendChild(empty);
    return;
  }

  for (const rec of (recordings || [])) {
    const audio = getOrCreateAudio(rec);
    const isActive = rec.id === state.activeRecordingId;
    const isEnabled = rec.enabled !== false;
    const isWorking = ["queued", "preparing", "transcribing", "summarizing"].includes(rec.status);

    const card = document.createElement("div");
    card.className = `rec-card ${isActive ? "active" : ""} ${!isEnabled ? "is-disabled" : ""}`;
    card.dataset.recId = rec.id;
    card.dataset.durationSec = String(rec.duration_sec || 0);

    card.onclick = () => {
      if (state.activeRecordingId !== rec.id) {
        state.activeRecordingId = rec.id;
        renderRecordings(recordings, sources);
      }
    };

    const top = document.createElement("div");
    top.className = "rec-top";

    const chk = document.createElement("input");
    chk.type = "checkbox";
    chk.className = "source-check";
    chk.checked = isEnabled;
    chk.title = isEnabled ? "Incluida en el contexto (clic para desactivar)" : "Desactivada del contexto (clic para activar)";
    chk.onclick = (e) => e.stopPropagation();
    chk.onchange = () => setSourceEnabledState("recording", rec.id, chk.checked);

    const playBtn = document.createElement("button");
    playBtn.type = "button";
    playBtn.className = "play-circle";
    playBtn.innerHTML = audio.paused ? ICON_PLAY : ICON_PAUSE;
    playBtn.onclick = (e) => {
      e.stopPropagation();
      state.activeRecordingId = rec.id;
      if (audio.paused) {
        stopAllOtherAudio(rec.id);
        audio.play().catch((err) => showToast(`No se pudo reproducir: ${err.message}`, true));
        showRecordingTranscript(rec.id);
      } else {
        audio.pause();
      }
      renderRecordings(recordings, sources);
    };

    const info = document.createElement("div");
    info.className = "rec-info";

    const nameRow = document.createElement("div");
    nameRow.className = "rec-name-row";

    const name = document.createElement("div");
    name.className = "rec-name";
    name.title = rec.filename;
    name.textContent = rec.filename;
    nameRow.appendChild(name);

    if (isActive && !isWorking) {
      const eq = document.createElement("span");
      eq.className = `eq-bars ${!audio.paused ? "playing" : ""}`;
      eq.innerHTML = "<span></span><span></span><span></span>";
      nameRow.appendChild(eq);
    }

    const meta = document.createElement("div");
    meta.className = "rec-meta-line";

    const dur = Number.isFinite(audio.duration) && audio.duration > 0
      ? audio.duration
      : (rec.duration_sec || 0);
    const timeSpan = document.createElement("span");
    timeSpan.className = "rec-time";
    timeSpan.textContent = `${formatDuration(audio.currentTime || 0)} / ${formatDuration(dur)}`;
    meta.appendChild(timeSpan);

    if (rec.language) {
      const lang = document.createElement("span");
      lang.textContent = rec.language.toUpperCase();
      meta.appendChild(lang);
    }

    const statusPill = document.createElement("span");
    if (rec.status === "ready") {
      statusPill.className = "rec-status-pill ready";
      statusPill.textContent = "Listo";
    } else if (rec.status === "error") {
      statusPill.className = "rec-status-pill error";
      statusPill.textContent = "Error";
    } else {
      statusPill.className = "rec-status-pill working";
      statusPill.textContent = `${Math.round((rec.progress || 0) * 100)}%`;
    }
    meta.appendChild(statusPill);

    info.appendChild(nameRow);

    const rightBtns = document.createElement("div");
    rightBtns.className = "rec-controls-right";

    if (isWorking) {
      const stopBtn = document.createElement("button");
      stopBtn.type = "button";
      stopBtn.className = "rec-stop-btn";
      stopBtn.title = "Detener procesamiento de IA";
      stopBtn.innerHTML = `<span class="stop-sq" aria-hidden="true"></span>`;
      stopBtn.onclick = (e) => {
        e.stopPropagation();
        forceStopAllAI();
      };
      rightBtns.appendChild(stopBtn);
    }

    const speedBtn = document.createElement("button");
    speedBtn.type = "button";
    speedBtn.className = "speed-btn";
    speedBtn.title = "Cambiar velocidad de reproducción";
    speedBtn.textContent = `${audio.playbackRate || 1}x`;
    speedBtn.onclick = (e) => {
      e.stopPropagation();
      const speeds = [1, 1.25, 1.5, 2];
      const next = speeds[(speeds.indexOf(audio.playbackRate) + 1) % speeds.length];
      audio.playbackRate = next;
      syncPlayerCardDOM(rec.id);
    };

    if (!isWorking) {
      const retransBtn = document.createElement("button");
      retransBtn.type = "button";
      retransBtn.className = "retranscribe-btn";
      retransBtn.title = "Re-transcribir con el modelo seleccionado en Configuración";
      retransBtn.innerHTML = ICON_RETRANSCRIBE;
      retransBtn.onclick = (e) => {
        e.stopPropagation();
        const model = state.systemStatus?.settings?.whisper_model || "large-v3-turbo";
        openConfirm(
          "Re-transcribir grabación",
          `¿Volver a transcribir "${rec.filename}" con ${model}? Se actualizarán su transcripción y resumen.`,
          async () => {
            await api(`/api/sessions/${rec.session_id}/recordings/${rec.id}/retranscribe`, {
              method: "POST",
            });
            await selectSession(rec.session_id);
            showToast("Re-transcripción iniciada en segundo plano.");
          }
        );
      };
      rightBtns.appendChild(retransBtn);
    }

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "rec-remove-btn";
    removeBtn.title = "Quitar grabación";
    removeBtn.textContent = "✕";
    removeBtn.onclick = (e) => {
      e.stopPropagation();
      openConfirm(
        "Eliminar grabación",
        `¿Quitar "${rec.filename}" de esta conversación?`,
        async () => {
          card.style.opacity = "0.35";
          card.style.pointerEvents = "none";
          try {
            audio.pause();
            audioPlayers.delete(rec.id);
            await api(`/api/sessions/${rec.session_id}/recordings/${rec.id}`, {
              method: "DELETE",
            });
            podcastFormState.get(rec.session_id)?.selectedRecIds.delete(rec.id);
            sessionQuizzes.get(rec.session_id)?.selectedRecIds.delete(rec.id);
            invalidateMemo("sources-structure");
            invalidateMemo("summary-pane");
            invalidateMemo("quiz-pane");
            invalidateMemo("podcast-pane");
            await selectSession(rec.session_id);
            await loadSessions(false);
            showToast("Grabación eliminada.");
          } catch (err) {
            card.style.opacity = "";
            card.style.pointerEvents = "";
            throw err;
          }
        }
      );
    };

    rightBtns.appendChild(speedBtn);
    rightBtns.appendChild(removeBtn);

    top.appendChild(chk);
    top.appendChild(playBtn);
    top.appendChild(info);
    card.appendChild(top);

    const footRow = document.createElement("div");
    footRow.className = "rec-foot-row";
    footRow.appendChild(meta);
    footRow.appendChild(rightBtns);
    card.appendChild(footRow);

    if (isWorking) {
      const stageLine = document.createElement("div");
      stageLine.className = "rec-stage-line";
      stageLine.style.fontSize = "11.5px";
      stageLine.style.opacity = "0.85";
      stageLine.textContent = rec.stage || "Procesando...";
      card.appendChild(stageLine);

      const bar = document.createElement("div");
      bar.className = "scrubber-track";
      const fill = document.createElement("div");
      fill.className = "scrubber-fill working-shimmer working-fill";
      fill.style.width = `${Math.max(5, Math.round((rec.progress || 0) * 100))}%`;
      bar.appendChild(fill);
      card.appendChild(bar);
    } else if (rec.status === "error" && rec.error_message) {
      const err = document.createElement("div");
      err.style.fontSize = "12px";
      err.style.color = "#fca5a5";
      err.textContent = rec.error_message;
      card.appendChild(err);
    } else {
      const scrubWrap = document.createElement("div");
      scrubWrap.className = "scrubber-wrap";
      const track = document.createElement("div");
      track.className = "scrubber-track";
      track.title = "Haz clic para saltar en el audio";
      const fill = document.createElement("div");
      fill.className = "scrubber-fill";
      const pct = dur > 0 ? Math.min(100, ((audio.currentTime || 0) / dur) * 100) : 0;
      fill.style.width = `${pct}%`;
      track.appendChild(fill);

      track.onclick = (e) => {
        e.stopPropagation();
        const rect = track.getBoundingClientRect();
        const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        const targetDur = Number.isFinite(audio.duration) && audio.duration > 0
          ? audio.duration
          : (rec.duration_sec || 0);
        if (targetDur > 0) {
          audio.currentTime = ratio * targetDur;
          syncPlayerCardDOM(rec.id);
        }
      };

      scrubWrap.appendChild(track);
      card.appendChild(scrubWrap);
    }

    list.appendChild(card);
  }

  // Document & Image Sources (PDF, PPTX, DOCX, TXT/MD, Images)
  for (const src of (sources || [])) {
    const isEnabled = src.enabled !== false;
    const isWorking = ["queued", "extracting", "indexing", "summarizing"].includes(src.status);
    const kind = (src.source_type || "doc").toLowerCase();

    const card = document.createElement("div");
    card.className = `source-card ${!isEnabled ? "is-disabled" : ""}`;
    card.dataset.sourceId = src.id;
    card.onclick = () => openSourceViewer(src);

    const top = document.createElement("div");
    top.className = "rec-top";

    const chk = document.createElement("input");
    chk.type = "checkbox";
    chk.className = "source-check";
    chk.checked = isEnabled;
    chk.title = isEnabled ? "Fuente activa en el contexto (clic para desactivar)" : "Fuente desactivada (clic para activar)";
    chk.onclick = (e) => e.stopPropagation();
    chk.onchange = () => setSourceEnabledState("source", src.id, chk.checked);

    let iconEl;
    if (kind === "image") {
      iconEl = document.createElement("img");
      iconEl.className = "source-thumb";
      iconEl.src = `/api/sources/${src.id}/file`;
      iconEl.alt = src.filename;
    } else {
      iconEl = document.createElement("div");
      iconEl.className = `source-type-badge kind-${kind}`;
      iconEl.textContent = kind === "text" ? "TXT" : kind.toUpperCase();
    }

    const info = document.createElement("div");
    info.className = "rec-info";

    const nameRow = document.createElement("div");
    nameRow.className = "rec-name-row";
    const name = document.createElement("div");
    name.className = "rec-name";
    name.title = `${src.filename} (clic para previsualizar)`;
    name.textContent = src.filename;
    nameRow.appendChild(name);

    const meta = document.createElement("div");
    meta.className = "rec-meta-line";
    const unitLabel =
      kind === "image"
        ? "Imagen"
        : kind === "pptx"
        ? `${src.page_count || 1} diap.`
        : `${src.page_count || 1} pág.`;
    const sizeKb = src.size_bytes > 0 ? `${Math.max(1, Math.round(src.size_bytes / 1024))} KB` : "";
    const metaSpan = document.createElement("span");
    metaSpan.textContent = sizeKb ? `${unitLabel} · ${sizeKb}` : unitLabel;
    meta.appendChild(metaSpan);

    const statusPill = document.createElement("span");
    if (src.status === "ready") {
      statusPill.className = "rec-status-pill ready";
      statusPill.textContent = "Indexado";
    } else if (src.status === "error") {
      statusPill.className = "rec-status-pill error";
      statusPill.textContent = "Error";
    } else {
      statusPill.className = "rec-status-pill working";
      statusPill.textContent = `${Math.round((src.progress || 0) * 100)}%`;
    }
    meta.appendChild(statusPill);

    info.appendChild(nameRow);

    const rightBtns = document.createElement("div");
    rightBtns.className = "rec-controls-right";

    if (isWorking) {
      const stopBtn = document.createElement("button");
      stopBtn.type = "button";
      stopBtn.className = "rec-stop-btn";
      stopBtn.title = "Detener procesamiento de IA";
      stopBtn.innerHTML = `<span class="stop-sq" aria-hidden="true"></span>`;
      stopBtn.onclick = (e) => {
        e.stopPropagation();
        forceStopAllAI();
      };
      rightBtns.appendChild(stopBtn);
    }

    if (!isWorking) {
      const reprocessBtn = document.createElement("button");
      reprocessBtn.type = "button";
      reprocessBtn.className = "retranscribe-btn";
      reprocessBtn.title = "Re-indexar y resumir fuente con la configuración actual";
      reprocessBtn.innerHTML = ICON_RETRANSCRIBE;
      reprocessBtn.onclick = (e) => {
        e.stopPropagation();
        openConfirm(
          "Re-indexar fuente",
          `¿Volver a extraer, indexar en FTS5 y resumir "${src.filename}" con el modelo actual?`,
          async () => {
            await api(`/api/sessions/${src.session_id}/sources/${src.id}/reprocess`, {
              method: "POST",
            });
            await selectSession(src.session_id);
            showToast("Re-indexación iniciada en segundo plano.");
          }
        );
      };
      rightBtns.appendChild(reprocessBtn);
    }

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "rec-remove-btn";
    removeBtn.title = "Quitar fuente";
    removeBtn.textContent = "✕";
    removeBtn.onclick = (e) => {
      e.stopPropagation();
      openConfirm(
        "Eliminar fuente",
        `¿Quitar "${src.filename}" de esta conversación y de su índice local?`,
        async () => {
          card.style.opacity = "0.35";
          card.style.pointerEvents = "none";
          try {
            await api(`/api/sessions/${src.session_id}/sources/${src.id}`, {
              method: "DELETE",
            });
            podcastFormState.get(src.session_id)?.selectedSrcIds.delete(src.id);
            sessionQuizzes.get(src.session_id)?.selectedSrcIds.delete(src.id);
            invalidateMemo("sources-structure");
            invalidateMemo("summary-pane");
            invalidateMemo("quiz-pane");
            invalidateMemo("podcast-pane");
            await selectSession(src.session_id);
            await loadSessions(false);
            showToast("Fuente eliminada.");
          } catch (err) {
            card.style.opacity = "";
            card.style.pointerEvents = "";
            throw err;
          }
        }
      );
    };
    rightBtns.appendChild(removeBtn);

    top.appendChild(chk);
    top.appendChild(iconEl);
    top.appendChild(info);
    card.appendChild(top);

    const footRow = document.createElement("div");
    footRow.className = "rec-foot-row";
    footRow.appendChild(meta);
    footRow.appendChild(rightBtns);
    card.appendChild(footRow);

    if (isWorking) {
      const stageLine = document.createElement("div");
      stageLine.className = "rec-stage-line";
      stageLine.style.fontSize = "11.5px";
      stageLine.style.opacity = "0.85";
      stageLine.textContent = src.stage || "Indexando fuente...";
      card.appendChild(stageLine);

      const bar = document.createElement("div");
      bar.className = "scrubber-track";
      const fill = document.createElement("div");
      fill.className = "scrubber-fill working-shimmer working-fill";
      fill.style.width = `${Math.max(8, Math.round((src.progress || 0) * 100))}%`;
      bar.appendChild(fill);
      card.appendChild(bar);
    } else if (src.status === "error" && src.error_message) {
      const err = document.createElement("div");
      err.style.fontSize = "12px";
      err.style.color = "#fca5a5";
      err.textContent = src.error_message;
      card.appendChild(err);
    }

    list.appendChild(card);
  }

  if (state.podcastBusy) {
    applyPodcastLockToDOM(true);
  }
}

function seekRecording(recId, seconds, options = {}) {
  const recordings = state.activeSession?.recordings || [];
  const rec = recordings.find((r) => r.id === recId);
  if (!rec) return;

  state.activeRecordingId = recId;
  const audio = getOrCreateAudio(rec);
  stopAllOtherAudio(recId);

  const highlightCard = () => {
    renderRecordings(recordings);
    if (options.pulse) {
      requestAnimationFrame(() => {
        const card = document.querySelector(`.rec-card[data-rec-id="${recId}"]`);
        if (card) {
          card.scrollIntoView({ behavior: "smooth", block: "nearest" });
          card.classList.remove("pulse-seek");
          void card.offsetWidth;
          card.classList.add("pulse-seek");
        }
      });
    }
  };

  const jump = () => {
    if (state.activeRecordingId !== recId || state.activeSessionId !== rec.session_id) return;
    audio.currentTime = Math.max(0, Number(seconds) || 0);
    audio.play().catch(() => {});
    if (state.activeTab !== "transcript" || state.transcriptRecFilter !== recId) {
      showRecordingTranscript(recId);
    }
    highlightCard();
  };

  if (audio.readyState >= 1) {
    jump();
  } else {
    audio.addEventListener("loadedmetadata", jump, { once: true });
    audio.load();
  }
}

function resolveRecordingForTimestamp(triggerEl, seconds) {
  const recordings = (state.activeSession?.recordings || []).filter(
    (r) => r.status === "ready" || r.duration_sec > 0 || r.transcript_text
  );
  if (recordings.length === 0) return null;

  const explicitRecId = triggerEl?.dataset?.recId;
  if (explicitRecId && recordings.some((r) => r.id === explicitRecId)) {
    return explicitRecId;
  }
  if (recordings.length === 1) return recordings[0].id;

  const contextContainer = triggerEl.closest("p, li, .msg-bubble, .content-card");
  const contextText = (contextContainer?.textContent || "").toLowerCase();
  const contextNorm = normalizeRefTokens(contextText);

  for (let i = 0; i < recordings.length; i += 1) {
    const rec = recordings[i];
    const recNorm = normalizeRefTokens(rec.filename);
    if (
      contextText.includes((rec.filename || "").toLowerCase()) ||
      (recNorm && recNorm.length >= 4 && contextNorm.includes(recNorm)) ||
      contextText.includes(`grabación ${i + 1}`) ||
      contextText.includes(`audio ${i + 1}`)
    ) {
      return rec.id;
    }
  }

  const activeRec = recordings.find((r) => r.id === state.activeRecordingId);
  if (activeRec && (!activeRec.duration_sec || seconds <= activeRec.duration_sec + 1)) {
    return activeRec.id;
  }

  const fitting = recordings.find((r) => r.duration_sec && seconds <= r.duration_sec + 1);
  return (fitting || recordings[0]).id;
}

// --- Summary Skeleton & Content Panes ---

function buildSkeletonHtml(statusLabel, showStopBtn = true, stopScope = "") {
  const stopBtnHtml = showStopBtn
    ? `<button type="button" class="skeleton-stop-btn"${stopScope ? ` data-stop-scope="${escapeHtml(stopScope)}"` : ""} title="Detener generación de IA"><span class="stop-sq" aria-hidden="true"></span> Detener</button>`
    : "";
  return `
    <div class="summary-skeleton-card">
      <div class="skeleton-banner">
        <div class="think-dots" aria-hidden="true"><span></span><span></span><span></span></div>
        <span class="think-label">${escapeHtml(statusLabel)}</span>
        ${stopBtnHtml}
      </div>
      <div class="skel-group">
        <div class="skel-line title"></div>
        <div class="skel-line w95"></div>
        <div class="skel-line w85"></div>
      </div>
      <div class="skel-group">
        <div class="skel-line title"></div>
        <div class="skel-bullet-row"><span class="skel-dot"></span><div class="skel-line w85"></div></div>
        <div class="skel-bullet-row"><span class="skel-dot"></span><div class="skel-line w75"></div></div>
        <div class="skel-bullet-row"><span class="skel-dot"></span><div class="skel-line w65"></div></div>
      </div>
    </div>
  `;
}

function renderSummaryTab(session, animateReveal = false) {
  const pane = document.getElementById("tab-summary");
  const recordings = session.recordings || [];
  const sources = session.sources || [];
  const allItems = [...recordings, ...sources];
  const readyRecs = recordings.filter((r) => r.status === "ready" && r.enabled !== false);
  const readySrcs = sources.filter((s) => s.status === "ready" && s.enabled !== false);
  const workingItem =
    recordings.find((r) => ["queued", "preparing", "transcribing", "summarizing"].includes(r.status)) ||
    sources.find((s) => ["queued", "extracting", "indexing", "summarizing"].includes(s.status));

  if (allItems.length === 0) {
    if (!animateReveal && !memoChanged("summary-pane", `${session.id}:empty`)) return;
    pane.innerHTML = `<div class="chat-placeholder"><h3>Sin fuentes aún</h3><p>Adjunta grabaciones, PDFs, diapositivas, apuntes o imágenes desde el panel de Fuentes o arrástralos aquí.</p></div>`;
    return;
  }

  if (readyRecs.length === 0 && readySrcs.length === 0) {
    if (workingItem) {
      const stageMsg = `${workingItem.stage || "Procesando"} (${Math.round((workingItem.progress || 0) * 100)}%)`;
      const existingLabel = pane.querySelector(".summary-skeleton-card .think-label");
      if (existingLabel && memoChanged("summary-pane", `${session.id}:skel:${stageMsg}`)) {
        existingLabel.textContent = stageMsg;
        return;
      }
      if (!memoChanged("summary-pane", `${session.id}:skel:${stageMsg}`)) return;
      pane.innerHTML = buildSkeletonHtml(stageMsg);
    } else {
      if (!animateReveal && !memoChanged("summary-pane", `${session.id}:all-disabled`)) return;
      pane.innerHTML = `<div class="chat-placeholder"><h3>Todas las fuentes están desactivadas</h3><p>Marca la casilla de al menos una fuente en el panel derecho para incluirla en el resumen y chat.</p></div>`;
    }
    return;
  }

  const summarySig = [
    session.id,
    session.combined_summary || "",
    readyRecs.map((r) => `${r.id}:${(r.summary_text || "").length}:${r.whisper_model || ""}`).join(","),
    readySrcs.map((s) => `${s.id}:${(s.summary_text || "").length}:${s.page_count || 0}`).join(","),
  ].join("|");
  if (!animateReveal && !memoChanged("summary-pane", summarySig)) return;
  if (animateReveal) renderMemo.set("summary-pane", summarySig);

  const revealClass = animateReveal ? "summary-reveal" : "";
  const totalActive = readyRecs.length + readySrcs.length;

  let summaryMarkdown = "";
  if (totalActive >= 2 && session.combined_summary) {
    summaryMarkdown = session.combined_summary;
  } else if (totalActive === 1) {
    summaryMarkdown = readyRecs[0]?.summary_text || readySrcs[0]?.summary_text || "";
  } else {
    const mergedParts = [
      ...readyRecs.map((r) => (r.summary_text ? `### ${r.filename}\n${r.summary_text}` : "")),
      ...readySrcs.map((s) => (s.summary_text ? `### ${s.filename}\n${s.summary_text}` : "")),
    ].filter(Boolean);
    summaryMarkdown = mergedParts.join("\n\n");
  }

  const chipsHtml = [
    ...readyRecs.map((r) => `<span class="summary-src-tag">🎙 ${escapeHtml(r.filename)}</span>`),
    ...readySrcs.map((s) => {
      const ico = s.source_type === "image" ? "🖼" : "📄";
      return `<button type="button" class="chat-src-pill" data-source-id="${escapeHtml(s.id)}" data-locator="" title="Abrir fuente">${ico} ${escapeHtml(s.filename)}</button>`;
    }),
  ].join("");

  const bodyHtml = summaryMarkdown
    ? `<div class="md">${renderMarkdown(summaryMarkdown, true)}</div>`
    : `<div class="chat-placeholder">
        <h3>Fuentes indexadas y listas</h3>
        <p>El chat y la búsqueda FTS5 ya están disponibles. Puedes generar el resumen general bajo demanda cuando lo necesites.</p>
        <button type="button" class="pill-btn dark sm eco-generate-summary-btn">Generar resumen ahora</button>
      </div>`;

  pane.innerHTML = `
    <div class="content-card combined ${revealClass}">
      <div class="content-card-header">
        <span>Resumen general de la sesión</span>
        <span class="card-badge lime">${totalActive} fuente(s) activa(s)</span>
      </div>
      ${chipsHtml ? `<div class="summary-sources-strip">${chipsHtml}</div>` : ""}
      ${bodyHtml}
    </div>
  `;

  const ecoBtn = pane.querySelector(".eco-generate-summary-btn");
  if (ecoBtn) {
    ecoBtn.onclick = () => document.getElementById("btn-resummarize")?.click();
  }
}

function getSpeakerToneClass(speaker, orderedSpeakers) {
  if (speaker === "Tú") return "spk-me";
  const others = orderedSpeakers.filter((s) => s !== "Tú");
  const idx = others.indexOf(speaker);
  const slot = (idx >= 0 ? idx : 0) % 6 + 1;
  return `spk-${slot}`;
}

function renderTranscriptTab(session) {
  syncMicrophonePreview(session);
  const pane = document.getElementById("transcript-content");
  const query = (document.getElementById("transcript-search").value || "").trim().toLowerCase();
  const activeRec = (session.recordings || []).find((r) => r.id === state.transcriptRecFilter);
  const recordings = (session.recordings || []).filter(
    (r) => r === activeRec || (r.transcript_text && r.enabled !== false)
  );
  const sources = (session.sources || []).filter((s) => s.extracted_text && s.enabled !== false);

  state.transcriptRecFilter = activeRec?.id || null;
  if (!activeRec) state.transcriptSpeakerFilter = null;

  const copyBtn = document.getElementById("btn-copy-transcript");
  if (copyBtn && !copyBtn.classList.contains("is-confirmed")) {
    copyBtn.textContent = state.transcriptRecFilter ? "Copiar audio" : "Copiar todo";
  }

  const activeSpkList = activeRec
    ? Array.from(new Set((activeRec.segments || []).map((s) => (s.speaker || "").trim()).filter(Boolean)))
    : [];
  if (state.transcriptSpeakerFilter && !activeSpkList.includes(state.transcriptSpeakerFilter)) {
    state.transcriptSpeakerFilter = null;
  }

  const contentSig = [
    session.id,
    query,
    state.transcriptRecFilter || "grid",
    state.transcriptSpeakerFilter || "",
    activeRec?.status || "",
    activeSpkList.join(","),
    recordings.map((r) => `${r.id}:${(r.transcript_text || "").length}:${(r.segments || []).length}:${r.whisper_model || ""}`).join(","),
    sources.map((s) => s.id).join(","),
  ].join("|");
  if (!memoChanged("content-pane", contentSig)) return;
  transcriptPlayback = null;

  if (recordings.length === 0) {
    if (sources.length > 0) {
      const srcPills = sources
        .map((s) => {
          const ico = s.source_type === "image" ? "🖼" : "📄";
          return `<button type="button" class="chat-src-pill" data-source-id="${escapeHtml(s.id)}" data-locator="">${ico} ${escapeHtml(s.filename)}</button>`;
        })
        .join("");
      pane.innerHTML = `
        <div class="chat-placeholder">
          <h3>Sin audios transcritos en esta sesión</h3>
          <p>Este apartado muestra las transcripciones exactas con marcas de tiempo de tus audios y clases grabadas. Tus documentos e imágenes siguen disponibles con un clic:</p>
          <div style="display:flex;flex-wrap:wrap;justify-content:center;gap:6px;margin-top:12px;">${srcPills}</div>
        </div>
      `;
    } else {
      pane.innerHTML = `<div class="chat-placeholder"><h3>Transcripción de audio vacía</h3><p>Adjunta o activa una grabación de audio en el panel de Fuentes para consultar aquí su texto literal y saltar a cualquier minuto.</p></div>`;
    }
    return;
  }

  pane.innerHTML = "";

  const searchPattern = query ? new RegExp(query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi") : null;
  const highlightMatch = (el, text) => {
    if (searchPattern) {
      const safe = escapeHtml(text);
      el.innerHTML = safe.replace(searchPattern, '<mark class="hl">$&</mark>');
    } else {
      el.textContent = text;
    }
  };

  // --- Cards Grid Overview (select an audio to read its full transcript) ---
  if (!state.transcriptRecFilter) {
    const galleryHead = document.createElement("div");
    galleryHead.className = "transcript-gallery-head";
    galleryHead.innerHTML = `
      <span class="transcript-gallery-title">Audios transcritos (${recordings.length})</span>
      <span class="transcript-gallery-hint">Haz clic en un audio para leer su transcripción completa</span>
    `;
    pane.appendChild(galleryHead);

    const grid = document.createElement("div");
    grid.className = "transcript-cards-grid";
    let visibleCards = 0;

    for (const rec of recordings) {
      const segs = rec.segments || [];
      const rawText = String(rec.transcript_text || "").trim();
      const matchingSegs = query
        ? segs.filter((s) => (s.text || "").toLowerCase().includes(query))
        : segs;
      const textMatched = query ? rawText.toLowerCase().includes(query) : true;
      const matchCount = query ? (segs.length > 0 ? matchingSegs.length : (textMatched ? 1 : 0)) : 0;

      if (query && matchCount === 0) continue;
      visibleCards += 1;

      const wordCount = rawText ? rawText.split(/\s+/).length : 0;
      const snippetSource =
        (query && matchingSegs[0]?.text) ||
        (segs[0]?.text && segs.map((s) => s.text).slice(0, 3).join(" ")) ||
        rawText;
      const snippet = snippetSource.length > 175
        ? `${snippetSource.slice(0, 175).trim()}…`
        : snippetSource;

      const card = document.createElement("div");
      card.className = "transcript-audio-card";
      card.tabIndex = 0;
      card.setAttribute("role", "button");
      card.setAttribute("aria-label", `Abrir transcripción de ${rec.filename}`);

      const top = document.createElement("div");
      top.className = "transcript-card-top";

      const titleWrap = document.createElement("div");
      titleWrap.className = "transcript-card-title-wrap";
      const ico = document.createElement("span");
      ico.className = "transcript-card-icon";
      ico.textContent = "🎙";
      const titleEl = document.createElement("span");
      titleEl.className = "transcript-card-title";
      titleEl.title = rec.filename;
      titleEl.textContent = rec.filename;
      titleWrap.appendChild(ico);
      titleWrap.appendChild(titleEl);

      const playBtn = document.createElement("button");
      playBtn.type = "button";
      playBtn.className = "transcript-card-play";
      playBtn.title = "Reproducir audio desde el inicio";
      playBtn.innerHTML = ICON_PLAY;
      playBtn.onclick = (e) => {
        e.stopPropagation();
        seekRecording(rec.id, 0, { pulse: true });
      };

      top.appendChild(titleWrap);
      top.appendChild(playBtn);
      card.appendChild(top);

      const meta = document.createElement("div");
      meta.className = "transcript-card-meta";
      const durBadge = document.createElement("span");
      durBadge.className = "card-badge";
      durBadge.textContent = formatDuration(rec.duration_sec);
      meta.appendChild(durBadge);

      if (rec.language) {
        const langBadge = document.createElement("span");
        langBadge.className = "card-badge";
        langBadge.textContent = String(rec.language).toUpperCase();
        meta.appendChild(langBadge);
      }

      const segInfo = document.createElement("span");
      segInfo.textContent = segs.length > 0
        ? `${segs.length} fragm. · ${wordCount} pal.`
        : `${wordCount} palabras`;
      meta.appendChild(segInfo);

      if (query && matchCount > 0) {
        const matchBadge = document.createElement("span");
        matchBadge.className = "transcript-match-badge";
        matchBadge.textContent = `${matchCount} coinc.`;
        meta.appendChild(matchBadge);
      }
      card.appendChild(meta);

      const preview = document.createElement("div");
      preview.className = "transcript-card-preview";
      highlightMatch(preview, snippet || "Sin texto disponible.");
      card.appendChild(preview);

      const foot = document.createElement("div");
      foot.className = "transcript-card-foot";
      foot.innerHTML = `<span>Abrir transcripción completa</span><span>→</span>`;
      card.appendChild(foot);

      const openDetail = () => {
        state.transcriptRecFilter = rec.id;
        const scroll = document.getElementById("center-scroll");
        if (scroll) scroll.scrollTop = 0;
        renderTranscriptTab(session);
      };
      card.onclick = openDetail;
      card.onkeydown = (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          openDetail();
        }
      };

      grid.appendChild(card);
    }

    if (visibleCards === 0) {
      pane.innerHTML = `<div class="chat-placeholder"><h3>Sin coincidencias</h3><p>Ninguna transcripción contiene "${escapeHtml(query)}".</p></div>`;
    } else {
      pane.appendChild(grid);
    }
    return;
  }

  // --- Focused Audio Transcript Detail View ---
  const navBar = document.createElement("div");
  navBar.className = "transcript-detail-nav";

  const backBtn = document.createElement("button");
  backBtn.type = "button";
  backBtn.className = "pill-btn outline sm transcript-back-btn";
  backBtn.textContent = "← Todos los audios";
  backBtn.onclick = () => {
    state.transcriptRecFilter = null;
    state.transcriptSpeakerFilter = null;
    renderTranscriptTab(session);
  };
  navBar.appendChild(backBtn);

  if (recordings.length >= 2) {
    const switcher = document.createElement("div");
    switcher.className = "transcript-filter-bar";
    for (const r of recordings) {
      const pillBtn = document.createElement("button");
      pillBtn.type = "button";
      pillBtn.className = `quiz-choice-pill ${r.id === activeRec.id ? "active" : ""}`;
      pillBtn.textContent = `🎙 ${r.filename}`;
      pillBtn.onclick = () => {
        state.transcriptRecFilter = r.id;
        state.transcriptSpeakerFilter = null;
        renderTranscriptTab(session);
      };
      switcher.appendChild(pillBtn);
    }
    navBar.appendChild(switcher);
  }
  pane.appendChild(navBar);

  const card = document.createElement("div");
  card.className = "content-card";

  const header = document.createElement("div");
  header.className = "content-card-header";
  header.innerHTML = `
    <span>🎙 ${escapeHtml(activeRec.filename)}</span>
    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
      <button type="button" class="pill-btn outline xs transcript-detail-play-btn" title="Reproducir este audio">▶ Escuchar</button>
      ${activeRec.whisper_model && activeRec.transcript_text ? `<span class="card-badge">${escapeHtml(activeRec.whisper_model)}</span>` : ""}
      <span class="card-badge">${formatDuration(activeRec.duration_sec)}${activeRec.segments?.length ? " · Pulsa una frase para escucharla" : ""}</span>
    </div>
  `;
  header.querySelector(".transcript-detail-play-btn").onclick = () => {
    seekRecording(activeRec.id, 0, { pulse: true });
  };
  card.appendChild(header);

  if (!activeRec.transcript_text) {
    const message = document.createElement("p");
    message.className = "rec-empty";
    message.textContent = activeRec.status === "error"
      ? activeRec.error_message || "No se pudo transcribir este audio."
      : activeRec.status === "ready"
        ? "Este audio no tiene texto transcrito."
        : "La transcripción aparecerá aquí cuando esté lista.";
    card.appendChild(message);
    pane.appendChild(card);
    return;
  }

  const segs = activeRec.segments || [];
  const speakerCounts = new Map();
  for (const s of segs) {
    const spk = (s.speaker || "").trim();
    if (spk) speakerCounts.set(spk, (speakerCounts.get(spk) || 0) + 1);
  }
  const orderedSpeakers = Array.from(speakerCounts.keys());

  if (orderedSpeakers.length > 0) {
    const toolbar = document.createElement("div");
    toolbar.className = "speaker-toolbar";

    const pillsWrap = document.createElement("div");
    pillsWrap.className = "speaker-filter-pills";

    const allPill = document.createElement("button");
    allPill.type = "button";
    allPill.className = `speaker-filter-pill ${!state.transcriptSpeakerFilter ? "active" : ""}`;
    allPill.textContent = `Todos (${segs.length})`;
    allPill.onclick = () => {
      state.transcriptSpeakerFilter = null;
      renderTranscriptTab(session);
    };
    pillsWrap.appendChild(allPill);

    for (const spk of orderedSpeakers) {
      const tone = getSpeakerToneClass(spk, orderedSpeakers);
      const pill = document.createElement("button");
      pill.type = "button";
      pill.className = `speaker-filter-pill ${tone} ${state.transcriptSpeakerFilter === spk ? "active" : ""}`;
      pill.innerHTML = `<span class="speaker-dot"></span><span>${escapeHtml(spk)}</span><span class="speaker-count">${speakerCounts.get(spk)}</span>`;
      pill.onclick = () => {
        state.transcriptSpeakerFilter = state.transcriptSpeakerFilter === spk ? null : spk;
        renderTranscriptTab(session);
      };
      pillsWrap.appendChild(pill);
    }
    toolbar.appendChild(pillsWrap);

    const renameBtn = document.createElement("button");
    renameBtn.type = "button";
    renameBtn.className = "pill-btn outline xs speaker-rename-toggle";
    renameBtn.textContent = "Renombrar hablantes";
    toolbar.appendChild(renameBtn);

    const renamePanel = document.createElement("form");
    renamePanel.className = "speaker-rename-panel hidden";
    renamePanel.innerHTML = `
      <div class="speaker-rename-grid">
        ${orderedSpeakers.map((spk, idx) => `
          <label class="speaker-rename-row">
            <span class="speaker-badge ${getSpeakerToneClass(spk, orderedSpeakers)}">${escapeHtml(spk)}</span>
            <input type="text" class="pill-input sm speaker-rename-input" data-old-speaker="${escapeHtml(spk)}" value="${escapeHtml(spk)}" maxlength="40" placeholder="Nombre del hablante ${idx + 1}" />
          </label>
        `).join("")}
      </div>
      <div class="speaker-rename-actions">
        <button type="button" class="pill-btn outline xs speaker-rename-cancel">Cancelar</button>
        <button type="submit" class="pill-btn dark xs speaker-rename-save">Guardar nombres</button>
      </div>
    `;
    renameBtn.onclick = () => renamePanel.classList.toggle("hidden");
    renamePanel.querySelector(".speaker-rename-cancel").onclick = () => renamePanel.classList.add("hidden");
    renamePanel.onsubmit = async (ev) => {
      ev.preventDefault();
      const mapping = {};
      renamePanel.querySelectorAll(".speaker-rename-input").forEach((inp) => {
        const oldName = inp.dataset.oldSpeaker;
        const nextName = inp.value.trim();
        if (oldName && nextName && oldName !== nextName) {
          mapping[oldName] = nextName;
        }
      });
      if (!Object.keys(mapping).length) {
        renamePanel.classList.add("hidden");
        return;
      }
      const saveBtn = renamePanel.querySelector(".speaker-rename-save");
      saveBtn.disabled = true;
      try {
        await api(`/api/sessions/${session.id}/recordings/${activeRec.id}/speakers`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ mapping }),
        });
        if (state.transcriptSpeakerFilter && mapping[state.transcriptSpeakerFilter]) {
          state.transcriptSpeakerFilter = mapping[state.transcriptSpeakerFilter];
        }
        await selectSession(session.id);
        showToast("Nombres de hablantes actualizados.");
      } catch (err) {
        showToast(err.message, true);
      } finally {
        saveBtn.disabled = false;
      }
    };

    card.appendChild(toolbar);
    card.appendChild(renamePanel);
  }

  const rows = [];
  let matchedAny = false;
  let lastRenderedSpeaker = null;
  if (segs.length > 0) {
    for (const seg of segs) {
      const text = seg.text || "";
      const speaker = (seg.speaker || "").trim();
      if (state.transcriptSpeakerFilter && speaker !== state.transcriptSpeakerFilter) continue;
      if (query && !text.toLowerCase().includes(query) && !speaker.toLowerCase().includes(query)) continue;
      matchedAny = true;

      const row = document.createElement("div");
      const turnChanged = Boolean(speaker && speaker !== lastRenderedSpeaker);
      row.className = `segment-row${turnChanged ? " speaker-turn-start" : ""}`;
      if (speaker) lastRenderedSpeaker = speaker;

      const playSegment = () => seekRecording(activeRec.id, seg.start, { pulse: true });
      const tsBtn = document.createElement("button");
      tsBtn.className = "timestamp-btn";
      tsBtn.textContent = formatDuration(seg.start);
      tsBtn.onclick = playSegment;

      if (speaker) {
        const spkBadge = document.createElement("span");
        spkBadge.className = `speaker-badge ${getSpeakerToneClass(speaker, orderedSpeakers)}`;
        spkBadge.textContent = speaker;
        spkBadge.title = "Filtrar por este hablante";
        spkBadge.onclick = () => {
          state.transcriptSpeakerFilter = state.transcriptSpeakerFilter === speaker ? null : speaker;
          renderTranscriptTab(session);
        };
        row.appendChild(tsBtn);
        row.appendChild(spkBadge);
      } else {
        row.appendChild(tsBtn);
      }

      const textEl = document.createElement("div");
      textEl.className = "segment-text";
      const words = document.createElement("span");
      words.className = "segment-words";
      words.onclick = () => {
        if (window.getSelection()?.isCollapsed !== false) playSegment();
      };
      const start = Number(seg.start);
      const end = Number(seg.end);
      const parts = Array.from(text.matchAll(/\S+/g));
      const timings = seg.words?.length === parts.length ? seg.words : null;
      const matches = searchPattern ? Array.from(text.matchAll(searchPattern)) : [];
      let matchIndex = 0;
      let offset = 0;
      const timedWords = parts.map((partMatch, i) => {
        const part = partMatch[0];
        const partStart = partMatch.index;
        const partEnd = partStart + part.length;
        words.append(text.slice(offset, partStart));
        offset = partEnd;

        const word = document.createElement("span");
        word.className = "segment-word";
        let cursor = partStart;
        while (matchIndex < matches.length && matches[matchIndex].index < partEnd) {
          const matchStart = matches[matchIndex].index;
          const matchEnd = matchStart + matches[matchIndex][0].length;
          if (matchEnd > cursor) {
            const mark = document.createElement("mark");
            mark.className = "hl";
            mark.textContent = text.slice(Math.max(cursor, matchStart), Math.min(partEnd, matchEnd));
            word.append(text.slice(cursor, Math.max(cursor, matchStart)), mark);
            cursor = Math.min(partEnd, matchEnd);
          }
          if (matchEnd > partEnd) break;
          matchIndex += 1;
        }
        word.append(text.slice(cursor, partEnd));
        words.appendChild(word);

        const wordStart = timings?.[i].start ?? start + (end - start) * i / parts.length;
        const wordEnd = timings?.[i].end ?? start + (end - start) * (i + 1) / parts.length;
        return { start: wordStart, end: wordEnd, element: word };
      });
      words.append(text.slice(offset));
      textEl.appendChild(words);

      row.appendChild(textEl);
      card.appendChild(row);
      rows.push({ start, end, element: row, words: timedWords });
    }
  } else if (!query || activeRec.transcript_text.toLowerCase().includes(query)) {
    matchedAny = true;
    const p = document.createElement("div");
    p.className = "segment-text";
    highlightMatch(p, activeRec.transcript_text);
    card.appendChild(p);
  }

  if (!matchedAny) {
    const emptyMatch = document.createElement("div");
    emptyMatch.className = "rec-empty";
    emptyMatch.textContent = "Sin coincidencias en este audio para la búsqueda actual.";
    card.appendChild(emptyMatch);
  }

  pane.appendChild(card);
  transcriptPlayback = { recId: activeRec.id, rows, currentRow: null, currentWord: null };
  const audio = audioPlayers.get(activeRec.id);
  if (audio && (!audio.paused || audio.currentTime > 0)) syncTranscriptPlayback(audio);
}

// --- Interactive Customizable Exam Studio ---

function getQuizState(sessionId) {
  let qs = sessionQuizzes.get(sessionId);
  if (!qs) {
    qs = {
      count: 5,
      mode: "mixed",
      difficulty: "medium",
      questionStyle: "conceptual",
      focus: "",
      includeHints: true,
      examMode: false,
      timeLimitMin: 0,
      selectedRecIds: new Set(),
      selectedSrcIds: new Set(),
      knownItemIds: new Set(),
      questions: null,
      answers: {},
      drafts: {},
      revealedHints: {},
      examSubmitted: false,
      startedAt: null,
      elapsedSec: 0,
      timerHandle: null,
      submittingExam: false,
      loading: false,
    };
    sessionQuizzes.set(sessionId, qs);
  }
  const activeSession = state.sessions && state.activeSession?.id === sessionId ? state.activeSession : null;
  if (activeSession) {
    const readyRecs = (activeSession.recordings || []).filter((r) => r.status === "ready");
    const readySrcs = (activeSession.sources || []).filter((s) => s.status === "ready");
    for (const r of readyRecs) {
      if (!qs.knownItemIds.has(`r:${r.id}`)) {
        qs.knownItemIds.add(`r:${r.id}`);
        if (r.enabled !== false) qs.selectedRecIds.add(r.id);
      }
    }
    for (const s of readySrcs) {
      if (!qs.knownItemIds.has(`s:${s.id}`)) {
        qs.knownItemIds.add(`s:${s.id}`);
        if (s.enabled !== false) qs.selectedSrcIds.add(s.id);
      }
    }
  }
  return qs;
}

function stopQuizTimer(qs) {
  if (qs && qs.timerHandle) {
    clearInterval(qs.timerHandle);
    qs.timerHandle = null;
  }
}

function startQuizTimer(session, qs) {
  stopQuizTimer(qs);
  qs.startedAt = Date.now();
  qs.elapsedSec = 0;
  qs.timerHandle = setInterval(() => {
    if (!qs.startedAt || (qs.examMode && qs.examSubmitted)) {
      stopQuizTimer(qs);
      return;
    }
    qs.elapsedSec = Math.floor((Date.now() - qs.startedAt) / 1000);
    const badge = document.querySelector("#tab-quiz .quiz-timer-badge");
    if (badge) {
      if (qs.timeLimitMin > 0) {
        const totalSec = qs.timeLimitMin * 60;
        const rem = Math.max(0, totalSec - qs.elapsedSec);
        badge.textContent = `⏱ ${formatDuration(rem)}`;
        badge.classList.toggle("urgent", rem <= 60);
        if (rem === 0 && qs.examMode && !qs.examSubmitted && !qs.submittingExam) {
          stopQuizTimer(qs);
          showToast("Tiempo agotado. Entregando y calificando tu examen...");
          submitFullExam(session);
        } else if (rem === 0 && !qs.examMode) {
          stopQuizTimer(qs);
        }
      } else {
        badge.textContent = `⏱ ${formatDuration(qs.elapsedSec)}`;
      }
    }
  }, 1000);
}

async function generateQuizForSession(session) {
  if (!session) return;
  const qs = getQuizState(session.id);
  if (qs.loading) return;

  const recIds = Array.from(qs.selectedRecIds);
  const srcIds = Array.from(qs.selectedSrcIds);
  if (recIds.length === 0 && srcIds.length === 0) {
    showToast("Selecciona al menos una fuente para generar el examen.", true);
    return;
  }

  stopQuizTimer(qs);
  const controller = new AbortController();
  state.quizAbortController = controller;
  qs.loading = true;
  qs.questions = null;
  qs.answers = {};
  qs.drafts = {};
  qs.revealedHints = {};
  qs.examSubmitted = false;
  qs.submittingExam = false;
  qs.elapsedSec = 0;
  qs.startedAt = null;
  syncGlobalStopButton();
  renderQuizTab(session);

  try {
    const res = await api(`/api/sessions/${session.id}/quiz/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        num_questions: qs.count,
        mode: qs.mode,
        difficulty: qs.difficulty,
        question_style: qs.questionStyle,
        focus_prompt: qs.focus,
        include_hints: qs.includeHints,
        recording_ids: recIds,
        source_ids: srcIds,
      }),
    });
    qs.questions = res.questions || [];
    qs.answers = {};
    qs.drafts = {};
    qs.revealedHints = {};
    qs.examSubmitted = false;
    if (qs.questions.length > 0) {
      startQuizTimer(session, qs);
    }
  } catch (err) {
    if (controller.signal.aborted || isAbortError(err)) {
      showToast("Generación de examen detenida.");
    } else {
      showToast(err.message, true);
    }
  } finally {
    if (state.quizAbortController === controller) {
      state.quizAbortController = null;
    }
    qs.loading = false;
    syncGlobalStopButton();
    if (state.activeSession && state.activeSession.id === session.id) {
      renderQuizTab(state.activeSession);
    }
  }
}

function computeQuizStats(qs) {
  const questions = qs.questions || [];
  const total = questions.length;
  let answeredCount = 0;
  let correctCount = 0;
  let partialCount = 0;
  let wrongCount = 0;
  let scoreSum = 0;

  for (const q of questions) {
    const ans = qs.answers[q.id];
    if (!ans) continue;
    if (ans.type === "mcq" || ans.type === "true_false") {
      if (qs.examMode && !qs.examSubmitted) {
        answeredCount += 1;
      } else {
        answeredCount += 1;
        if (ans.isCorrect) {
          correctCount += 1;
          scoreSum += 100;
        } else {
          wrongCount += 1;
        }
      }
    } else if (ans.type === "open" && !ans.grading && ans.result) {
      answeredCount += 1;
      const sc = Number(ans.result.score || 0);
      scoreSum += sc;
      if (ans.result.verdict === "correct" || sc >= 85) correctCount += 1;
      else if (ans.result.verdict === "partial" || sc >= 45) partialCount += 1;
      else wrongCount += 1;
    } else if (qs.examMode && !qs.examSubmitted && (qs.drafts[q.id] || "").trim()) {
      answeredCount += 1;
    }
  }

  const progressPct = Math.round((answeredCount / Math.max(1, total)) * 100);
  const avgScore = answeredCount > 0 ? Math.round(scoreSum / answeredCount) : 0;
  return { total, answeredCount, correctCount, partialCount, wrongCount, progressPct, avgScore };
}

function buildQuizQuestionCard(session, q, idx, total, animate = true) {
  const qs = getQuizState(session.id);
  const letters = ["A", "B", "C", "D"];
  const card = document.createElement("div");
  card.className = "quiz-card";
  card.dataset.qId = String(q.id);
  if (animate) {
    card.style.animationDelay = `${idx * 45}ms`;
  } else {
    card.style.animation = "none";
  }

  const ans = qs.answers[q.id];
  const isLockedEval = !qs.examMode ? Boolean(ans) : qs.examSubmitted;
  const typeLabel =
    q.type === "true_false"
      ? "Verdadero / Falso"
      : q.type === "mcq"
      ? "Opción múltiple"
      : "Respuesta abierta";

  const top = document.createElement("div");
  top.className = "quiz-card-top";
  top.innerHTML = `
    <span class="quiz-q-num">Pregunta ${idx + 1} de ${total}</span>
    <span class="card-badge">${typeLabel}</span>
  `;
  card.appendChild(top);

  const qText = document.createElement("div");
  qText.className = "quiz-q-text";
  qText.textContent = q.question;
  card.appendChild(qText);

  if (q.hint && qs.includeHints && !isLockedEval) {
    const hintRow = document.createElement("div");
    hintRow.className = "quiz-hint-row";
    const isHintOpen = Boolean(qs.revealedHints[q.id]);
    const hintBtn = document.createElement("button");
    hintBtn.type = "button";
    hintBtn.className = "quiz-hint-btn";
    hintBtn.textContent = isHintOpen ? "Ocultar pista" : "💡 Ver pista";
    hintBtn.onclick = () => {
      qs.revealedHints[q.id] = !qs.revealedHints[q.id];
      updateSingleQuizCard(session, q);
    };
    hintRow.appendChild(hintBtn);
    if (isHintOpen) {
      const hintBox = document.createElement("div");
      hintBox.className = "quiz-hint-box";
      hintBox.textContent = q.hint;
      hintRow.appendChild(hintBox);
    }
    card.appendChild(hintRow);
  }

  if (q.type === "mcq" || q.type === "true_false") {
    const optsWrap = document.createElement("div");
    optsWrap.className = `quiz-options ${q.type === "true_false" ? "tf-grid" : ""}`;
    const optsList = q.type === "true_false" && (!q.options || q.options.length < 2)
      ? ["Verdadero", "Falso"]
      : (q.options || []);

    optsList.forEach((optText, optIdx) => {
      const optBtn = document.createElement("button");
      optBtn.type = "button";
      optBtn.className = "quiz-option";
      if (isLockedEval) {
        optBtn.disabled = true;
        if (optIdx === q.correct_index) {
          optBtn.classList.add("correct");
        } else if (ans && optIdx === ans.selectedIndex && !ans.isCorrect) {
          optBtn.classList.add("wrong");
        }
      } else if (qs.examMode && ans && optIdx === ans.selectedIndex) {
        optBtn.classList.add("selected-pending");
      }

      const prefix = q.type === "true_false" ? (optIdx === 0 ? "V" : "F") : (letters[optIdx] || optIdx + 1);
      optBtn.innerHTML = `<span class="quiz-option-letter">${prefix}</span><span>${escapeHtml(optText)}</span>`;
      if (!isLockedEval) {
        optBtn.onclick = () => submitMcqAnswer(session, q, optIdx);
      }
      optsWrap.appendChild(optBtn);
    });
    card.appendChild(optsWrap);

    if (isLockedEval) {
      const fb = document.createElement("div");
      fb.className = "quiz-feedback";
      const tsPill = q.timestamp ? ` ${makeCitationPillHtml(q.timestamp)}` : "";
      const verdictHead = ans
        ? (ans.isCorrect ? "✓ Correcto" : "✕ Incorrecto")
        : "— Sin responder";
      const correctLabel = optsList[q.correct_index] || "";
      const expl = q.explanation ? escapeHtml(q.explanation) : `La respuesta correcta es: ${escapeHtml(correctLabel)}.`;
      fb.innerHTML = `<strong>${verdictHead}</strong><div>${expl}${tsPill}</div>`;
      card.appendChild(fb);
    }
  } else {
    const openWrap = document.createElement("div");
    openWrap.className = "quiz-open-wrap";

    const textarea = document.createElement("textarea");
    textarea.className = "quiz-open-input";
    textarea.placeholder = "Escribe tu respuesta basada en tus fuentes...";
    textarea.value = ans ? ans.userAnswer || "" : qs.drafts[q.id] || "";
    textarea.disabled = isLockedEval || Boolean(!qs.examMode && ans);
    if (!textarea.disabled) {
      textarea.addEventListener("input", () => {
        qs.drafts[q.id] = textarea.value;
        if (qs.examMode) {
          updateQuizProgressAndSummary(session);
        }
      });
    }

    openWrap.appendChild(textarea);

    if (!qs.examMode && !ans) {
      const actions = document.createElement("div");
      actions.className = "quiz-open-actions";
      const evalBtn = document.createElement("button");
      evalBtn.type = "button";
      evalBtn.className = "pill-btn dark sm";
      evalBtn.textContent = "Evaluar respuesta";
      evalBtn.onclick = () => submitOpenAnswer(session, q, textarea.value);
      actions.appendChild(evalBtn);
      openWrap.appendChild(actions);
    } else if (ans && ans.grading) {
      const loadingBox = document.createElement("div");
      loadingBox.className = "quiz-grade-box";
      loadingBox.innerHTML = `
        <div class="thinking-indicator">
          <div class="think-dots" aria-hidden="true"><span></span><span></span><span></span></div>
          <span class="think-label">Evaluando tu respuesta contra las fuentes...</span>
          <button type="button" class="inline-stop-btn" data-stop-scope="quiz-grade" title="Detener evaluación"><span class="stop-sq" aria-hidden="true"></span> Detener</button>
        </div>
      `;
      openWrap.appendChild(loadingBox);
    } else if (isLockedEval && ans && ans.result) {
      const res = ans.result;
      const gradeBox = document.createElement("div");
      gradeBox.className = `quiz-grade-box ${res.verdict}`;
      const verdictMap = {
        correct: "Correcta",
        partial: "Parcialmente correcta",
        incorrect: "Por mejorar",
      };
      const tsVal = res.timestamp || q.timestamp || "";
      const tsPill = tsVal ? ` ${makeCitationPillHtml(tsVal)}` : "";
      const missingHtml = res.missing_points
        ? `<div><strong>Te faltó mencionar:</strong> ${escapeHtml(res.missing_points)}</div>`
        : "";
      const expectedHtml = q.expected_answer
        ? `<div><strong>Referencia:</strong> ${escapeHtml(q.expected_answer)}${tsPill}</div>`
        : tsPill
        ? `<div><strong>Referencia en fuentes:</strong>${tsPill}</div>`
        : "";

      gradeBox.innerHTML = `
        <div class="quiz-grade-head">
          <span class="quiz-verdict-pill ${res.verdict}">${verdictMap[res.verdict] || "Evaluada"}</span>
          <span>${res.score}/100</span>
        </div>
        <div>${escapeHtml(res.feedback)}</div>
        ${missingHtml}
        ${expectedHtml}
      `;
      openWrap.appendChild(gradeBox);
    } else if (qs.examMode && qs.examSubmitted && !ans) {
      const tsPill = q.timestamp ? ` ${makeCitationPillHtml(q.timestamp)}` : "";
      const gradeBox = document.createElement("div");
      gradeBox.className = "quiz-grade-box incorrect";
      gradeBox.innerHTML = `
        <div class="quiz-grade-head">
          <span class="quiz-verdict-pill incorrect">Sin responder</span>
          <span>0/100</span>
        </div>
        <div><strong>Respuesta esperada:</strong> ${escapeHtml(q.expected_answer || "Revisa tus fuentes.")}${tsPill}</div>
      `;
      openWrap.appendChild(gradeBox);
    }

    card.appendChild(openWrap);
  }

  return card;
}

function getMissedQuizQuestions(qs) {
  return (qs.questions || []).filter((q) => {
    const ans = qs.answers[q.id];
    if (!ans) return true;
    if (ans.type === "mcq" || ans.type === "true_false") return !ans.isCorrect;
    if (ans.type === "open" && ans.result) return Number(ans.result.score || 0) < 85;
    return true;
  });
}

function retryWrongQuizQuestions(session) {
  const qs = getQuizState(session.id);
  const missed = getMissedQuizQuestions(qs);
  if (missed.length === 0) return;
  qs.questions = missed.map((q, idx) => ({ ...q, id: idx + 1 }));
  qs.answers = {};
  qs.drafts = {};
  qs.revealedHints = {};
  qs.examSubmitted = false;
  qs.submittingExam = false;
  startQuizTimer(session, qs);
  invalidateMemo("quiz-pane");
  renderQuizTab(session);
}

function buildQuizSummaryNode(session, qs, stats) {
  const missedCount = getMissedQuizQuestions(qs).length;
  const sumCard = document.createElement("div");
  sumCard.className = "quiz-summary-card";
  sumCard.innerHTML = `
    <div>
      <div style="font-size:12.5px; opacity:0.8;">Examen completado · ⏱ ${formatDuration(qs.elapsedSec || 0)}</div>
      <div style="font-size:21px; font-weight:800;">Calificación final: ${stats.avgScore} / 100</div>
      <div class="quiz-summary-breakdown">
        <span>✓ ${stats.correctCount} correctas</span>
        ${stats.partialCount > 0 ? `<span>· ~ ${stats.partialCount} parciales</span>` : ""}
        <span>· ✕ ${stats.wrongCount + Math.max(0, stats.total - stats.correctCount - stats.partialCount - stats.wrongCount)} por repasar</span>
      </div>
    </div>
    <div class="quiz-summary-actions">
      ${missedCount > 0 ? `<button type="button" class="pill-btn lime sm pod-retry-missed">Repasar ${missedCount} fallada(s)</button>` : ""}
      <button type="button" class="pill-btn dark-muted sm pod-new-exam">Configurar otro examen</button>
    </div>
  `;
  const retryBtn = sumCard.querySelector(".pod-retry-missed");
  if (retryBtn) {
    retryBtn.onclick = () => retryWrongQuizQuestions(session);
  }
  sumCard.querySelector(".pod-new-exam").onclick = () => {
    stopQuizTimer(qs);
    qs.questions = null;
    qs.answers = {};
    qs.drafts = {};
    qs.revealedHints = {};
    qs.examSubmitted = false;
    renderQuizTab(session);
  };
  return sumCard;
}

function updateQuizProgressAndSummary(session) {
  const pane = document.getElementById("tab-quiz");
  const qs = getQuizState(session.id);
  if (!pane || !qs.questions) return;

  const stats = computeQuizStats(qs);
  const progCard = pane.querySelector(".quiz-progress-card");
  if (progCard) {
    const spans = progCard.querySelectorAll(".quiz-progress-top-label");
    if (spans[0]) spans[0].textContent = `Progreso del examen: ${stats.answeredCount} / ${stats.total}`;
    if (spans[1]) {
      spans[1].textContent = qs.examMode && !qs.examSubmitted
        ? "Modo examen real: las notas se revelan al entregar"
        : stats.answeredCount > 0
        ? `Promedio: ${stats.avgScore}/100`
        : "Selecciona o redacta tus respuestas";
    }
    const fill = progCard.querySelector(".quiz-progress-fill");
    if (fill) fill.style.width = `${stats.progressPct}%`;
  }

  const allDone = qs.examMode ? qs.examSubmitted : stats.answeredCount === stats.total;
  if (allDone) {
    stopQuizTimer(qs);
  }
  const existingSummary = pane.querySelector(".quiz-summary-card");
  if (allDone && !existingSummary) {
    pane.appendChild(buildQuizSummaryNode(session, qs, stats));
  }
}

function updateSingleQuizCard(session, q) {
  if (!state.activeSession || state.activeSession.id !== session.id) return;
  const pane = document.getElementById("tab-quiz");
  const qs = getQuizState(session.id);
  if (!pane || !qs.questions) return;

  const idx = qs.questions.findIndex((item) => item.id === q.id);
  if (idx === -1) return;

  const existingCard = pane.querySelector(`.quiz-card[data-q-id="${q.id}"]`);
  if (!existingCard) {
    renderQuizTab(session);
    return;
  }

  const updatedCard = buildQuizQuestionCard(session, q, idx, qs.questions.length, false);
  existingCard.replaceWith(updatedCard);
  updateQuizProgressAndSummary(session);
  invalidateMemo("quiz-pane");
}

function submitMcqAnswer(session, q, selectedIndex) {
  const qs = getQuizState(session.id);
  if (qs.examMode) {
    if (qs.examSubmitted) return;
    qs.answers[q.id] = {
      type: q.type === "true_false" ? "true_false" : "mcq",
      selectedIndex,
      isCorrect: selectedIndex === q.correct_index,
    };
    updateSingleQuizCard(session, q);
    return;
  }

  if (qs.answers[q.id]) return;
  const isCorrect = selectedIndex === q.correct_index;
  qs.answers[q.id] = {
    type: q.type === "true_false" ? "true_false" : "mcq",
    selectedIndex,
    isCorrect,
  };
  updateSingleQuizCard(session, q);
}

async function submitOpenAnswer(session, q, userAnswerText) {
  const clean = (userAnswerText || "").trim();
  if (!clean) {
    showToast("Escribe tu respuesta antes de evaluar.", true);
    return;
  }
  const qs = getQuizState(session.id);
  const controller = new AbortController();
  state.quizGradeAbortController = controller;
  qs.drafts[q.id] = clean;
  qs.answers[q.id] = { type: "open", userAnswer: clean, grading: true, result: null };
  syncGlobalStopButton();
  updateSingleQuizCard(session, q);

  try {
    const result = await api(`/api/sessions/${session.id}/quiz/grade`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        question: q.question,
        expected_answer: q.expected_answer || "",
        timestamp: q.timestamp || "",
        user_answer: clean,
      }),
    });
    qs.answers[q.id] = { type: "open", userAnswer: clean, grading: false, result };
  } catch (err) {
    delete qs.answers[q.id];
    if (controller.signal.aborted || isAbortError(err)) {
      showToast("Evaluación detenida.");
    } else {
      showToast(err.message, true);
    }
  } finally {
    if (state.quizGradeAbortController === controller) {
      state.quizGradeAbortController = null;
    }
    syncGlobalStopButton();
    updateSingleQuizCard(session, q);
  }
}

async function submitFullExam(session) {
  const qs = getQuizState(session.id);
  if (!qs.questions || qs.submittingExam || qs.examSubmitted) return;

  const controller = new AbortController();
  state.quizGradeAbortController = controller;
  qs.submittingExam = true;
  stopQuizTimer(qs);
  syncGlobalStopButton();

  const submitBar = document.querySelector("#tab-quiz .quiz-submit-exam-bar");
  const submitBtn = submitBar?.querySelector(".quiz-submit-exam-btn");
  let tempStopBtn = null;
  if (submitBtn) {
    submitBtn.disabled = true;
    submitBtn.textContent = "Calificando examen...";
  }
  if (submitBar) {
    tempStopBtn = document.createElement("button");
    tempStopBtn.type = "button";
    tempStopBtn.className = "inline-stop-btn";
    tempStopBtn.dataset.stopScope = "quiz-grade";
    tempStopBtn.innerHTML = `<span class="stop-sq" aria-hidden="true"></span> Detener`;
    submitBar.appendChild(tempStopBtn);
  }

  try {
    for (const q of qs.questions) {
      if (controller.signal.aborted) {
        throw new DOMException("Aborted", "AbortError");
      }
      if (q.type === "open") {
        const draft = (qs.drafts[q.id] || "").trim();
        if (draft && !qs.answers[q.id]?.result) {
          const result = await api(`/api/sessions/${session.id}/quiz/grade`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            signal: controller.signal,
            body: JSON.stringify({
              question: q.question,
              expected_answer: q.expected_answer || "",
              timestamp: q.timestamp || "",
              user_answer: draft,
            }),
          });
          qs.answers[q.id] = { type: "open", userAnswer: draft, grading: false, result };
        }
      }
    }
    qs.examSubmitted = true;
    invalidateMemo("quiz-pane");
    renderQuizTab(session);
  } catch (err) {
    if (controller.signal.aborted || isAbortError(err)) {
      showToast("Calificación detenida. Tus respuestas siguen guardadas.");
    } else {
      showToast(err.message, true);
    }
    if (submitBtn) {
      submitBtn.disabled = false;
      submitBtn.textContent = "Entregar y calificar examen";
    }
  } finally {
    if (tempStopBtn) tempStopBtn.remove();
    if (state.quizGradeAbortController === controller) {
      state.quizGradeAbortController = null;
    }
    qs.submittingExam = false;
    syncGlobalStopButton();
  }
}

const STUDIO_ICONS = {
  mic: `<svg aria-hidden="true" viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="2" width="6" height="12" rx="3"/><path d="M5 10v2a7 7 0 0 0 14 0v-2M12 19v3M9 22h6"/></svg>`,
  doc: `<svg aria-hidden="true" viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="8" y1="13" x2="16" y2="13"/><line x1="8" y1="17" x2="13" y2="17"/></svg>`,
  image: `<svg aria-hidden="true" viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>`,
  check: `<svg aria-hidden="true" viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>`,
  mixed: `<svg aria-hidden="true" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><path d="M14 17.5h7M17.5 14v7"/></svg>`,
  mcq: `<svg aria-hidden="true" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="7" r="2.2"/><path d="M11 7h9"/><circle cx="6" cy="12" r="2.2" fill="currentColor"/><path d="M11 12h9"/><circle cx="6" cy="17" r="2.2"/><path d="M11 17h9"/></svg>`,
  true_false: `<svg aria-hidden="true" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="4 8 7 11 11 6"/><path d="M14 9h6"/><path d="M5 15l4 4M9 15l-4 4"/><path d="M14 17h6"/></svg>`,
  open: `<svg aria-hidden="true" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>`,
  immediate: `<svg aria-hidden="true" viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>`,
  exam: `<svg aria-hidden="true" viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="13" r="8"/><path d="M12 9v4l2.5 1.5"/><path d="M9 2h6"/></svg>`,
  conversation: `<svg aria-hidden="true" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/></svg>`,
  exam_review: `<svg aria-hidden="true" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4.5"/><circle cx="12" cy="12" r="1" fill="currentColor"/></svg>`,
  debate: `<svg aria-hidden="true" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 3l4 4-4 4"/><path d="M20 7H9"/><path d="M8 21l-4-4 4-4"/><path d="M4 17h11"/></svg>`,
  step_by_step: `<svg aria-hidden="true" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 18h6v-4h5v-4h5V6"/></svg>`,
};

const QUIZ_MODE_LABELS = {
  mixed: "Mixto completo",
  mcq: "Opción múltiple",
  true_false: "Verdadero / Falso",
  open: "Respuesta abierta",
};
const QUIZ_DIFF_LABELS = { easy: "Básico", medium: "Intermedio", hard: "Difícil / Parcial" };
const QUIZ_STYLE_LABELS = { conceptual: "Conceptual", practical: "Casos prácticos", tricky: "Trampas" };
const PODCAST_STYLE_LABELS = {
  conversation: "Conversación natural",
  exam_review: "Repaso de examen",
  debate: "Debate analítico",
  step_by_step: "Paso a paso",
};
const PODCAST_LEN_LABELS = { short: "Corto (~2 min)", medium: "Medio (~4 min)", long: "Largo (~7 min)" };

function buildSourceTilesHtml(readyRecs, readySrcs, selectedRecIds, selectedSrcIds, prefix) {
  return [
    ...readyRecs.map((r) => {
      const sel = selectedRecIds.has(r.id);
      const dur = r.duration_sec ? formatDuration(r.duration_sec) : "Audio";
      return `
        <button type="button" class="source-tile ${sel ? "selected" : ""}" data-${prefix}-kind="rec" data-${prefix}-item="${escapeHtml(r.id)}" aria-pressed="${sel}">
          <span class="source-tile-icon">${STUDIO_ICONS.mic}</span>
          <span class="source-tile-copy">
            <span class="source-tile-name">${escapeHtml(r.filename)}</span>
            <span class="source-tile-meta">${escapeHtml(dur)}</span>
          </span>
          <span class="source-tile-check">${STUDIO_ICONS.check}</span>
        </button>`;
    }),
    ...readySrcs.map((s) => {
      const sel = selectedSrcIds.has(s.id);
      const isImg = s.source_type === "image";
      const icon = isImg ? STUDIO_ICONS.image : STUDIO_ICONS.doc;
      const meta = isImg ? "Imagen" : (s.page_count ? `${s.page_count} pág.` : "Documento");
      return `
        <button type="button" class="source-tile ${sel ? "selected" : ""}" data-${prefix}-kind="src" data-${prefix}-item="${escapeHtml(s.id)}" aria-pressed="${sel}">
          <span class="source-tile-icon">${icon}</span>
          <span class="source-tile-copy">
            <span class="source-tile-name">${escapeHtml(s.filename)}</span>
            <span class="source-tile-meta">${escapeHtml(meta)}</span>
          </span>
          <span class="source-tile-check">${STUDIO_ICONS.check}</span>
        </button>`;
    }),
  ].join("");
}

function syncQuizSetupDOM(setup, session, qs) {
  if (!setup) return;
  const readyRecs = (session?.recordings || []).filter((r) => r.status === "ready");
  const readySrcs = (session?.sources || []).filter((s) => s.status === "ready");
  const totalSources = readyRecs.length + readySrcs.length;
  const selectedTotal = qs.selectedRecIds.size + qs.selectedSrcIds.size;

  setup.querySelectorAll(".source-tile").forEach((tile) => {
    const target = tile.dataset.quizKind === "rec" ? qs.selectedRecIds : qs.selectedSrcIds;
    const active = target.has(tile.dataset.quizItem);
    tile.classList.toggle("selected", active);
    tile.setAttribute("aria-pressed", String(active));
  });

  const srcCount = setup.querySelector("[data-quiz-src-count]");
  if (srcCount) srcCount.textContent = `${selectedTotal}/${totalSources} activa(s)`;

  const applyActive = (group, current) => {
    setup.querySelectorAll(`[data-quiz-group="${group}"] [data-val]`).forEach((btn) => {
      const on = btn.dataset.val === String(current);
      btn.classList.toggle("active", on);
      btn.setAttribute("aria-pressed", String(on));
    });
  };
  applyActive("count", qs.count);
  applyActive("mode", qs.mode);
  applyActive("difficulty", qs.difficulty);
  applyActive("questionStyle", qs.questionStyle);
  applyActive("examMode", qs.examMode ? "true" : "false");
  applyActive("timeLimitMin", qs.timeLimitMin);

  const hintToggle = setup.querySelector("[data-quiz-hints]");
  if (hintToggle) {
    hintToggle.classList.toggle("active", Boolean(qs.includeHints));
    hintToggle.setAttribute("aria-pressed", String(Boolean(qs.includeHints)));
  }

  const deckCount = setup.querySelector("[data-deck-count]");
  if (deckCount && deckCount.textContent !== String(qs.count)) {
    deckCount.textContent = String(qs.count);
    deckCount.classList.remove("pop-num");
    void deckCount.offsetWidth;
    deckCount.classList.add("pop-num");
  }
  const deckMode = setup.querySelector("[data-deck-mode]");
  if (deckMode) deckMode.textContent = QUIZ_MODE_LABELS[qs.mode] || qs.mode;
  const setSpec = (attr, val) => {
    const el = setup.querySelector(`[${attr}]`);
    if (el) el.textContent = val;
  };
  setSpec("data-deck-sources", selectedTotal > 0 ? `${selectedTotal} de ${totalSources}` : "Ninguna");
  setSpec("data-deck-diff", QUIZ_DIFF_LABELS[qs.difficulty] || qs.difficulty);
  setSpec("data-deck-style", QUIZ_STYLE_LABELS[qs.questionStyle] || qs.questionStyle);
  setSpec("data-deck-modality", qs.examMode ? "Modo examen real" : "Feedback inmediato");
  setSpec("data-deck-timer", qs.timeLimitMin > 0 ? `${qs.timeLimitMin} min` : "Sin límite");
  setSpec("data-deck-hints", qs.includeHints ? "Activadas" : "Ocultas");

  const emptyWarn = setup.querySelector("[data-deck-warn]");
  if (emptyWarn) emptyWarn.classList.toggle("hidden", selectedTotal > 0);

  const btn = setup.querySelector("#btn-generate-quiz");
  if (btn) btn.disabled = Boolean(state.podcastBusy) || selectedTotal === 0;
}

function renderQuizTab(session) {
  const pane = document.getElementById("tab-quiz");
  const resetBtn = document.getElementById("btn-reset-quiz");
  if (!pane) return;

  const readyRecs = (session?.recordings || []).filter((r) => r.status === "ready");
  const readySrcs = (session?.sources || []).filter((s) => s.status === "ready");
  const qs = getQuizState(session?.id || "none");
  const isSetupView = !qs.loading && (!qs.questions || qs.questions.length === 0);

  const quizSig = isSetupView
    ? [
        "setup",
        session?.id || "",
        readyRecs.map((r) => `${r.id}:${r.filename}`).join(","),
        readySrcs.map((s) => `${s.id}:${s.filename}`).join(","),
      ].join("|")
    : [
        "active",
        session?.id || "",
        readyRecs.length,
        readySrcs.length,
        qs.loading ? 1 : 0,
        qs.examSubmitted ? 1 : 0,
        qs.questions ? qs.questions.length : -1,
        JSON.stringify(qs.answers),
      ].join("|");

  if (!memoChanged("quiz-pane", quizSig)) {
    if (isSetupView && session && (readyRecs.length > 0 || readySrcs.length > 0)) {
      syncQuizSetupDOM(pane.querySelector(".quiz-setup-shell"), session, qs);
    }
    return;
  }

  if (!session || (readyRecs.length === 0 && readySrcs.length === 0)) {
    if (resetBtn) resetBtn.classList.add("hidden");
    pane.innerHTML = `<div class="chat-placeholder"><h3>Estudio de exámenes</h3><p>Cuando al menos una grabación o fuente documental esté lista podrás personalizar aquí simulacros de examen basados en tus materiales.</p></div>`;
    return;
  }

  if (qs.loading) {
    if (resetBtn) resetBtn.classList.add("hidden");
    const modelName = getActiveModelLabel();
    pane.innerHTML = buildSkeletonHtml(
      `${modelName} está diseñando tu examen personalizado...`,
      true,
      "quiz-gen"
    );
    return;
  }

  if (!qs.questions || qs.questions.length === 0) {
    if (resetBtn) resetBtn.classList.add("hidden");
    pane.innerHTML = "";

    const sourceTilesHtml = buildSourceTilesHtml(readyRecs, readySrcs, qs.selectedRecIds, qs.selectedSrcIds, "quiz");
    const formatCards = [
      { id: "mixed", title: "Mixto completo", desc: "Combina opción múltiple, V/F y respuesta abierta" },
      { id: "mcq", title: "Opción múltiple", desc: "4 alternativas balanceadas con distractores claros" },
      { id: "true_false", title: "Verdadero / Falso", desc: "Enunciados directos balanceados al 50/50" },
      { id: "open", title: "Respuesta abierta", desc: "Desarrollo libre calificado con retroalimentación IA" },
    ];
    const modeCards = [
      { id: "false", icon: "immediate", title: "Feedback inmediato", desc: "Corrige cada pregunta al instante" },
      { id: "true", icon: "exam", title: "Modo examen real", desc: "Nota final al entregar todas" },
    ];
    const focusSuggestions = [
      "Definiciones clave",
      "Fórmulas y reglas",
      "Casos prácticos",
      "Diferencias y contrastes",
    ];

    const setup = document.createElement("div");
    setup.className = "quiz-setup-shell";
    setup.innerHTML = `
      <div class="studio-composer-layout">
        <div class="studio-configurator">
          <div class="studio-head">
            <span class="studio-eyebrow">EXÁMENES · SIMULACRO LOCAL</span>
            <h2>Configura tu evaluación</h2>
            <p>Combina grabaciones y apuntes en un simulacro adaptado al nivel del examen.</p>
          </div>

          <section class="studio-section">
            <div class="studio-section-head">
              <span class="studio-section-label">1. Fuentes a evaluar</span>
              <div class="studio-section-actions">
                <span class="studio-count-badge" data-quiz-src-count></span>
                <button type="button" class="studio-link-btn" data-quiz-select-all="1">Todas</button>
                <button type="button" class="studio-link-btn" data-quiz-select-all="0">Ninguna</button>
              </div>
            </div>
            <div class="source-tiles-grid">${sourceTilesHtml}</div>
          </section>

          <section class="studio-section">
            <div class="studio-section-head">
              <span class="studio-section-label">2. Formato de preguntas</span>
            </div>
            <div class="studio-cards-2x2" data-quiz-group="mode">
              ${formatCards.map((c) => `
                <button type="button" class="studio-option-card ${qs.mode === c.id ? "active" : ""}" data-val="${c.id}">
                  <span class="studio-card-top">
                    <span class="studio-card-icon">${STUDIO_ICONS[c.id]}</span>
                    <span class="studio-card-dot"></span>
                  </span>
                  <span class="studio-card-title">${c.title}</span>
                  <span class="studio-card-desc">${c.desc}</span>
                </button>`).join("")}
            </div>
          </section>

          <section class="studio-section">
            <div class="studio-section-head">
              <span class="studio-section-label">3. Extensión, dificultad y estilo</span>
            </div>
            <div class="studio-params-stack">
              <div class="seg-control-track" data-quiz-group="count" aria-label="Cantidad de preguntas">
                ${[3, 5, 8, 10, 15].map((n) => `<button type="button" class="seg-control-item" data-val="${n}">${n} preg.</button>`).join("")}
              </div>
              <div class="studio-dual-tracks">
                <div class="seg-control-track" data-quiz-group="difficulty" aria-label="Dificultad">
                  <button type="button" class="seg-control-item" data-val="easy">Básico</button>
                  <button type="button" class="seg-control-item" data-val="medium">Intermedio</button>
                  <button type="button" class="seg-control-item" data-val="hard">Difícil</button>
                </div>
                <div class="seg-control-track" data-quiz-group="questionStyle" aria-label="Estilo">
                  <button type="button" class="seg-control-item" data-val="conceptual">Conceptual</button>
                  <button type="button" class="seg-control-item" data-val="practical">Casos</button>
                  <button type="button" class="seg-control-item" data-val="tricky">Trampas</button>
                </div>
              </div>
            </div>
          </section>

          <section class="studio-section">
            <div class="studio-section-head">
              <span class="studio-section-label">4. Modalidad, tiempo y pistas</span>
            </div>
            <div class="studio-cards-2x1" data-quiz-group="examMode">
              ${modeCards.map((c) => `
                <button type="button" class="studio-option-card compact" data-val="${c.id}">
                  <span class="studio-card-top">
                    <span class="studio-card-icon">${STUDIO_ICONS[c.icon]}</span>
                    <span class="studio-card-dot"></span>
                  </span>
                  <span class="studio-card-title">${c.title}</span>
                  <span class="studio-card-desc">${c.desc}</span>
                </button>`).join("")}
            </div>
            <div class="studio-timer-row">
              <div class="seg-control-track" data-quiz-group="timeLimitMin" aria-label="Tiempo límite">
                <button type="button" class="seg-control-item" data-val="0">Sin límite</button>
                <button type="button" class="seg-control-item" data-val="5">5 min</button>
                <button type="button" class="seg-control-item" data-val="10">10 min</button>
                <button type="button" class="seg-control-item" data-val="15">15 min</button>
              </div>
              <button type="button" class="studio-switch-pill" data-quiz-hints aria-pressed="false">
                <span class="switch-dot"></span>
                <span>Incluir pistas</span>
              </button>
            </div>
          </section>

          <section class="studio-section">
            <div class="studio-section-head">
              <span class="studio-section-label">5. Enfoque temático (opcional)</span>
            </div>
            <textarea class="podcast-focus-input quiz-focus-input" placeholder="Ej. Enfócate en definiciones clave, fórmulas del capítulo 3 o casos clínicos...">${escapeHtml(qs.focus)}</textarea>
            <div class="studio-focus-chips">
              ${focusSuggestions.map((label) => `<button type="button" class="focus-preset-chip" data-focus-chip="${escapeHtml(label)}">+ ${escapeHtml(label)}</button>`).join("")}
            </div>
          </section>
        </div>

        <aside class="studio-summary-deck obsidian-card">
          <div class="deck-top-row">
            <span class="deck-badge">PLAN DE EXAMEN</span>
            <span class="deck-mode-tag" data-deck-mode></span>
          </div>
          <div class="deck-hero-metric">
            <span class="deck-big-num" data-deck-count>${qs.count}</span>
            <span class="deck-big-unit">preguntas<br/>a medida</span>
          </div>
          <div class="deck-spec-list">
            <div class="deck-spec-row"><span>Fuentes</span><strong data-deck-sources></strong></div>
            <div class="deck-spec-row"><span>Nivel</span><strong data-deck-diff></strong></div>
            <div class="deck-spec-row"><span>Enfoque</span><strong data-deck-style></strong></div>
            <div class="deck-spec-row"><span>Evaluación</span><strong data-deck-modality></strong></div>
            <div class="deck-spec-row"><span>Temporizador</span><strong data-deck-timer></strong></div>
            <div class="deck-spec-row"><span>Pistas</span><strong data-deck-hints></strong></div>
          </div>
          <p class="deck-warning hidden" data-deck-warn>Selecciona al menos una fuente para habilitar la generación.</p>
          <button type="button" id="btn-generate-quiz" class="quiz-start-btn">
            <span>Generar examen</span>
            <span class="cta-circle" aria-hidden="true">→</span>
          </button>
        </aside>
      </div>
    `;

    setup.querySelectorAll(".source-tile").forEach((tile) => {
      tile.onclick = () => {
        const target = tile.dataset.quizKind === "rec" ? qs.selectedRecIds : qs.selectedSrcIds;
        const id = tile.dataset.quizItem;
        if (target.has(id)) target.delete(id);
        else target.add(id);
        syncQuizSetupDOM(setup, session, qs);
      };
    });

    setup.querySelectorAll("[data-quiz-select-all]").forEach((btn) => {
      btn.onclick = () => {
        const selectAll = btn.dataset.quizSelectAll === "1";
        qs.selectedRecIds.clear();
        qs.selectedSrcIds.clear();
        if (selectAll) {
          readyRecs.forEach((r) => qs.selectedRecIds.add(r.id));
          readySrcs.forEach((s) => qs.selectedSrcIds.add(s.id));
        }
        syncQuizSetupDOM(setup, session, qs);
      };
    });

    const focusArea = setup.querySelector(".quiz-focus-input");
    focusArea.addEventListener("input", () => {
      qs.focus = focusArea.value;
    });
    setup.querySelectorAll("[data-focus-chip]").forEach((chip) => {
      chip.onclick = () => {
        const phrase = chip.dataset.focusChip;
        const cur = focusArea.value.trim();
        focusArea.value = cur ? `${cur}, ${phrase.toLowerCase()}` : phrase;
        qs.focus = focusArea.value;
        focusArea.focus();
      };
    });

    const bindGroup = (group, parser = (v) => v) => {
      setup.querySelectorAll(`[data-quiz-group="${group}"] [data-val]`).forEach((b) => {
        b.onclick = () => {
          qs[group] = parser(b.dataset.val);
          syncQuizSetupDOM(setup, session, qs);
        };
      });
    };
    bindGroup("count", Number);
    bindGroup("mode");
    bindGroup("difficulty");
    bindGroup("questionStyle");
    bindGroup("examMode", (v) => v === "true");
    bindGroup("timeLimitMin", Number);

    setup.querySelector("[data-quiz-hints]").onclick = () => {
      qs.includeHints = !qs.includeHints;
      syncQuizSetupDOM(setup, session, qs);
    };

    setup.querySelector("#btn-generate-quiz").onclick = () => generateQuizForSession(session);
    syncQuizSetupDOM(setup, session, qs);
    pane.appendChild(setup);
    return;
  }

  if (resetBtn) resetBtn.classList.remove("hidden");

  const stats = computeQuizStats(qs);
  pane.innerHTML = "";

  const timerText = qs.timeLimitMin > 0
    ? `⏱ ${formatDuration(Math.max(0, qs.timeLimitMin * 60 - (qs.elapsedSec || 0)))}`
    : `⏱ ${formatDuration(qs.elapsedSec || 0)}`;

  const progCard = document.createElement("div");
  progCard.className = "quiz-progress-card";
  progCard.innerHTML = `
    <div class="quiz-progress-top">
      <span class="quiz-progress-top-label">Progreso del examen: ${stats.answeredCount} / ${stats.total}</span>
      <div style="display:flex;align-items:center;gap:10px;">
        <span class="quiz-progress-top-label">${
          qs.examMode && !qs.examSubmitted
            ? "Modo examen real"
            : stats.answeredCount > 0
            ? `Promedio: ${stats.avgScore}/100`
            : "Selecciona o redacta tus respuestas"
        }</span>
        <span class="quiz-timer-badge">${timerText}</span>
      </div>
    </div>
    <div class="quiz-progress-track">
      <div class="quiz-progress-fill" style="width:${stats.progressPct}%"></div>
    </div>
  `;
  pane.appendChild(progCard);

  qs.questions.forEach((q, idx) => {
    pane.appendChild(buildQuizQuestionCard(session, q, idx, stats.total, true));
  });

  if (qs.examMode && !qs.examSubmitted) {
    const submitBar = document.createElement("div");
    submitBar.className = "quiz-submit-exam-bar";
    submitBar.innerHTML = `
      <div>
        <strong>¿Listo para entregar?</strong>
        <div style="font-size:12.5px;opacity:0.78;">Respondidas: ${stats.answeredCount} de ${stats.total}. Al entregar se calificarán tus respuestas abiertas y verás el desglose completo.</div>
      </div>
      <button type="button" class="quiz-start-btn quiz-submit-exam-btn" style="width:auto;padding:10px 20px;">
        <span>Entregar y calificar examen</span>
      </button>
    `;
    submitBar.querySelector(".quiz-submit-exam-btn").onclick = () => submitFullExam(session);
    pane.appendChild(submitBar);
  }

  const allDone = qs.examMode ? qs.examSubmitted : stats.answeredCount === stats.total;
  if (allDone) {
    pane.appendChild(buildQuizSummaryNode(session, qs, stats));
  }
}

// --- Local Podcast Studio Tab (Exclusive Mode + Surgical Updates) ---

function getPodcastFormState(session) {
  let st = podcastFormState.get(session.id);
  if (!st) {
    st = {
      selectedRecIds: new Set(),
      selectedSrcIds: new Set(),
      knownItemIds: new Set(),
      focus: "",
      style: "conversation",
      length: "medium",
      language: "es",
      submitting: false,
    };
    podcastFormState.set(session.id, st);
  }
  const readyRecs = (session.recordings || []).filter((r) => r.status === "ready");
  const readySrcs = (session.sources || []).filter((s) => s.status === "ready");
  for (const r of readyRecs) {
    if (!st.knownItemIds.has(`r:${r.id}`)) {
      st.knownItemIds.add(`r:${r.id}`);
      if (r.enabled !== false) st.selectedRecIds.add(r.id);
    }
  }
  for (const s of readySrcs) {
    if (!st.knownItemIds.has(`s:${s.id}`)) {
      st.knownItemIds.add(`s:${s.id}`);
      if (s.enabled !== false) st.selectedSrcIds.add(s.id);
    }
  }
  return st;
}

function getOrCreatePodcastAudio(pod) {
  let audio = podcastAudioPlayers.get(pod.id);
  if (!audio) {
    audio = new Audio(`/api/podcasts/${pod.id}/audio`);
    audio.preload = "metadata";
    audio.addEventListener("timeupdate", () => syncPodcastAudioCardDOM(pod.id));
    audio.addEventListener("play", () => syncPodcastAudioCardDOM(pod.id));
    audio.addEventListener("pause", () => syncPodcastAudioCardDOM(pod.id));
    audio.addEventListener("ended", () => syncPodcastAudioCardDOM(pod.id));
    audio.addEventListener("loadedmetadata", () => syncPodcastAudioCardDOM(pod.id));
    podcastAudioPlayers.set(pod.id, audio);
  }
  return audio;
}

function syncPodcastAudioCardDOM(podId) {
  const card = document.querySelector(`.podcast-episode-card[data-pod-id="${podId}"]`);
  const audio = podcastAudioPlayers.get(podId);
  if (!card || !audio) return;

  const playBtn = card.querySelector(".play-circle");
  if (playBtn) playBtn.innerHTML = audio.paused ? ICON_PLAY : ICON_PAUSE;

  const dur = Number.isFinite(audio.duration) && audio.duration > 0
    ? audio.duration
    : Number(card.dataset.durationSec || 0);
  const cur = audio.currentTime || 0;
  const pct = dur > 0 ? Math.min(100, (cur / dur) * 100) : 0;

  const fill = card.querySelector(".scrubber-fill");
  if (fill) fill.style.width = `${pct}%`;

  const timeEl = card.querySelector(".pod-time");
  if (timeEl) timeEl.textContent = `${formatDuration(cur)} / ${formatDuration(dur)}`;

  const speedBtn = card.querySelector(".pod-speed-btn");
  if (speedBtn) speedBtn.textContent = `${audio.playbackRate || 1}x`;
}

function getPodcastStageHeadline(activePod) {
  if (!activePod) return "Preparando motores locales...";
  if (activePod.status === "scripting") return "1/3 Escribiendo el guion de diálogo con IA...";
  if ((activePod.progress || 0) < 0.43) return "2/3 Liberando VRAM y preparando voces Kokoro-82M...";
  return "3/3 Sintetizando voces neuronales Kokoro-82M en tu equipo...";
}

async function cancelActivePodcast(sessionId, podcastId, btnEl) {
  if (!sessionId || !podcastId || state.cancellingPodcastId === podcastId) return;
  state.cancellingPodcastId = podcastId;
  if (btnEl) {
    btnEl.disabled = true;
    btnEl.textContent = "Cancelando...";
  }
  try {
    await api(`/api/sessions/${sessionId}/podcasts/${podcastId}/cancel`, {
      method: "POST",
    });
    syncPodcastBusyUI(false);
    showToast("Generación de podcast cancelada.");
    await selectSession(sessionId);
    await refreshSystemStatus();
  } catch (err) {
    showToast(err.message, true);
    if (btnEl) {
      btnEl.disabled = false;
      btnEl.textContent = "✕ Cancelar";
    }
  } finally {
    state.cancellingPodcastId = null;
  }
}

async function generatePodcastForSession(session) {
  const st = getPodcastFormState(session);
  if (st.submitting || state.podcastBusy) return;
  const recIds = Array.from(st.selectedRecIds);
  const srcIds = Array.from(st.selectedSrcIds);
  if (recIds.length === 0 && srcIds.length === 0) {
    showToast("Selecciona al menos una fuente para crear el podcast.", true);
    return;
  }

  st.submitting = true;
  state.podcastStartedAt = Date.now();
  syncPodcastBusyUI(true);
  const centerScroll = document.getElementById("center-scroll");
  if (centerScroll) {
    centerScroll.scrollTo({ top: 0, behavior: "smooth" });
  }
  renderPodcastTab(session);

  try {
    await api(`/api/sessions/${session.id}/podcasts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        focus_prompt: st.focus,
        style: st.style,
        length: st.length,
        language: st.language,
        recording_ids: recIds,
        source_ids: srcIds,
      }),
    });
    await selectSession(session.id);
    await refreshSystemStatus();
  } catch (err) {
    syncPodcastBusyUI(false);
    showToast(err.message, true);
  } finally {
    st.submitting = false;
    if (state.activeSession && state.activeSession.id === session.id) {
      renderPodcastTab(state.activeSession);
    }
  }
}

function syncPodcastSetupDOM(pane, session, st) {
  if (!pane || !st) return;
  const readyRecs = (session?.recordings || []).filter((r) => r.status === "ready");
  const readySrcs = (session?.sources || []).filter((s) => s.status === "ready");
  const totalSources = readyRecs.length + readySrcs.length;
  const selectedTotal = st.selectedRecIds.size + st.selectedSrcIds.size;

  pane.querySelectorAll(".source-tile").forEach((tile) => {
    const targetSet = tile.dataset.podKind === "rec" ? st.selectedRecIds : st.selectedSrcIds;
    const active = targetSet.has(tile.dataset.podItem);
    tile.classList.toggle("selected", active);
    tile.setAttribute("aria-pressed", String(active));
  });

  const srcCount = pane.querySelector("[data-pod-src-count]");
  if (srcCount) srcCount.textContent = `${selectedTotal}/${totalSources} activa(s)`;

  const applyActive = (group, current) => {
    pane.querySelectorAll(`[data-pod-group="${group}"] [data-val]`).forEach((btn) => {
      const on = btn.dataset.val === String(current);
      btn.classList.toggle("active", on);
      btn.setAttribute("aria-pressed", String(on));
    });
  };
  applyActive("style", st.style);
  applyActive("length", st.length);
  applyActive("language", st.language);

  const hostsEl = pane.querySelector("[data-pod-hosts]");
  if (hostsEl) {
    const isEn = st.language === "en";
    hostsEl.innerHTML = isEn
      ? `<span class="cast-chip host-a"><span class="cast-dot">A</span>Claire</span><span class="cast-sep">+</span><span class="cast-chip host-b"><span class="cast-dot">B</span>Ryan</span>`
      : `<span class="cast-chip host-a"><span class="cast-dot">A</span>Sofía</span><span class="cast-sep">+</span><span class="cast-chip host-b"><span class="cast-dot">B</span>Mateo</span>`;
  }

  const setSpec = (attr, val) => {
    const el = pane.querySelector(`[${attr}]`);
    if (el) el.textContent = val;
  };
  setSpec("data-pod-deck-style", PODCAST_STYLE_LABELS[st.style] || st.style);
  setSpec("data-pod-deck-len", PODCAST_LEN_LABELS[st.length] || st.length);
  setSpec("data-pod-deck-sources", selectedTotal > 0 ? `${selectedTotal} de ${totalSources}` : "Ninguna");

  const warn = pane.querySelector("[data-pod-warn]");
  if (warn) warn.classList.toggle("hidden", selectedTotal > 0);

  const btn = pane.querySelector(".pod-start-btn");
  if (btn) btn.disabled = selectedTotal === 0;
}

function renderPodcastTab(session) {
  const pane = document.getElementById("tab-podcast");
  if (!pane) return;

  const readyRecs = (session?.recordings || []).filter((r) => r.status === "ready");
  const readySrcs = (session?.sources || []).filter((s) => s.status === "ready");
  const podcasts = session?.podcasts || [];
  const activePod = podcasts.find((p) =>
    ["queued", "scripting", "synthesizing", "mixing"].includes(p.status)
  );
  const st = session ? getPodcastFormState(session) : null;
  const isLocked = Boolean(state.podcastBusy || activePod || st?.submitting);
  const activeScriptLen = activePod ? (activePod.script || []).length : 0;

  const structSig = [
    session?.id || "",
    readyRecs.map((r) => `${r.id}:${r.filename}`).join(","),
    readySrcs.map((s) => `${s.id}:${s.filename}`).join(","),
    isLocked ? 1 : 0,
    activePod ? activePod.id : "",
    activeScriptLen,
    podcasts.map((p) => `${p.id}:${p.status === "ready" || p.status === "error" ? p.status : "working"}:${(p.script || []).length}:${p.error_message || ""}`).join(","),
  ].join("|");

  if (!memoChanged("podcast-pane", structSig)) {
    if (isLocked) {
      const stage = pane.querySelector(".podcast-live-stage");
      if (stage) {
        const pctNum = activePod ? Math.max(8, Math.round((activePod.progress || 0) * 100)) : 8;
        const pctEl = stage.querySelector(".podcast-live-pct");
        if (pctEl) pctEl.textContent = `${pctNum}%`;
        const titleEl = stage.querySelector(".podcast-live-title");
        const nextTitle = getPodcastStageHeadline(activePod);
        if (titleEl && titleEl.textContent !== nextTitle) titleEl.textContent = nextTitle;
        const subEl = stage.querySelector(".podcast-live-stage-detail");
        const nextDetail = activePod?.stage || "Modo exclusivo activo: dedicando toda la memoria local al podcast.";
        if (subEl && subEl.textContent !== nextDetail) subEl.textContent = nextDetail;
        const fillEl = stage.querySelector(".podcast-live-fill");
        if (fillEl) fillEl.style.width = `${pctNum}%`;

        const stepIdx = !activePod || activePod.status === "queued" || activePod.status === "scripting"
          ? 0
          : (activePod.progress || 0) < 0.43
          ? 1
          : 2;
        stage.querySelectorAll(".podcast-stage-step").forEach((el, idx) => {
          el.classList.toggle("done", idx < stepIdx);
          el.classList.toggle("active", idx === stepIdx);
        });
      }
    } else if (session && st && (readyRecs.length > 0 || readySrcs.length > 0)) {
      syncPodcastSetupDOM(pane, session, st);
    }
    return;
  }

  pane.innerHTML = "";

  if (!session || (readyRecs.length === 0 && readySrcs.length === 0)) {
    pane.innerHTML = `<div class="chat-placeholder"><h3>Podcast Studio local</h3><p>Procesa al menos una grabación, PDF, presentación o imagen para generar un podcast de repaso con dos conductores 100% en tu equipo.</p></div>`;
    return;
  }

  if (isLocked) {
    const pctNum = activePod ? Math.max(8, Math.round((activePod.progress || 0) * 100)) : 8;
    const stepIdx = !activePod || activePod.status === "queued" || activePod.status === "scripting"
      ? 0
      : (activePod.progress || 0) < 0.43
      ? 1
      : 2;
    const elapsedText = formatElapsedClock(state.podcastStartedAt ? Date.now() - state.podcastStartedAt : 0);
    const stageDetail = activePod?.stage || "Iniciando generación en modo exclusivo local...";

    const stage = document.createElement("div");
    stage.className = "podcast-live-stage";
    stage.dataset.podId = activePod?.id || "pending";
    const canCancelNow = Boolean(activePod && activePod.id && activePod.id !== "pending");
    const isCancelling = Boolean(canCancelNow && state.cancellingPodcastId === activePod.id);
    stage.innerHTML = `
      <div class="podcast-live-head">
        <span class="podcast-live-badge"><span class="podcast-live-dot"></span>EN PRODUCCIÓN · MODO EXCLUSIVO</span>
        <div class="podcast-live-metrics">
          <span class="podcast-elapsed">${elapsedText}</span>
          <span class="podcast-live-pct">${pctNum}%</span>
          ${canCancelNow ? `<button type="button" class="pod-cancel-btn" ${isCancelling ? "disabled" : ""}>${isCancelling ? "Cancelando..." : "✕ Cancelar"}</button>` : ""}
        </div>
      </div>
      <div class="podcast-live-hero">
        <div class="podcast-wave-anim" aria-hidden="true">
          <span></span><span></span><span></span><span></span><span></span><span></span><span></span>
        </div>
        <div class="podcast-live-copy">
          <h2 class="podcast-live-title">${escapeHtml(getPodcastStageHeadline(activePod))}</h2>
          <p class="podcast-live-stage-detail podcast-live-sub">${escapeHtml(stageDetail)}</p>
        </div>
      </div>
      <div class="podcast-live-bar">
        <div class="podcast-live-fill" style="width:${pctNum}%"></div>
      </div>
      <div class="podcast-stage-steps">
        <span class="podcast-stage-step ${stepIdx > 0 ? "done" : "active"}">1/3 Guion con Ollama</span>
        <span class="podcast-stage-step ${stepIdx > 1 ? "done" : stepIdx === 1 ? "active" : ""}">2/3 Liberar VRAM</span>
        <span class="podcast-stage-step ${stepIdx === 2 ? "active" : ""}">3/3 Voces Kokoro-82M</span>
      </div>
      <div class="podcast-live-lock-note">
        🔒 Las demás pestañas, chats, exámenes y cargas están bloqueadas temporalmente para dedicar toda la CPU/GPU al podcast. Pulsa <strong>Cancelar</strong> si deseas detenerlo al instante.
      </div>
    `;

    const cancelBtn = stage.querySelector(".pod-cancel-btn");
    if (cancelBtn && activePod) {
      cancelBtn.onclick = () => cancelActivePodcast(session.id, activePod.id, cancelBtn);
    }

    if (activePod && Array.isArray(activePod.script) && activePod.script.length > 0) {
      const liveScript = document.createElement("div");
      liveScript.className = "podcast-live-script";
      liveScript.innerHTML = `<div class="podcast-live-script-head">Guion redactado (${activePod.script.length} intervenciones) — sintetizando audio ahora:</div>`;
      const listEl = document.createElement("div");
      listEl.className = "podcast-script-list";
      for (const turn of activePod.script) {
        const turnEl = document.createElement("div");
        turnEl.className = `podcast-turn ${turn.speaker === "B" ? "host-b" : "host-a"}`;
        turnEl.innerHTML = `<span class="podcast-turn-speaker">${escapeHtml(turn.name || turn.speaker)}</span><span>${escapeHtml(turn.text)}</span>`;
        listEl.appendChild(turnEl);
      }
      liveScript.appendChild(listEl);
      stage.appendChild(liveScript);
    }

    pane.appendChild(stage);
  } else {
    const sourceTilesHtml = buildSourceTilesHtml(readyRecs, readySrcs, st.selectedRecIds, st.selectedSrcIds, "pod");
    const styleCards = [
      { id: "conversation", title: "Conversación natural", desc: "Charla amena conectando ideas y ejemplos" },
      { id: "exam_review", title: "Repaso de examen", desc: "Preguntas clave, advertencias y síntesis rápida" },
      { id: "debate", title: "Debate analítico", desc: "Contrasta matices, pros y contras del tema" },
      { id: "step_by_step", title: "Paso a paso", desc: "Desglose estructurado desde las bases" },
    ];
    const focusSuggestions = [
      "Temas más difíciles",
      "Ejemplos cotidianos",
      "Preguntas tipo parcial",
      "Resumen ejecutivo",
    ];

    const studio = document.createElement("div");
    studio.className = "podcast-studio-shell";
    studio.innerHTML = `
      <div class="studio-composer-layout">
        <div class="studio-configurator">
          <div class="studio-head">
            <span class="studio-eyebrow">ESTUDIO DE AUDIO · KOKORO-82M LOCAL</span>
            <h2>Produce un episodio a dos voces</h2>
            <p>Selecciona tus fuentes, define el tono editorial y genera un diálogo neuronal sin salir de tu equipo.</p>
          </div>

          <section class="studio-section">
            <div class="studio-section-head">
              <span class="studio-section-label">1. Material del episodio</span>
              <div class="studio-section-actions">
                <span class="studio-count-badge" data-pod-src-count></span>
                <button type="button" class="studio-link-btn" data-pod-select-all="1">Todas</button>
                <button type="button" class="studio-link-btn" data-pod-select-all="0">Ninguna</button>
              </div>
            </div>
            <div class="source-tiles-grid">${sourceTilesHtml}</div>
          </section>

          <section class="studio-section">
            <div class="studio-section-head">
              <span class="studio-section-label">2. Estilo editorial del programa</span>
            </div>
            <div class="studio-cards-2x2" data-pod-group="style">
              ${styleCards.map((c) => `
                <button type="button" class="studio-option-card" data-val="${c.id}">
                  <span class="studio-card-top">
                    <span class="studio-card-icon">${STUDIO_ICONS[c.id]}</span>
                    <span class="studio-card-dot"></span>
                  </span>
                  <span class="studio-card-title">${c.title}</span>
                  <span class="studio-card-desc">${c.desc}</span>
                </button>`).join("")}
            </div>
          </section>

          <section class="studio-section">
            <div class="studio-section-head">
              <span class="studio-section-label">3. Dirección y enfoque del guion (opcional)</span>
            </div>
            <textarea class="podcast-focus-input" placeholder="Ej. Comparen las fórmulas del pizarrón con la clase grabada, aclaren los puntos más confusos y den ejemplos concretos...">${escapeHtml(st.focus)}</textarea>
            <div class="studio-focus-chips">
              ${focusSuggestions.map((label) => `<button type="button" class="focus-preset-chip" data-pod-focus-chip="${escapeHtml(label)}">+ ${escapeHtml(label)}</button>`).join("")}
            </div>
          </section>
        </div>

        <aside class="studio-summary-deck podcast-deck obsidian-card">
          <div class="deck-top-row">
            <span class="deck-badge">CABINA DE EMISIÓN</span>
            <div class="deck-static-bars" aria-hidden="true">
              <span style="height:36%"></span><span style="height:72%"></span><span style="height:100%"></span><span style="height:56%"></span><span style="height:82%"></span><span style="height:44%"></span>
            </div>
          </div>

          <div class="deck-cast-box">
            <div class="deck-sublabel">Locutores en cabina</div>
            <div class="deck-cast-row" data-pod-hosts></div>
          </div>

          <div class="deck-control-block">
            <div class="deck-sublabel">Idioma del episodio</div>
            <div class="seg-control-track dark-track" data-pod-group="language">
              <button type="button" class="seg-control-item" data-val="es">Español</button>
              <button type="button" class="seg-control-item" data-val="en">English</button>
            </div>
          </div>

          <div class="deck-control-block">
            <div class="deck-sublabel">Duración objetivo</div>
            <div class="seg-control-track dark-track" data-pod-group="length">
              <button type="button" class="seg-control-item" data-val="short">Corto (~2m)</button>
              <button type="button" class="seg-control-item" data-val="medium">Medio (~4m)</button>
              <button type="button" class="seg-control-item" data-val="long">Largo (~7m)</button>
            </div>
          </div>

          <div class="deck-spec-list">
            <div class="deck-spec-row"><span>Fuentes</span><strong data-pod-deck-sources></strong></div>
            <div class="deck-spec-row"><span>Formato</span><strong data-pod-deck-style></strong></div>
            <div class="deck-spec-row"><span>Duración</span><strong data-pod-deck-len></strong></div>
          </div>

          <p class="deck-exclusive-note">Modo exclusivo local: libera VRAM al sintetizar y permite cancelar en vivo.</p>
          <p class="deck-warning hidden" data-pod-warn>Selecciona al menos una fuente para grabar el episodio.</p>

          <button type="button" class="quiz-start-btn pod-start-btn">
            <span>Grabar podcast local</span>
            <span class="cta-circle" aria-hidden="true">→</span>
          </button>
        </aside>
      </div>
    `;

    studio.querySelectorAll(".source-tile").forEach((tile) => {
      tile.onclick = () => {
        const kind = tile.dataset.podKind;
        const id = tile.dataset.podItem;
        const targetSet = kind === "rec" ? st.selectedRecIds : st.selectedSrcIds;
        if (targetSet.has(id)) targetSet.delete(id);
        else targetSet.add(id);
        syncPodcastSetupDOM(pane, session, st);
      };
    });

    studio.querySelectorAll("[data-pod-select-all]").forEach((btn) => {
      btn.onclick = () => {
        const selectAll = btn.dataset.podSelectAll === "1";
        st.selectedRecIds.clear();
        st.selectedSrcIds.clear();
        if (selectAll) {
          readyRecs.forEach((r) => st.selectedRecIds.add(r.id));
          readySrcs.forEach((s) => st.selectedSrcIds.add(s.id));
        }
        syncPodcastSetupDOM(pane, session, st);
      };
    });

    const focusEl = studio.querySelector(".podcast-focus-input");
    focusEl.addEventListener("input", () => {
      st.focus = focusEl.value;
    });
    studio.querySelectorAll("[data-pod-focus-chip]").forEach((chip) => {
      chip.onclick = () => {
        const phrase = chip.dataset.podFocusChip;
        const cur = focusEl.value.trim();
        focusEl.value = cur ? `${cur}, ${phrase.toLowerCase()}` : phrase;
        st.focus = focusEl.value;
        focusEl.focus();
      };
    });

    ["style", "length", "language"].forEach((group) => {
      studio.querySelectorAll(`[data-pod-group="${group}"] [data-val]`).forEach((b) => {
        b.onclick = () => {
          st[group] = b.dataset.val;
          syncPodcastSetupDOM(pane, session, st);
        };
      });
    });

    studio.querySelector(".pod-start-btn").onclick = () => generatePodcastForSession(session);
    pane.appendChild(studio);
    syncPodcastSetupDOM(pane, session, st);
  }

  // Completed / Errored Episodes
  const finishedPods = podcasts.filter((p) => p.status === "ready" || p.status === "error");
  for (const pod of finishedPods) {
    const ep = document.createElement("div");
    ep.className = "podcast-episode-card";
    ep.dataset.podId = pod.id;
    ep.dataset.durationSec = String(pod.duration_sec || 0);

    const styleNames = {
      conversation: "Conversación",
      exam_review: "Repaso examen",
      debate: "Debate",
      step_by_step: "Paso a paso",
    };

    if (pod.status === "ready") {
      const audio = getOrCreatePodcastAudio(pod);
      const top = document.createElement("div");
      top.className = "podcast-episode-top";

      const playBtn = document.createElement("button");
      playBtn.type = "button";
      playBtn.className = "play-circle";
      playBtn.innerHTML = audio.paused ? ICON_PLAY : ICON_PAUSE;
      playBtn.onclick = () => {
        if (audio.paused) {
          stopAllOtherAudio(pod.id);
          audio.play().catch((err) => showToast(`No se pudo reproducir: ${err.message}`, true));
        } else {
          audio.pause();
        }
        syncPodcastAudioCardDOM(pod.id);
      };

      const meta = document.createElement("div");
      meta.className = "podcast-episode-meta";
      const dur = Number.isFinite(audio.duration) && audio.duration > 0 ? audio.duration : (pod.duration_sec || 0);
      meta.innerHTML = `
        <div class="podcast-episode-title">${escapeHtml(pod.title)}</div>
        <div class="podcast-episode-sub">
          <span class="pod-time">${formatDuration(audio.currentTime || 0)} / ${formatDuration(dur)}</span>
          · ${escapeHtml(styleNames[pod.style] || pod.style)} · ${(pod.language || "es").toUpperCase()}
          · ${escapeHtml(pod.tts_engine || "piper-onnx")}
        </div>
      `;

      const actions = document.createElement("div");
      actions.className = "podcast-episode-actions";

      const speedBtn = document.createElement("button");
      speedBtn.type = "button";
      speedBtn.className = "pill-btn outline xs pod-speed-btn";
      speedBtn.textContent = `${audio.playbackRate || 1}x`;
      speedBtn.onclick = () => {
        const speeds = [1, 1.25, 1.5, 2];
        audio.playbackRate = speeds[(speeds.indexOf(audio.playbackRate) + 1) % speeds.length];
        syncPodcastAudioCardDOM(pod.id);
      };

      const dlLink = document.createElement("a");
      dlLink.className = "pill-btn outline xs";
      dlLink.href = `/api/podcasts/${pod.id}/audio`;
      dlLink.download = `${pod.title || "podcast"}.wav`;
      dlLink.textContent = "Descargar .wav";
      dlLink.style.textDecoration = "none";

      const delBtn = document.createElement("button");
      delBtn.type = "button";
      delBtn.className = "pill-btn danger-soft xs";
      delBtn.textContent = "Eliminar";
      delBtn.onclick = () => {
        openConfirm("Eliminar podcast", `¿Eliminar "${pod.title}"?`, async () => {
          audio.pause();
          podcastAudioPlayers.delete(pod.id);
          await api(`/api/sessions/${session.id}/podcasts/${pod.id}`, { method: "DELETE" });
          await selectSession(session.id);
          showToast("Podcast eliminado.");
        });
      };

      actions.appendChild(speedBtn);
      actions.appendChild(dlLink);
      actions.appendChild(delBtn);

      top.appendChild(playBtn);
      top.appendChild(meta);
      top.appendChild(actions);
      ep.appendChild(top);

      const scrubWrap = document.createElement("div");
      scrubWrap.className = "scrubber-wrap";
      const track = document.createElement("div");
      track.className = "scrubber-track";
      const fill = document.createElement("div");
      fill.className = "scrubber-fill";
      const pct = dur > 0 ? Math.min(100, ((audio.currentTime || 0) / dur) * 100) : 0;
      fill.style.width = `${pct}%`;
      track.appendChild(fill);
      track.onclick = (e) => {
        const rect = track.getBoundingClientRect();
        const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        const targetDur = Number.isFinite(audio.duration) && audio.duration > 0 ? audio.duration : (pod.duration_sec || 0);
        if (targetDur > 0) {
          audio.currentTime = ratio * targetDur;
          syncPodcastAudioCardDOM(pod.id);
        }
      };
      scrubWrap.appendChild(track);
      ep.appendChild(scrubWrap);
    } else {
      ep.innerHTML = `
        <div class="podcast-episode-top">
          <div class="podcast-episode-meta">
            <div class="podcast-episode-title">${escapeHtml(pod.title)}</div>
            <div class="podcast-episode-sub" style="color:#ef4444;">Error: ${escapeHtml(pod.error_message || "Falló la generación.")}</div>
          </div>
          <button type="button" class="pill-btn danger-soft xs pod-err-del">Eliminar</button>
        </div>
      `;
      ep.querySelector(".pod-err-del").onclick = async () => {
        await api(`/api/sessions/${session.id}/podcasts/${pod.id}`, { method: "DELETE" });
        await selectSession(session.id);
      };
    }

    if (Array.isArray(pod.script) && pod.script.length > 0) {
      const details = document.createElement("details");
      details.className = "podcast-script-details";
      const sum = document.createElement("summary");
      sum.textContent = `Ver guion del episodio (${pod.script.length} intervenciones)`;
      const listEl = document.createElement("div");
      listEl.className = "podcast-script-list";
      for (const turn of pod.script) {
        const turnEl = document.createElement("div");
        turnEl.className = `podcast-turn ${turn.speaker === "B" ? "host-b" : "host-a"}`;
        turnEl.innerHTML = `<span class="podcast-turn-speaker">${escapeHtml(turn.name || turn.speaker)}</span><span>${escapeHtml(turn.text)}</span>`;
        listEl.appendChild(turnEl);
      }
      details.appendChild(sum);
      details.appendChild(listEl);
      ep.appendChild(details);
    }

    pane.appendChild(ep);
  }
}

// --- Upload Recordings & Sources ---

async function uploadRecordingFiles(sessionId, files) {
  const body = new FormData();
  files.forEach((file) => body.append("files", file));
  return api(`/api/sessions/${sessionId}/recordings`, { method: "POST", body });
}

function getCaptureButtonLabel(mode, isRecording) {
  if (isRecording) return "Detener grabación";
  if (mode === "system") return "Grabar audio del equipo";
  if (mode === "both") return "Grabar equipo + micrófono";
  return "Grabar con micrófono";
}

function setCaptureLevel(level) {
  const fill = document.getElementById("capture-level-fill");
  if (!fill) return;
  const pct = Math.round(Math.min(1, Math.max(0, Number(level) || 0)) * 100);
  fill.style.width = `${pct}%`;
}

function syncCaptureModeBar() {
  const bar = document.getElementById("capture-mode-bar");
  if (!bar) return;
  const locked = Boolean(microphoneCapture) || Boolean(systemCaptureState) || state.podcastBusy;
  const nativeAvail = Boolean(state.systemStatus?.system_audio_available);
  bar.querySelectorAll(".capture-mode-btn").forEach((btn) => {
    const mode = btn.dataset.captureMode;
    const isActive = state.captureMode === mode;
    btn.classList.toggle("active", isActive);
    btn.setAttribute("aria-pressed", String(isActive));
    btn.disabled = locked;
    if (mode !== "mic") {
      btn.title = nativeAvail
        ? (mode === "system"
          ? "Captura nativa de audio del equipo (WASAPI, sin diálogo)"
          : "Combina audio del equipo y micrófono (WASAPI)")
        : "Captura audio de pantalla/pestaña mediante el selector del navegador";
    }
  });
}

function syncMicrophoneControls() {
  const btn = document.getElementById("btn-microphone");
  const browserRec = microphoneCapture?.recorder?.state === "recording";
  const sysRec = Boolean(systemCaptureState?.recording);
  const recording = browserRec || sysRec;
  const busyStarting = (Boolean(microphoneCapture) || Boolean(systemCaptureState)) && !recording;
  btn.disabled = !recording && (busyStarting || state.podcastBusy);
  btn.setAttribute("aria-pressed", String(recording));
  document.getElementById("microphone-label").textContent = getCaptureButtonLabel(state.captureMode, recording);
  document.getElementById("microphone-live").disabled = Boolean(microphoneCapture) || Boolean(systemCaptureState) || state.podcastBusy;
  document.getElementById("microphone-time").classList.toggle("hidden", !recording);
  document.getElementById("capture-level-wrap")?.classList.toggle("hidden", !recording);
  if (!recording) setCaptureLevel(0);
  syncCaptureModeBar();
}

function syncMicrophonePreview(session) {
  if (microphonePreview?.sessionId === session?.id && microphonePreview?.recordingId) {
    const recording = session.recordings?.find((item) => item.id === microphonePreview.recordingId);
    if (recording?.transcript_text) {
      state.transcriptRecFilter ||= recording.id;
      microphonePreview = null;
    } else if (recording?.status === "ready" || recording?.status === "error") {
      document.getElementById("microphone-live-status").textContent =
        recording.error_message || "La transcripción final quedó sin texto.";
    }
  }
  document.getElementById("microphone-live-panel").classList.toggle(
    "hidden", !microphonePreview || microphonePreview.sessionId !== session?.id,
  );
}

async function transcribeMicrophoneFragment(capture, parts, startSec) {
  if (capture.previewRequest || capture.recorder.state !== "recording" || !parts.length) return;

  const controller = new AbortController();
  capture.previewRequest = controller;
  const status = document.getElementById("microphone-live-status");
  const preview = document.getElementById("microphone-preview");
  const body = new FormData();
  body.append("file", new Blob(parts, { type: parts[0].type }), `fragmento.${capture.extension}`);
  try {
    const { text } = await api("/api/recordings/live-transcript", {
      method: "POST",
      body,
      signal: controller.signal,
    });
    if (controller.signal.aborted) return;
    if (text) {
      const follow = preview.scrollHeight - preview.scrollTop - preview.clientHeight <= 24;
      const row = document.createElement("div");
      row.className = "segment-row";
      const stamp = document.createElement("span");
      stamp.className = "timestamp-btn";
      stamp.textContent = formatDuration(startSec);
      const words = document.createElement("div");
      words.className = "segment-text";
      words.textContent = text;
      row.append(stamp, words);
      preview.append(row);
      if (follow) preview.scrollTop = preview.scrollHeight;
    }
    status.textContent = preview.textContent ? "Grabando…" : "Esperando voz…";
  } catch (err) {
    if (!controller.signal.aborted) status.textContent = err.message;
  } finally {
    capture.previewRequest = null;
  }
}

function startMicrophoneFragment(capture) {
  if (capture.recorder.state !== "recording") return;

  try {
    const recorder = new MediaRecorder(capture.stream, { mimeType: capture.recorder.mimeType });
    const startSec = (performance.now() - capture.startedAt) / 1000;
    const parts = [];
    capture.previewRecorder = recorder;
    recorder.ondataavailable = ({ data }) => { if (data.size) parts.push(data); };
    recorder.onstop = () => transcribeMicrophoneFragment(capture, parts, startSec);
    recorder.start();
    capture.previewTimer = setTimeout(() => {
      if (recorder.state === "recording") recorder.stop();
      startMicrophoneFragment(capture);
    }, capture.previewIntervalMs);
  } catch (_) {
    document.getElementById("microphone-live-status").textContent =
      "La vista previa no está disponible. El audio se transcribirá al detener la grabación.";
  }
}

async function saveMicrophoneCapture(capture, parts) {
  clearInterval(capture.clock);
  clearTimeout(capture.previewTimer);
  if (capture.vuRaf) cancelAnimationFrame(capture.vuRaf);
  capture.previewRequest?.abort();
  if (capture.previewRecorder?.state === "recording") capture.previewRecorder.stop();
  capture.extraStreams?.forEach((s) => s.getTracks().forEach((track) => track.stop()));
  if (capture.audioCtx) {
    try { await capture.audioCtx.close(); } catch (_) {}
  }
  capture.stream?.getTracks().forEach((track) => track.stop());
  syncMicrophoneControls();

  const status = document.getElementById("microphone-status");
  const liveStatus = capture.live ? document.getElementById("microphone-live-status") : null;
  let file;
  try {
    if (!parts.length) {
      status.textContent = "No se recibió audio de la captura.";
      if (liveStatus) liveStatus.textContent = status.textContent;
      return;
    }
    const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
    const prefix = capture.mode === "both" ? "Equipo-Mic" : (capture.mode === "system" ? "Audio-equipo" : "Micrófono");
    file = new File(parts, `${prefix}-${stamp}.${capture.extension}`, { type: parts[0].type });
    status.textContent = "Guardando grabación…";
    if (liveStatus) liveStatus.textContent = status.textContent;
    const [recording] = await uploadRecordingFiles(capture.sessionId, [file]);
    status.textContent = "Grabación guardada. Transcripción en cola.";
    if (liveStatus) {
      microphonePreview.recordingId = recording.id;
      liveStatus.textContent = "Preparando la transcripción completa…";
    }
  } catch (err) {
    status.textContent = err.message;
    if (liveStatus) liveStatus.textContent = err.message;
    if (file) {
      const download = document.getElementById("microphone-download");
      if (download.hasAttribute("href")) URL.revokeObjectURL(download.href);
      download.href = URL.createObjectURL(file);
      download.download = file.name;
      download.classList.remove("hidden");
    }
    return;
  } finally {
    microphoneCapture = null;
    syncMicrophoneControls();
  }

  try {
    if (state.activeSessionId === capture.sessionId) {
      await selectSession(capture.sessionId);
      if (state.activeSessionId === capture.sessionId && !state.podcastBusy) switchTab("transcript");
    }
    await loadSessions(false);
  } catch (err) {
    showToast(err.message, true);
  }
}

function attachBrowserVuMeter(capture, stream) {
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;
    const ctx = capture.audioCtx || new AudioCtx();
    capture.audioCtx = ctx;
    const source = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 512;
    source.connect(analyser);
    const data = new Uint8Array(analyser.fftSize);
    let smoothed = 0;
    const tick = () => {
      if (!microphoneCapture || microphoneCapture !== capture) return;
      analyser.getByteTimeDomainData(data);
      let sum = 0;
      for (let i = 0; i < data.length; i += 1) {
        const v = (data[i] - 128) / 128;
        sum += v * v;
      }
      const rms = Math.sqrt(sum / data.length);
      const norm = Math.min(1, Math.sqrt(rms * 4.2));
      smoothed = smoothed * 0.55 + norm * 0.45;
      setCaptureLevel(smoothed);
      capture.vuRaf = requestAnimationFrame(tick);
    };
    capture.vuRaf = requestAnimationFrame(tick);
  } catch (_) {}
}

async function pollSystemCaptureLive(capture) {
  if (!systemCaptureState || systemCaptureState !== capture || !capture.recording) return;
  if (!capture.liveInFlight) {
    capture.liveInFlight = true;
    const statusEl = document.getElementById("microphone-live-status");
    const preview = document.getElementById("microphone-preview");
    try {
      const { text } = await api("/api/recordings/system-capture/live-transcript", { method: "POST" });
      if (systemCaptureState === capture && capture.recording && text) {
        const clean = text.trim();
        if (clean && clean !== capture.lastLiveSnippet) {
          capture.lastLiveSnippet = clean;
          const follow = preview.scrollHeight - preview.scrollTop - preview.clientHeight <= 24;
          const row = document.createElement("div");
          row.className = "segment-row";
          const stamp = document.createElement("span");
          stamp.className = "timestamp-btn";
          stamp.textContent = document.getElementById("microphone-time").textContent || "00:00";
          const words = document.createElement("div");
          words.className = "segment-text";
          words.textContent = clean;
          row.append(stamp, words);
          preview.append(row);
          if (follow) preview.scrollTop = preview.scrollHeight;
        }
        statusEl.textContent = preview.textContent ? "Grabando audio del equipo…" : "Escuchando audio…";
      }
    } catch (err) {
      if (systemCaptureState === capture && capture.recording) {
        statusEl.textContent = err.message;
      }
    } finally {
      capture.liveInFlight = false;
    }
  }
  if (systemCaptureState === capture && capture.recording) {
    capture.liveTimer = setTimeout(() => pollSystemCaptureLive(capture), capture.previewIntervalMs);
  }
}

async function stopNativeSystemCapture() {
  const capture = systemCaptureState;
  if (!capture) return;
  capture.recording = false;
  clearInterval(capture.statusTimer);
  clearTimeout(capture.liveTimer);
  const status = document.getElementById("microphone-status");
  const liveStatus = capture.live ? document.getElementById("microphone-live-status") : null;
  status.textContent = "Finalizando y guardando audio…";
  if (liveStatus) liveStatus.textContent = status.textContent;
  syncMicrophoneControls();

  try {
    const res = await api("/api/recordings/system-capture/stop", { method: "POST" });
    status.textContent = "Audio del equipo guardado. Transcripción en cola.";
    if (liveStatus && res.recording?.id) {
      microphonePreview.recordingId = res.recording.id;
      liveStatus.textContent = "Preparando la transcripción completa…";
    }
  } catch (err) {
    status.textContent = err.message;
    if (liveStatus) liveStatus.textContent = err.message;
    showToast(err.message, true);
  } finally {
    systemCaptureState = null;
    syncMicrophoneControls();
  }

  try {
    if (state.activeSessionId === capture.sessionId) {
      await selectSession(capture.sessionId);
      if (state.activeSessionId === capture.sessionId && !state.podcastBusy) switchTab("transcript");
    }
    await loadSessions(false);
  } catch (err) {
    showToast(err.message, true);
  }
}

async function startNativeSystemCapture() {
  const status = document.getElementById("microphone-status");
  const settings = state.systemStatus?.settings;
  const capture = {
    sessionId: state.activeSessionId,
    mode: state.captureMode,
    live: document.getElementById("microphone-live").checked,
    previewIntervalMs: settings?.whisper_model === PARAKEET_MODEL_ID && settings.whisper_device === "auto" ? 4500 : 8000,
    recording: false,
    statusTimer: null,
    liveTimer: null,
    liveInFlight: false,
    lastLiveSnippet: "",
  };
  systemCaptureState = capture;
  status.textContent = "Iniciando captura WASAPI del equipo…";
  syncMicrophoneControls();

  try {
    if (!capture.sessionId) {
      await createNewSession();
      capture.sessionId = state.activeSessionId;
    }
    if (!capture.sessionId || state.podcastBusy) throw new Error("Espera a que termine el podcast.");

    const started = await api(`/api/sessions/${capture.sessionId}/recordings/system-capture/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: capture.mode }),
    });
    capture.recording = true;
    const clock = document.getElementById("microphone-time");
    clock.textContent = "00:00";
    status.textContent = capture.live
      ? ""
      : `Capturando ${capture.mode === "both" ? "equipo + micrófono" : "audio del equipo"} (${started.device_name || "WASAPI"}).`;
    syncMicrophoneControls();

    capture.statusTimer = setInterval(async () => {
      if (!systemCaptureState || systemCaptureState !== capture || !capture.recording) return;
      try {
        const st = await api("/api/recordings/system-capture/status");
        if (!systemCaptureState || systemCaptureState !== capture) return;
        clock.textContent = formatElapsedClock((st.elapsed_sec || 0) * 1000);
        setCaptureLevel(st.level || 0);
      } catch (_) {}
    }, 180);

    if (capture.live) {
      microphonePreview = { sessionId: capture.sessionId };
      document.getElementById("microphone-preview").replaceChildren();
      document.getElementById("microphone-live-status").textContent = "Esperando el primer fragmento…";
      syncMicrophonePreview(state.activeSession);
      if (state.activeSessionId === capture.sessionId) {
        switchTab("transcript");
        document.getElementById("center-scroll").scrollTop = 0;
      }
      capture.liveTimer = setTimeout(() => pollSystemCaptureLive(capture), capture.previewIntervalMs);
    }
  } catch (err) {
    status.textContent = err.message;
    systemCaptureState = null;
    syncMicrophoneControls();
  }
}

async function toggleMicrophoneCapture() {
  if (systemCaptureState?.recording) {
    await stopNativeSystemCapture();
    return;
  }
  if (microphoneCapture?.recorder?.state === "recording") {
    microphoneCapture.recorder.stop();
    syncMicrophoneControls();
    return;
  }
  if (microphoneCapture || systemCaptureState || state.podcastBusy) return;

  if (state.captureMode !== "mic" && state.systemStatus?.system_audio_available) {
    await startNativeSystemCapture();
    return;
  }

  const status = document.getElementById("microphone-status");
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    status.textContent = "El navegador no permite grabar desde esta página.";
    return;
  }

  const settings = state.systemStatus?.settings;
  const capture = {
    sessionId: state.activeSessionId,
    mode: state.captureMode,
    live: document.getElementById("microphone-live").checked,
    previewIntervalMs: settings?.whisper_model === PARAKEET_MODEL_ID && settings.whisper_device === "auto" ? 4000 : 8000,
    extraStreams: [],
  };
  microphoneCapture = capture;
  status.textContent = capture.mode === "mic"
    ? "Esperando permiso del micrófono…"
    : "Selecciona la pestaña o pantalla con audio en el navegador…";
  syncMicrophoneControls();

  try {
    if (capture.mode === "mic") {
      capture.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } else {
      const displayStream = await navigator.mediaDevices.getDisplayMedia({ audio: true, video: true });
      displayStream.getVideoTracks().forEach((t) => t.stop());
      if (!displayStream.getAudioTracks().length) {
        throw new Error("No se compartió ninguna pista de audio del equipo.");
      }
      capture.extraStreams.push(displayStream);
      if (capture.mode === "both") {
        const micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        capture.extraStreams.push(micStream);
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        const ctx = new AudioCtx();
        capture.audioCtx = ctx;
        const dest = ctx.createMediaStreamDestination();
        ctx.createMediaStreamSource(displayStream).connect(dest);
        ctx.createMediaStreamSource(micStream).connect(dest);
        capture.stream = dest.stream;
      } else {
        capture.stream = new MediaStream(displayStream.getAudioTracks());
      }
    }

    if (!capture.sessionId) {
      await createNewSession();
      capture.sessionId = state.activeSessionId;
    }
    if (!capture.sessionId || state.podcastBusy) throw new Error("Espera a que termine el podcast.");

    const mimeType = ["audio/webm;codecs=opus", "audio/mp4", "audio/ogg;codecs=opus"]
      .find((type) => MediaRecorder.isTypeSupported(type));
    const recorder = new MediaRecorder(capture.stream, mimeType ? { mimeType } : undefined);
    const parts = [];
    capture.recorder = recorder;
    recorder.ondataavailable = ({ data }) => { if (data.size) parts.push(data); };
    recorder.onstop = () => saveMicrophoneCapture(capture, parts);
    recorder.onerror = () => showToast("Se interrumpió la grabación de audio.", true);
    recorder.start();
    const container = recorder.mimeType.split(";")[0];
    capture.extension = { "audio/mp4": "m4a", "video/mp4": "mp4", "audio/ogg": "ogg" }[container] || "webm";

    capture.startedAt = performance.now();
    const clock = document.getElementById("microphone-time");
    clock.textContent = "00:00";
    capture.clock = setInterval(() => { clock.textContent = formatElapsedClock(performance.now() - capture.startedAt); }, 1000);
    attachBrowserVuMeter(capture, capture.stream);
    status.textContent = capture.live ? "" : "Se transcribirá al detener la grabación.";
    syncMicrophoneControls();
    if (capture.live) {
      microphonePreview = { sessionId: capture.sessionId };
      document.getElementById("microphone-preview").replaceChildren();
      document.getElementById("microphone-live-status").textContent = "Esperando el primer fragmento…";
      syncMicrophonePreview(state.activeSession);
      if (state.activeSessionId === capture.sessionId) {
        switchTab("transcript");
        document.getElementById("center-scroll").scrollTop = 0;
      }
      startMicrophoneFragment(capture);
    }
  } catch (err) {
    capture.extraStreams?.forEach((s) => s.getTracks().forEach((track) => track.stop()));
    if (capture.audioCtx) {
      try { await capture.audioCtx.close(); } catch (_) {}
    }
    capture.stream?.getTracks().forEach((track) => track.stop());
    status.textContent = err.name === "NotAllowedError" || err.name === "NotFoundError"
      ? "Permiso denegado o dispositivo de audio no disponible."
      : err.message;
    microphoneCapture = null;
    syncMicrophoneControls();
  }
}

async function uploadFiles(fileList) {
  if (state.podcastBusy) {
    showToast("Las cargas están pausadas mientras se genera el podcast.", true);
    return;
  }
  if (!fileList || fileList.length === 0) return;

  if (!state.activeSessionId) {
    await createNewSession();
  }

  const audioFiles = [];
  const sourceFiles = [];
  const rejected = [];

  for (const f of fileList) {
    const dotIdx = (f.name || "").lastIndexOf(".");
    const ext = dotIdx >= 0 ? f.name.slice(dotIdx).toLowerCase() : "";
    if (AUDIO_EXTENSIONS.has(ext)) {
      audioFiles.push(f);
    } else if (SOURCE_EXTENSIONS.has(ext)) {
      sourceFiles.push(f);
    } else {
      rejected.push(f.name);
    }
  }

  if (rejected.length > 0 && audioFiles.length === 0 && sourceFiles.length === 0) {
    showToast(`Formato no soportado: ${rejected.join(", ")}`, true);
    return;
  }

  try {
    const totalCount = audioFiles.length + sourceFiles.length;
    showToast(`Indexando ${totalCount} fuente(s) localmente...`);

    if (audioFiles.length > 0) {
      await uploadRecordingFiles(state.activeSessionId, audioFiles);
    }

    if (sourceFiles.length > 0) {
      const fdSources = new FormData();
      sourceFiles.forEach((f) => fdSources.append("files", f));
      await api(`/api/sessions/${state.activeSessionId}/sources`, {
        method: "POST",
        body: fdSources,
      });
    }

    if (rejected.length > 0) {
      showToast(`Omitidos por formato no soportado: ${rejected.join(", ")}`, true);
    }

    await selectSession(state.activeSessionId);
    await loadSessions(false);
  } catch (err) {
    showToast(err.message, true);
  }
}

// --- Dynamic AI Quick Prompts ---

async function ensureDynamicSuggestions(session) {
  if (!session) return;
  const settings = state.systemStatus?.settings;
  if (settings?.auto_summarize === false || settings?.eco_mode) return;
  const readyCount =
    (session.recordings || []).filter((r) => r.status === "ready" && r.enabled !== false).length +
    (session.sources || []).filter((s) => s.status === "ready" && s.enabled !== false).length;
  if (readyCount === 0) return;

  const isStillProcessing =
    (session.recordings || []).some((r) => ["queued", "preparing", "transcribing", "summarizing"].includes(r.status)) ||
    (session.sources || []).some((s) => ["queued", "extracting", "indexing", "summarizing"].includes(s.status));
  if (isStillProcessing || state.podcastBusy) return;

  if (Array.isArray(session.suggested_prompts) && session.suggested_prompts.length > 0) return;
  const attemptKey = `${session.id}:${readyCount}`;
  if (state.attemptedSuggestionsKeys.has(attemptKey)) return;
  if (state.fetchingSuggestionsFor.has(session.id) || state.isStreaming) return;

  state.attemptedSuggestionsKeys.add(attemptKey);
  state.fetchingSuggestionsFor.add(session.id);
  renderQuickPrompts(session);

  try {
    const res = await api(`/api/sessions/${session.id}/suggestions`, { method: "POST" });
    if (state.activeSession && state.activeSession.id === session.id) {
      state.activeSession.suggested_prompts = res.suggested_prompts || [];
      renderQuickPrompts(state.activeSession);
    }
  } catch (_) {
    if (state.activeSession && state.activeSession.id === session.id) {
      renderQuickPrompts(state.activeSession);
    }
  } finally {
    state.fetchingSuggestionsFor.delete(session.id);
    if (state.activeSession && state.activeSession.id === session.id) {
      renderQuickPrompts(state.activeSession);
    }
  }
}

function renderQuickPrompts(session) {
  const quick = document.getElementById("quick-prompts");
  const readyCount =
    (session?.recordings || []).filter((r) => r.status === "ready" && r.enabled !== false).length +
    (session?.sources || []).filter((s) => s.status === "ready" && s.enabled !== false).length;
  const isFetching = Boolean(session && state.fetchingSuggestionsFor.has(session.id));

  if (!session || readyCount === 0) {
    if (!memoChanged("quick-prompts", "hidden")) return;
    quick.classList.add("hidden");
    quick.innerHTML = "";
    return;
  }

  if (isFetching) {
    if (!memoChanged("quick-prompts", `${session.id}:fetching`)) return;
    quick.classList.remove("hidden");
    quick.innerHTML = `
      <div class="prompt-chip-skel"></div>
      <div class="prompt-chip-skel"></div>
      <div class="prompt-chip-skel"></div>
    `;
    return;
  }

  const suggestions = (session.suggested_prompts && session.suggested_prompts.length > 0)
    ? session.suggested_prompts
    : [
        {
          label: "Puntos clave",
          prompt: "¿Cuáles son los puntos más importantes de las fuentes activas?",
        },
        {
          label: "Ponme a prueba",
          prompt: "Hazme 5 preguntas tipo examen basadas en las fuentes de estudio.",
        },
        {
          label: readyCount >= 2 ? "Conectar fuentes" : "Tareas y fechas",
          prompt: readyCount >= 2
            ? "¿Qué conceptos se complementan o cambian entre las distintas fuentes?"
            : "¿Qué tareas, fechas o acuerdos importantes se mencionaron?",
        },
      ];

  const sig = `${session.id}|${readyCount}|${JSON.stringify(suggestions)}`;
  if (!memoChanged("quick-prompts", sig)) return;

  quick.classList.remove("hidden");
  quick.innerHTML = "";
  suggestions.forEach((item, idx) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `prompt-chip ${idx === 0 ? "featured" : ""}`;
    btn.style.animationDelay = `${idx * 55}ms`;
    btn.title = item.prompt;
    btn.innerHTML = `${idx === 0 ? ICON_SPARKLE : ""}<span>${escapeHtml(item.label)}</span>`;
    btn.onclick = () => sendChatQuestion(item.prompt);
    quick.appendChild(btn);
  });
}

// --- Chat & Clear History ---

function updateChatControls(session) {
  const messages = session?.messages || [];
  const badge = document.getElementById("chat-badge-count");
  const clearBtn = document.getElementById("btn-clear-chat");

  if (messages.length > 0) {
    badge.textContent = String(messages.length);
    badge.classList.remove("hidden");
  } else {
    badge.classList.add("hidden");
  }

  clearBtn.disabled = messages.length === 0;
  renderQuickPrompts(session);
  requestAnimationFrame(updateTabIndicator);
}

function renderChatMessages(messages) {
  const container = document.getElementById("chat-messages");
  container.innerHTML = "";

  if (!messages || messages.length === 0) {
    container.innerHTML = `
      <div class="chat-placeholder">
        <h3>Pregunta a ${escapeHtml(getActiveModelLabel())}</h3>
        <p>Consulta cualquier detalle sobre las grabaciones de esta conversación. Puedes adjuntar más audios en cualquier momento para ampliar el contexto.</p>
      </div>
    `;
    return;
  }

  for (const msg of messages) {
    appendChatBubble(msg.role, msg.content);
  }
  const scroll = document.getElementById("center-scroll");
  scroll.scrollTop = scroll.scrollHeight;
}

function appendChatBubble(role, content, isThinkingPlaceholder = false) {
  const container = document.getElementById("chat-messages");
  const placeholder = container.querySelector(".chat-placeholder");
  if (placeholder) placeholder.remove();

  const bubble = document.createElement("div");
  bubble.className = `msg-bubble ${role}`;
  if (role === "assistant") {
    if (isThinkingPlaceholder) {
      bubble.innerHTML = `
        <div class="thinking-indicator">
          <div class="think-dots" aria-hidden="true"><span></span><span></span><span></span></div>
          <span class="think-label">${escapeHtml(getActiveModelLabel())} está pensando...</span>
          <button type="button" class="inline-stop-btn" data-stop-scope="chat" title="Detener respuesta"><span class="stop-sq" aria-hidden="true"></span> Detener</button>
        </div>
      `;
    } else {
      bubble.classList.add("md");
      bubble.innerHTML = renderMarkdown(content || "...", true);
    }
  } else {
    bubble.textContent = content;
  }
  container.appendChild(bubble);

  const scroll = document.getElementById("center-scroll");
  scroll.scrollTop = scroll.scrollHeight;
  return bubble;
}

function createEcoProgressController(parentEl, phaseLabel = "Leyendo fuentes") {
  const settings = state.systemStatus?.settings || {};
  if (!settings.eco_mode || !parentEl) return null;

  const threads = Math.max(1, Number(settings.eco_cpu_threads || 2));
  const wrap = document.createElement("div");
  wrap.className = "eco-progress-wrap";
  const labelEl = document.createElement("div");
  labelEl.className = "eco-progress-label";
  const track = document.createElement("div");
  track.className = "eco-progress-track";
  const bar = document.createElement("div");
  bar.className = "eco-progress-bar";
  track.appendChild(bar);
  wrap.appendChild(labelEl);
  wrap.appendChild(track);
  parentEl.prepend(wrap);

  const startedAt = Date.now();
  let chunks = 0;
  let pct = 5;
  let currentPhase = phaseLabel;

  const renderProgress = () => {
    const sec = Math.floor((Date.now() - startedAt) / 1000);
    const timeStr = `${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, "0")}`;
    bar.style.width = `${pct}%`;
    labelEl.textContent = `${currentPhase} · Modo Eco (${threads}h) · ${pct}% · ${timeStr}`;
  };
  renderProgress();
  const timer = setInterval(() => {
    if (chunks === 0 && pct < 35) {
      // Smoothly advance 5% -> 35% while Ollama evaluates the prompt context
      pct = Math.min(35, pct + Math.max(1, Math.round((35 - pct) * 0.18)));
      renderProgress();
    } else {
      renderProgress();
    }
  }, 500);

  return {
    onToken() {
      chunks += 1;
      currentPhase = "Generando respuesta";
      const genPct = Math.round(35 + 61 * (1 - Math.exp(-chunks / 140)));
      pct = Math.max(pct, Math.min(96, genPct));
      renderProgress();
    },
    stop() {
      clearInterval(timer);
      pct = 100;
      renderProgress();
      setTimeout(() => wrap.remove(), 220);
    },
  };
}

async function sendChatQuestion(questionText) {
  if (state.podcastBusy) {
    showToast("El chat está bloqueado mientras se genera el podcast.", true);
    return;
  }
  if (state.isStreaming || !state.activeSessionId) return;
  const question = (questionText || "").trim();
  if (!question) return;

  if (state.activeTab !== "chat") {
    switchTab("chat");
  }

  const controller = new AbortController();
  state.chatAbortController = controller;
  state.isStreaming = true;
  setChatSendButtonMode(true);
  syncGlobalStopButton();
  const inputEl = document.getElementById("chat-input");
  inputEl.value = "";
  inputEl.style.height = "auto";

  appendChatBubble("user", question);
  const assistantBubble = appendChatBubble("assistant", "", true);
  const ecoProgress = createEcoProgressController(assistantBubble, "Leyendo fuentes");
  const scroll = document.getElementById("center-scroll");

  let accumulated = "";
  let thinkingAccum = "";
  let thinkingBoxEl = null;
  let thinkingContentEl = null;
  let answerContainerEl = null;
  let wasStopped = false;

  try {
    const resp = await fetch(`/api/sessions/${state.activeSessionId}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({ question }),
    });

    if (!resp.ok) {
      let detail = `Error HTTP ${resp.status}`;
      try {
        const errData = await resp.json();
        if (errData.detail) detail = errData.detail;
      } catch (_) {}
      throw new Error(detail);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const events = buffer.split("\n\n");
      buffer = events.pop() || "";

      for (const evt of events) {
        for (const line of evt.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          const payload = JSON.parse(line.slice(6));
          if (payload.type === "meta") {
            state.lastContextMode = payload.context_mode || "full";
            if (state.activeSession) renderWorkspaceSubtitle(state.activeSession);
            const lbl = assistantBubble.querySelector(".think-label");
            if (lbl && payload.thinking_enabled) {
              lbl.textContent = `${payload.model || getActiveModelLabel()} está razonando...`;
            }
          } else if (payload.type === "thinking") {
            thinkingAccum += payload.text;
            if (!thinkingBoxEl) {
              thinkingBoxEl = document.createElement("details");
              thinkingBoxEl.className = "thinking-box";
              thinkingBoxEl.open = true;
              const sum = document.createElement("summary");
              sum.textContent = "Razonamiento interno";
              thinkingContentEl = document.createElement("div");
              thinkingContentEl.className = "thinking-box-content";
              thinkingBoxEl.appendChild(sum);
              thinkingBoxEl.appendChild(thinkingContentEl);
              assistantBubble.prepend(thinkingBoxEl);
            }
            thinkingContentEl.textContent = thinkingAccum;
            thinkingContentEl.scrollTop = thinkingContentEl.scrollHeight;
            scroll.scrollTop = scroll.scrollHeight;
          } else if (payload.type === "token") {
            if (!answerContainerEl) {
              const indicator = assistantBubble.querySelector(".thinking-indicator");
              if (indicator) indicator.remove();
              if (thinkingBoxEl) thinkingBoxEl.open = false;
              answerContainerEl = document.createElement("div");
              answerContainerEl.className = "md";
              assistantBubble.appendChild(answerContainerEl);
            }
            ecoProgress?.onToken();
            accumulated += payload.text;
            answerContainerEl.innerHTML =
              renderMarkdown(accumulated, true) + '<span class="stream-cursor"></span>';
            scroll.scrollTop = scroll.scrollHeight;
          } else if (payload.type === "done") {
            ecoProgress?.stop();
            if (answerContainerEl) {
              answerContainerEl.innerHTML = renderMarkdown(accumulated, true);
            }
          } else if (payload.type === "stopped") {
            wasStopped = true;
            break;
          } else if (payload.type === "error") {
            throw new Error(payload.detail || "Error en la respuesta del modelo.");
          }
        }
        if (wasStopped) break;
      }
      if (wasStopped) break;
    }

    ecoProgress?.stop();
    if (wasStopped || controller.signal.aborted) {
      const indicator = assistantBubble.querySelector(".thinking-indicator");
      if (indicator) indicator.remove();
      if (thinkingBoxEl) thinkingBoxEl.open = false;
      if (accumulated.trim()) {
        if (!answerContainerEl) {
          answerContainerEl = document.createElement("div");
          answerContainerEl.className = "md";
          assistantBubble.appendChild(answerContainerEl);
        }
        answerContainerEl.innerHTML =
          renderMarkdown(accumulated, true) +
          `<div class="stopped-note">Generación detenida por el usuario.</div>`;
      } else {
        assistantBubble.innerHTML = `<div class="stopped-note">Respuesta detenida por el usuario.</div>`;
      }
      showToast("Respuesta detenida.");
    } else if (answerContainerEl) {
      answerContainerEl.innerHTML = renderMarkdown(accumulated, true);
    }

    const fresh = await api(`/api/sessions/${state.activeSessionId}`);
    state.activeSession = fresh;
    renderContextMetrics(fresh.context_metrics);
    updateChatControls(fresh);
    renderWorkspaceSubtitle(fresh);
    await loadSessions(false);
  } catch (err) {
    ecoProgress?.stop();
    if (controller.signal.aborted || isAbortError(err)) {
      const indicator = assistantBubble.querySelector(".thinking-indicator");
      if (indicator) indicator.remove();
      if (thinkingBoxEl) thinkingBoxEl.open = false;
      if (accumulated.trim()) {
        if (!answerContainerEl) {
          answerContainerEl = document.createElement("div");
          answerContainerEl.className = "md";
          assistantBubble.appendChild(answerContainerEl);
        }
        answerContainerEl.innerHTML =
          renderMarkdown(accumulated, true) +
          `<div class="stopped-note">Generación detenida por el usuario.</div>`;
      } else {
        assistantBubble.innerHTML = `<div class="stopped-note">Respuesta detenida por el usuario.</div>`;
      }
      showToast("Respuesta detenida.");
      if (state.activeSessionId) {
        api(`/api/sessions/${state.activeSessionId}`).then((fresh) => {
          state.activeSession = fresh;
          renderContextMetrics(fresh.context_metrics);
          updateChatControls(fresh);
        }).catch(() => {});
      }
    } else {
      assistantBubble.innerHTML = `<span style="color:#fca5a5;">${escapeHtml(err.message)}</span>`;
      showToast(err.message, true);
    }
  } finally {
    ecoProgress?.stop();
    if (state.chatAbortController === controller) {
      state.chatAbortController = null;
    }
    state.isStreaming = false;
    setChatSendButtonMode(false);
    syncGlobalStopButton();
  }
}

function handleClearChatHistory() {
  if (!state.activeSessionId) return;
  openConfirm(
    "Borrar historial de chat",
    "Se eliminarán únicamente las preguntas y respuestas del chat. Las grabaciones, transcripciones y resúmenes se conservarán intactos.",
    async () => {
      await api(`/api/sessions/${state.activeSessionId}/messages`, {
        method: "DELETE",
      });
      await selectSession(state.activeSessionId);
      await loadSessions(false);
      showToast("Historial de chat borrado. Tus audios y resúmenes siguen intactos.");
    }
  );
}

// --- Native Export Markdown ---

async function exportActiveSessionMarkdown() {
  if (!state.activeSessionId) return;
  try {
    const resp = await fetch(`/api/sessions/${state.activeSessionId}/export`);
    if (!resp.ok) {
      throw new Error(`No se pudo exportar (HTTP ${resp.status})`);
    }
    const content = await resp.text();
    const disp = resp.headers.get("content-disposition") || "";
    const match = disp.match(/filename="([^"]+)"/i);
    const filename = match ? match[1] : `${state.activeSession?.title || "transcripcion"}.md`;

    if (window.pywebview?.api?.save_markdown) {
      const res = await window.pywebview.api.save_markdown(filename, content);
      if (res?.saved) {
        showToast("Archivo Markdown guardado.");
      } else if (res?.error) {
        throw new Error(res.error);
      }
      return;
    }

    const blob = new Blob([content], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    showToast("Archivo Markdown descargado.");
  } catch (err) {
    showToast(err.message, true);
  }
}

// --- Settings Modal (Instant Open) ---

function populateSettingsFromStatus(status) {
  const settings = status?.settings || {};
  const ollamaModels = status?.ollama_models || [];
  const whisperModels = status?.whisper_models || [
    { id: "tiny", label: "Whisper Tiny (~0.3 GB VRAM, ultra rápido)" },
    { id: "base", label: "Whisper Base (~0.4 GB VRAM, ligero)" },
    { id: "small", label: "Whisper Small (~0.6 GB VRAM, muy rápido)" },
    { id: "medium", label: "Whisper Medium (~1.6 GB VRAM, equilibrado)" },
    { id: "large-v3-turbo", label: "Whisper Large-v3-Turbo (~1.8 GB VRAM, rápido y preciso)" },
    { id: "large-v3", label: "Whisper Large-v3 (~3.2 GB VRAM, máxima precisión clases/ruido)" },
  ];

  const ollamaSelect = document.getElementById("set-ollama-model");
  ollamaSelect.innerHTML = "";
  if (ollamaModels.length === 0) {
    const opt = document.createElement("option");
    opt.value = settings.ollama_model || "";
    opt.textContent = opt.value || "Sin modelos detectados en Ollama (instala uno con ollama pull)";
    ollamaSelect.appendChild(opt);
  } else {
    for (const m of ollamaModels) {
      const opt = document.createElement("option");
      opt.value = m.name;
      opt.textContent = `${m.name} (${m.size_gb} GB)`;
      ollamaSelect.appendChild(opt);
    }
  }
  ollamaSelect.value = settings.ollama_model || ollamaSelect.options[0]?.value || "";

  const whisperSelect = document.getElementById("set-whisper-model");
  whisperSelect.innerHTML = "";
  for (const w of whisperModels) {
    const opt = document.createElement("option");
    opt.value = w.id;
    const prefix = w.installed ? "✓ " : "↓ ";
    const sizeText = !w.installed && w.size_hint && !w.label.includes(w.size_hint) ? ` · ${w.size_hint}` : "";
    opt.textContent = `${prefix}${w.label}${sizeText}`;
    whisperSelect.appendChild(opt);
  }
  whisperSelect.value = settings.whisper_model || whisperSelect.options[0]?.value || "large-v3-turbo";

  const syncAsrModelActionUI = () => {
    const cur = whisperModels.find((m) => m.id === whisperSelect.value);
    const pill = document.getElementById("asr-model-status-pill");
    const btn = document.getElementById("btn-asr-model-action");
    if (!pill || !btn) return;
    const downloading = status?.downloading_whisper_model || null;
    if (!cur) {
      pill.textContent = "";
      btn.classList.add("hidden");
      return;
    }
    if (downloading) {
      pill.className = "asr-model-status-pill remote";
      pill.textContent = `Descargando ${downloading}…`;
      btn.classList.remove("hidden");
      btn.disabled = true;
      btn.textContent = "Descargando…";
      return;
    }
    btn.disabled = false;
    if (cur.installed) {
      pill.className = "asr-model-status-pill installed";
      pill.textContent = "✓ Descargado en disco";
      btn.classList.toggle("hidden", !cur.can_delete);
      btn.textContent = "Eliminar pesos";
      btn.dataset.asrAction = "delete";
    } else {
      pill.className = "asr-model-status-pill remote";
      const hint = cur.size_hint ? ` (${cur.size_hint})` : "";
      pill.textContent = `↓ Sin descargar${hint} · baja solo si lo eliges`;
      btn.classList.toggle("hidden", !cur.can_download);
      btn.textContent = "Descargar ahora";
      btn.dataset.asrAction = "download";
    }
  };

  const langSelect = document.getElementById("set-whisper-lang");
  const autoLangOpt = langSelect.querySelector('option[value="auto"]');
  langSelect.value = settings.whisper_language || "auto";
  whisperSelect.onchange = () => {
    document.getElementById("parakeet-language-hint").classList.toggle(
      "hidden", whisperSelect.value !== PARAKEET_MODEL_ID
    );
    const requiresLanguage = Boolean(whisperModels.find((model) => model.id === whisperSelect.value)?.requires_language);
    document.getElementById("nemotron-language-hint").classList.toggle("hidden", !requiresLanguage);
    autoLangOpt.disabled = requiresLanguage;
    if (requiresLanguage && langSelect.value === "auto") {
      langSelect.value = "es";
      refreshCustomSelect(langSelect.id);
    }
    syncAsrModelActionUI();
  };
  whisperSelect.onchange();

  const whisperDeviceSelect = document.getElementById("set-whisper-device");
  const hasCuda = Boolean(status?.hardware?.gpu_available);
  const autoOpt = whisperDeviceSelect.querySelector('option[value="auto"]');
  const cudaOpt = whisperDeviceSelect.querySelector('option[value="cuda"]');
  if (autoOpt) {
    const gpuBackend = status?.whisper_metal_supported ? "Metal" : hasCuda ? "CUDA" : "";
    autoOpt.textContent = gpuBackend
      ? `Automático (GPU ${gpuBackend} con respaldo CPU)`
      : "Automático (solo CPU en este equipo)";
  }
  if (cudaOpt) {
    cudaOpt.disabled = !hasCuda;
  }
  const desiredDevice = settings.whisper_device || "auto";
  whisperDeviceSelect.value = !hasCuda && desiredDevice === "cuda" ? "cpu" : desiredDevice;
  document.getElementById("set-num-ctx").value = String(settings.num_ctx || 8192);
  document.getElementById("set-idle-unload").value = String(settings.idle_unload_sec || 180);
  document.getElementById("set-classroom-enhance").checked = settings.classroom_audio_enhance !== false;
  document.getElementById("set-enable-diarization").checked = settings.enable_diarization !== false;
  document.getElementById("set-enable-thinking").checked = Boolean(settings.enable_thinking);
  document.getElementById("set-eco-mode").checked = Boolean(settings.eco_mode);
  document.getElementById("set-auto-summarize").checked = settings.auto_summarize !== false;

  const cpuCount = Math.max(1, Number(status?.hardware?.cpu_count || 8));
  const threadsInput = document.getElementById("set-eco-cpu-threads");
  threadsInput.max = String(cpuCount);
  threadsInput.value = String(Math.min(cpuCount, Math.max(1, Number(settings.eco_cpu_threads || 2))));
  syncEcoThreadsUI(cpuCount);
  syncAppearanceUI(
    state.appearancePreviewBackup ? state.accentColor : (settings.accent_color || state.accentColor),
    state.appearancePreviewBackup ? state.fontScale : (settings.font_scale || state.fontScale)
  );

  [
    "set-ollama-model",
    "set-whisper-model",
    "set-whisper-device",
    "set-whisper-lang",
    "set-num-ctx",
    "set-idle-unload",
  ].forEach(refreshCustomSelect);
}

function syncEcoThreadsUI(cpuCountOverride = null) {
  const ecoChecked = document.getElementById("set-eco-mode").checked;
  const field = document.getElementById("eco-threads-field");
  const input = document.getElementById("set-eco-cpu-threads");
  const badge = document.getElementById("eco-threads-badge");
  if (!field || !input || !badge) return;
  field.classList.toggle("hidden", !ecoChecked);
  const total = Math.max(1, Number(cpuCountOverride || input.max || state.systemStatus?.hardware?.cpu_count || 8));
  const val = Math.min(total, Math.max(1, Number(input.value || 2)));
  const pct = Math.round((val / total) * 100);
  badge.textContent = `${val} de ${total} (~${pct}% CPU)`;
}

function openSettingsModal() {
  if (state.podcastBusy) {
    showToast("Ajustes bloqueados mientras se genera el podcast en modo exclusivo.", true);
    return;
  }
  state.appearancePreviewBackup = { accentColor: state.accentColor, fontScale: state.fontScale };
  populateSettingsFromStatus(state.systemStatus || {});
  document.getElementById("btn-open-settings").classList.add("active");
  openModal("settings-modal");
  refreshSystemStatus({ syncSettingsModal: true });
}

function closeSettingsModal(restoreBackup = true) {
  closeActiveDropdown();
  if (restoreBackup && state.appearancePreviewBackup) {
    applyAppearance(state.appearancePreviewBackup, true);
  }
  state.appearancePreviewBackup = null;
  document.getElementById("btn-open-settings").classList.remove("active");
  closeModal("settings-modal");
}

async function saveSettings(e) {
  e.preventDefault();
  const payload = {
    ollama_model: document.getElementById("set-ollama-model").value,
    whisper_model: document.getElementById("set-whisper-model").value,
    whisper_device: document.getElementById("set-whisper-device").value,
    whisper_language: document.getElementById("set-whisper-lang").value,
    num_ctx: Number(document.getElementById("set-num-ctx").value),
    idle_unload_sec: Number(document.getElementById("set-idle-unload").value),
    classroom_audio_enhance: document.getElementById("set-classroom-enhance").checked,
    enable_diarization: document.getElementById("set-enable-diarization").checked,
    enable_thinking: document.getElementById("set-enable-thinking").checked,
    eco_mode: document.getElementById("set-eco-mode").checked,
    auto_summarize: document.getElementById("set-auto-summarize").checked,
    eco_cpu_threads: Number(document.getElementById("set-eco-cpu-threads").value || 2),
    accent_color: state.accentColor,
    font_scale: state.fontScale,
  };
  await api("/api/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  state.appearancePreviewBackup = null;
  applyAppearance({ accentColor: state.accentColor, fontScale: state.fontScale }, true);
  closeSettingsModal(false);
  await refreshSystemStatus();
  if (state.activeSessionId) {
    const fresh = await api(`/api/sessions/${state.activeSessionId}`);
    state.activeSession = fresh;
    renderContextMetrics(fresh.context_metrics);
    renderWorkspaceSubtitle(fresh);
  }
  showToast("Configuración guardada.");
}

async function triggerUnloadVram() {
  const btn = document.getElementById("btn-unload-vram");
  const prevText = btn.textContent;
  btn.disabled = true;
  btn.textContent = "Liberando...";
  try {
    await api("/api/system/unload", { method: "POST" });
    await refreshSystemStatus();
    btn.textContent = "✓ Libre";
    btn.classList.add("is-confirmed");
    showToast("Memoria VRAM y RAM liberada.");
    setTimeout(() => {
      btn.classList.remove("is-confirmed");
      btn.textContent = prevText;
    }, 1500);
  } catch (err) {
    btn.textContent = prevText;
    showToast(err.message, true);
  } finally {
    btn.disabled = false;
  }
}

// --- Event Wiring ---

function initEvents() {
  document.getElementById("btn-toggle-theme").onclick = () => {
    const next = state.theme === "dark" ? "light" : "dark";
    applyTheme(next, true);
  };

  const studioBody = document.getElementById("studio-body");
  const drawerBackdrop = document.getElementById("drawer-backdrop");
  const isCompactDrawerMode = () => window.innerWidth <= 1040;

  const syncDrawerBackdrop = () => {
    if (!drawerBackdrop) return;
    const anyDrawerOpen = isCompactDrawerMode() && (
      !studioBody.classList.contains("left-collapsed")
      || !studioBody.classList.contains("right-collapsed")
    );
    drawerBackdrop.classList.toggle("hidden", !anyDrawerOpen);
  };

  let lastViewportTier = null;
  const applyResponsiveViewportLayout = (force = false) => {
    const w = window.innerWidth;
    const tier = w <= 1040 ? "narrow" : w <= 1180 ? "medium" : "wide";
    if (force || tier !== lastViewportTier) {
      lastViewportTier = tier;
      if (tier === "narrow") {
        studioBody.classList.add("left-collapsed", "right-collapsed");
      } else if (tier === "medium") {
        studioBody.classList.add("left-collapsed");
        studioBody.classList.remove("right-collapsed");
      } else {
        studioBody.classList.remove("left-collapsed", "right-collapsed");
      }
      setTimeout(updateTabIndicator, 250);
    }
    syncDrawerBackdrop();
  };

  document.getElementById("btn-toggle-sidebar").onclick = () => {
    const willOpen = studioBody.classList.contains("left-collapsed");
    studioBody.classList.toggle("left-collapsed");
    if (isCompactDrawerMode() && willOpen) {
      studioBody.classList.add("right-collapsed");
    }
    syncDrawerBackdrop();
    setTimeout(updateTabIndicator, 250);
  };
  document.getElementById("btn-toggle-sources").onclick = () => {
    const willOpen = studioBody.classList.contains("right-collapsed");
    studioBody.classList.toggle("right-collapsed");
    if (isCompactDrawerMode() && willOpen) {
      studioBody.classList.add("left-collapsed");
    }
    syncDrawerBackdrop();
    setTimeout(updateTabIndicator, 250);
  };
  drawerBackdrop?.addEventListener("click", () => {
    studioBody.classList.add("left-collapsed", "right-collapsed");
    syncDrawerBackdrop();
  });

  applyResponsiveViewportLayout(true);
  let resizeRaf = 0;
  window.addEventListener("resize", () => {
    if (resizeRaf) cancelAnimationFrame(resizeRaf);
    resizeRaf = requestAnimationFrame(() => applyResponsiveViewportLayout(false));
  });

  // Personalization live preview controls
  document.getElementById("accent-swatches")?.addEventListener("click", (e) => {
    const swatch = e.target.closest(".accent-swatch");
    if (!swatch?.dataset.accent) return;
    applyAppearance({ accentColor: swatch.dataset.accent, fontScale: state.fontScale }, false);
  });
  document.getElementById("set-accent-custom")?.addEventListener("input", (e) => {
    applyAppearance({ accentColor: e.target.value, fontScale: state.fontScale }, false);
  });
  document.getElementById("font-scale-group")?.addEventListener("click", (e) => {
    const btn = e.target.closest(".seg-control-item");
    if (!btn?.dataset.fontScale) return;
    applyAppearance({ accentColor: state.accentColor, fontScale: Number(btn.dataset.fontScale) }, false);
  });
  document.getElementById("set-font-scale")?.addEventListener("input", (e) => {
    applyAppearance({ accentColor: state.accentColor, fontScale: Number(e.target.value) }, false);
  });
  document.getElementById("btn-reset-appearance")?.addEventListener("click", () => {
    applyAppearance({ accentColor: "#c8ee26", fontScale: 1.0 }, false);
  });

  document.getElementById("btn-asr-model-action")?.addEventListener("click", async () => {
    const btn = document.getElementById("btn-asr-model-action");
    const select = document.getElementById("set-whisper-model");
    const modelId = select?.value;
    if (!modelId || btn.disabled) return;
    const action = btn.dataset.asrAction || "download";

    if (action === "delete") {
      openConfirm(
        "Eliminar pesos del modelo",
        `Se borrarán de tu disco los archivos descargados de '${modelId}' para liberar espacio. Podrás descargarlo de nuevo cuando quieras.`,
        async () => {
          await api(`/api/whisper/models/${encodeURIComponent(modelId)}`, { method: "DELETE" });
          await refreshSystemStatus({ syncSettingsModal: true });
          showToast(`Pesos de '${modelId}' eliminados del disco.`);
        }
      );
      return;
    }

    const prevText = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Descargando…";
    const pill = document.getElementById("asr-model-status-pill");
    if (pill) pill.textContent = `Descargando ${modelId}… (puede tardar 1–3 min)`;
    try {
      await api(`/api/whisper/models/${encodeURIComponent(modelId)}/download`, { method: "POST" });
      await refreshSystemStatus({ syncSettingsModal: true });
      showToast(`Modelo '${modelId}' descargado y listo.`);
    } catch (err) {
      btn.disabled = false;
      btn.textContent = prevText;
      showToast(err.message, true);
    }
  });

  const telPopover = document.getElementById("telemetry-popover");
  const telBtn = document.getElementById("btn-telemetry-popover");
  telBtn.onclick = (e) => {
    e.stopPropagation();
    telPopover.classList.toggle("hidden");
  };

  document.getElementById("btn-new-session").onclick = createNewSession;
  document.getElementById("btn-empty-new").onclick = createNewSession;
  document.getElementById("session-search").oninput = renderSessionList;

  document.getElementById("btn-rename-session").onclick = openRenameModal;
  document.getElementById("btn-close-rename").onclick = closeRenameModal;
  document.getElementById("btn-cancel-rename").onclick = closeRenameModal;
  document.getElementById("rename-form").onsubmit = async (e) => {
    e.preventDefault();
    if (!state.activeSession) return;
    const input = document.getElementById("rename-input");
    const next = input.value.trim();
    if (!next) return;
    const saveBtn = document.getElementById("btn-save-rename");
    saveBtn.disabled = true;
    try {
      await api(`/api/sessions/${state.activeSession.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: next }),
      });
      closeRenameModal();
      await selectSession(state.activeSession.id);
      await loadSessions(false);
      const titleEl = document.getElementById("session-title");
      titleEl.classList.remove("flash-confirm");
      void titleEl.offsetWidth;
      titleEl.classList.add("flash-confirm");
      showToast("Nombre actualizado.");
    } catch (err) {
      showToast(err.message, true);
    } finally {
      saveBtn.disabled = false;
    }
  };

  document.getElementById("audio-file-input").onchange = (e) => {
    uploadFiles(e.target.files);
    e.target.value = "";
  };

  document.getElementById("btn-microphone").onclick = toggleMicrophoneCapture;
  document.getElementById("capture-mode-bar")?.addEventListener("click", (e) => {
    const btn = e.target.closest(".capture-mode-btn");
    if (!btn || btn.disabled) return;
    const mode = btn.dataset.captureMode;
    if (!mode || mode === state.captureMode) return;
    state.captureMode = mode;
    document.getElementById("microphone-status").textContent =
      mode === "system"
        ? "Modo Equipo activo (opt-in): grabará y dividirá participantes (Hablante 1, 2…)."
        : mode === "both"
          ? "Modo Ambos (llamada): separará tu voz ('Tú') y los demás hablantes."
          : "";
    syncMicrophoneControls();
  });
  window.addEventListener("beforeunload", (e) => {
    if (!microphoneCapture && !systemCaptureState) return;
    e.preventDefault();
    e.returnValue = true;
  });

  document.getElementById("btn-quick-attach").onclick = () => {
    document.getElementById("audio-file-input").click();
  };

  const shell = document.getElementById("app-shell");
  let dragDepth = 0;
  shell.addEventListener("dragenter", (e) => {
    e.preventDefault();
    dragDepth += 1;
    shell.classList.add("drag-over");
  });
  shell.addEventListener("dragover", (e) => {
    e.preventDefault();
  });
  shell.addEventListener("dragleave", () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) shell.classList.remove("drag-over");
  });
  shell.addEventListener("drop", (e) => {
    e.preventDefault();
    dragDepth = 0;
    shell.classList.remove("drag-over");
    if (e.dataTransfer?.files?.length) {
      uploadFiles(e.dataTransfer.files);
    }
  });

  document.querySelectorAll(".seg-btn").forEach((btn) => {
    btn.onclick = () => switchTab(btn.dataset.tab);
  });

  const tabsContainer = document.getElementById("segmented-tabs");
  if (window.ResizeObserver && tabsContainer) {
    new ResizeObserver(() => updateTabIndicator()).observe(tabsContainer);
  }
  window.addEventListener("resize", updateTabIndicator);

  document.getElementById("center-scroll").addEventListener("click", (e) => {
    const stopBtn = e.target.closest(".skeleton-stop-btn, .inline-stop-btn");
    if (stopBtn) {
      e.preventDefault();
      e.stopPropagation();
      const scope = stopBtn.dataset.stopScope || "";
      if (scope === "chat") {
        stopActiveChatGeneration();
      } else if (scope === "resummarize") {
        stopResummarizeGeneration();
      } else if (scope === "quiz-gen") {
        stopQuizGeneration();
      } else if (scope === "quiz-grade") {
        stopQuizGrading();
      } else if (state.isResummarizing) {
        stopResummarizeGeneration();
      } else if (state.activeSessionId && sessionQuizzes.get(state.activeSessionId)?.loading) {
        stopQuizGeneration();
      } else {
        forceStopAllAI();
      }
      return;
    }

    const pill = e.target.closest(".chat-ts-pill");
    if (pill) {
      const seconds = Number(pill.dataset.seconds || 0);
      const recId = resolveRecordingForTimestamp(pill, seconds);
      if (recId) {
        seekRecording(recId, seconds, { pulse: true });
      }
      return;
    }

    const srcPill = e.target.closest(".chat-src-pill");
    if (srcPill && srcPill.dataset.sourceId) {
      const src = (state.activeSession?.sources || []).find((s) => s.id === srcPill.dataset.sourceId);
      if (src) {
        openSourceViewer(src, srcPill.dataset.locator || "");
      }
    }
  });

  document.getElementById("transcript-search").oninput = () => {
    if (state.activeSession) renderTranscriptTab(state.activeSession);
  };

  document.getElementById("btn-copy-transcript").onclick = async () => {
    if (!state.activeSession) return;
    const selected = (state.activeSession.recordings || []).filter(
      (r) => r.transcript_text && (state.transcriptRecFilter
        ? r.id === state.transcriptRecFilter
        : r.enabled !== false)
    );
    const blocks = selected.map((r) => `=== ${r.filename} ===\n${r.transcript_text}`);
    const text = blocks.join("\n\n");
    if (!text) return;
    await navigator.clipboard.writeText(text);
    const btn = document.getElementById("btn-copy-transcript");
    const prev = btn.textContent;
    btn.textContent = "✓ Copiado";
    btn.classList.add("is-confirmed");
    showToast("Transcripción copiada al portapapeles.");
    setTimeout(() => {
      btn.classList.remove("is-confirmed");
      btn.textContent = prev;
    }, 1600);
  };

  document.getElementById("btn-resummarize").onclick = async () => {
    if (state.isResummarizing) {
      stopResummarizeGeneration();
      return;
    }
    if (!state.activeSessionId) return;
    const btn = document.getElementById("btn-resummarize");
    const prevHtml = btn.innerHTML;
    const prevTitle = btn.title;
    const controller = new AbortController();
    state.resummarizeAbortController = controller;
    state.isResummarizing = true;
    btn.disabled = false;
    btn.classList.add("is-stop-btn");
    btn.title = "Detener regeneración del resumen";
    btn.innerHTML = `<span class="stop-sq" aria-hidden="true"></span><span>Detener</span>`;
    syncGlobalStopButton();

    if (state.activeTab !== "summary") {
      switchTab("summary");
    }
    const modelName = getActiveModelLabel();
    const summaryPane = document.getElementById("tab-summary");
    summaryPane.innerHTML = buildSkeletonHtml(
      `${modelName} está recreando el resumen...`,
      true,
      "resummarize"
    );
    const ecoProgress = createEcoProgressController(summaryPane, "Generando resumen");

    try {
      const fresh = await api(`/api/sessions/${state.activeSessionId}/resummarize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({}),
      });
      ecoProgress?.stop();
      state.activeSession = fresh;
      renderSummaryTab(fresh, true);
      showToast("Resumen actualizado.");
    } catch (err) {
      ecoProgress?.stop();
      if (state.activeSession) {
        invalidateMemo("summary-pane");
        renderSummaryTab(state.activeSession);
      }
      if (controller.signal.aborted || isAbortError(err)) {
        showToast("Resumen detenido.");
      } else {
        showToast(err.message, true);
      }
    } finally {
      ecoProgress?.stop();
      if (state.resummarizeAbortController === controller) {
        state.resummarizeAbortController = null;
      }
      state.isResummarizing = false;
      btn.classList.remove("is-stop-btn");
      btn.disabled = Boolean(state.podcastBusy);
      btn.innerHTML = prevHtml;
      btn.title = prevTitle;
      syncGlobalStopButton();
    }
  };

  document.getElementById("btn-export-md").onclick = exportActiveSessionMarkdown;

  document.getElementById("btn-reset-quiz").onclick = () => {
    if (!state.activeSession) return;
    const qs = getQuizState(state.activeSession.id);
    stopQuizTimer(qs);
    qs.questions = null;
    qs.answers = {};
    qs.drafts = {};
    qs.revealedHints = {};
    qs.examSubmitted = false;
    renderQuizTab(state.activeSession);
  };

  const chatInput = document.getElementById("chat-input");
  chatInput.addEventListener("input", () => {
    chatInput.style.height = "auto";
    chatInput.style.height = `${Math.min(110, chatInput.scrollHeight)}px`;
  });
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendChatQuestion(e.target.value);
    }
  });

  document.getElementById("btn-stop-ai").onclick = stopActiveAIOperation;

  document.getElementById("btn-send-chat").onclick = (e) => {
    if (state.isStreaming) {
      e.preventDefault();
      stopActiveChatGeneration();
    }
  };

  document.getElementById("chat-form").onsubmit = (e) => {
    e.preventDefault();
    if (!state.isStreaming) {
      sendChatQuestion(chatInput.value);
    }
  };

  document.getElementById("btn-clear-chat").onclick = handleClearChatHistory;

  document.getElementById("btn-unload-vram").onclick = triggerUnloadVram;

  document.getElementById("btn-open-settings").onclick = openSettingsModal;
  document.getElementById("btn-close-settings").onclick = closeSettingsModal;
  document.getElementById("btn-cancel-settings").onclick = closeSettingsModal;
  document.getElementById("settings-form").onsubmit = (e) =>
    saveSettings(e).catch((err) => showToast(err.message, true));

  document.getElementById("set-eco-mode").onchange = (e) => {
    syncEcoThreadsUI();
    if (!e.target.checked) return;
    document.getElementById("set-whisper-device").value = "cpu";
    document.getElementById("set-num-ctx").value = "4096";
    document.getElementById("set-idle-unload").value = "60";
    document.getElementById("set-enable-thinking").checked = false;
    ["set-whisper-device", "set-num-ctx", "set-idle-unload"].forEach(refreshCustomSelect);
  };

  document.getElementById("set-eco-cpu-threads").oninput = () => syncEcoThreadsUI();

  document.getElementById("btn-refresh-models").onclick = async () => {
    const btn = document.getElementById("btn-refresh-models");
    const prev = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Actualizando...";
    try {
      const fresh = await api("/api/system/status?refresh_models=true");
      state.systemStatus = fresh;
      renderTelemetry(fresh);
      populateSettingsFromStatus(fresh);
      btn.textContent = "✓ Actualizado";
      btn.classList.add("is-confirmed");
      setTimeout(() => {
        btn.classList.remove("is-confirmed");
        btn.textContent = prev;
      }, 1500);
    } catch (err) {
      btn.textContent = prev;
      showToast(err.message, true);
    } finally {
      btn.disabled = false;
    }
  };

  document.getElementById("btn-open-models-guide").onclick = () => {
    closeActiveDropdown();
    openModal("models-guide-modal");
  };
  document.getElementById("btn-close-models-guide").onclick = () => closeModal("models-guide-modal");
  document.getElementById("btn-ok-models-guide").onclick = () => closeModal("models-guide-modal");
  document.getElementById("btn-close-source-viewer").onclick = () => closeModal("source-viewer-modal");
  document.getElementById("btn-ok-source-viewer").onclick = () => closeModal("source-viewer-modal");

  document.getElementById("models-guide-modal").addEventListener("click", async (e) => {
    const copyBtn = e.target.closest(".guide-cmd-copy");
    if (!copyBtn) return;
    const cmd = copyBtn.dataset.cmd || "";
    if (!cmd) return;
    await navigator.clipboard.writeText(cmd);
    const prev = copyBtn.textContent;
    copyBtn.textContent = "✓ Copiado";
    copyBtn.classList.add("is-confirmed");
    setTimeout(() => {
      copyBtn.classList.remove("is-confirmed");
      copyBtn.textContent = prev;
    }, 1400);
  });

  document.getElementById("btn-close-confirm").onclick = closeConfirm;
  document.getElementById("btn-confirm-cancel").onclick = closeConfirm;
  document.getElementById("btn-confirm-ok").onclick = async () => {
    const fn = state.confirmCallback;
    closeConfirm();
    if (fn) {
      try {
        await fn();
      } catch (err) {
        showToast(err.message, true);
      }
    }
  };

  document.addEventListener("click", (e) => {
    if (activeDropdownMenu && !e.target.closest(".bento-select") && !e.target.closest(".bento-select-menu")) {
      closeActiveDropdown();
    }
    const telPopoverEl = document.getElementById("telemetry-popover");
    if (telPopoverEl && !telPopoverEl.classList.contains("hidden") && !e.target.closest(".telemetry-anchor")) {
      telPopoverEl.classList.add("hidden");
    }
  });

  ["settings-modal", "models-guide-modal", "source-viewer-modal", "confirm-modal", "rename-modal"].forEach((id) => {
    document.getElementById(id).addEventListener("click", (e) => {
      if (e.target.id === id) {
        if (id === "settings-modal") closeSettingsModal();
        else if (id === "models-guide-modal") closeModal("models-guide-modal");
        else if (id === "source-viewer-modal") closeModal("source-viewer-modal");
        else if (id === "confirm-modal") closeConfirm();
        else if (id === "rename-modal") closeRenameModal();
      }
    });
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      const telPopoverEl = document.getElementById("telemetry-popover");
      if (telPopoverEl && !telPopoverEl.classList.contains("hidden")) {
        telPopoverEl.classList.add("hidden");
      } else if (activeDropdownMenu) {
        closeActiveDropdown();
      } else if (isModalOpen("source-viewer-modal")) {
        closeModal("source-viewer-modal");
      } else if (isModalOpen("models-guide-modal")) {
        closeModal("models-guide-modal");
      } else if (isModalOpen("settings-modal")) {
        closeSettingsModal();
      } else if (isModalOpen("rename-modal")) {
        closeRenameModal();
      } else if (isModalOpen("confirm-modal")) {
        closeConfirm();
      }
    }
  });
}

// --- Bootstrap ---

async function initApp() {
  const savedTheme = (() => {
    try {
      return localStorage.getItem("transcript_theme") || "light";
    } catch (_) {
      return "light";
    }
  })();
  applyTheme(savedTheme, false);

  try {
    const savedAccent = localStorage.getItem("transcript_accent_color");
    const savedScale = localStorage.getItem("transcript_font_scale");
    applyAppearance(
      {
        accentColor: savedAccent || "#c8ee26",
        fontScale: savedScale ? Number(savedScale) : 1.0,
      },
      false
    );
  } catch (_) {}

  initAllCustomSelects();
  initEvents();
  switchTab("summary");
  await refreshSystemStatus();
  await loadSessions(true);
  setInterval(refreshSystemStatus, 3000);
}

window.addEventListener("DOMContentLoaded", initApp);
