/* TransCrypting Frontend — Bento Lime/Obsidian/Porcelain (100% local) */

const AUDIO_EXTENSIONS = new Set([
  ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".aac", ".wma", ".mp4", ".mkv", ".mov", ".opus",
]);
const SOURCE_EXTENSIONS = new Set([
  ".pdf", ".pptx", ".docx", ".txt", ".md", ".png", ".jpg", ".jpeg", ".webp", ".bmp",
]);
const TAB_ORDER = ["summary", "transcript", "chat", "quiz", "podcast"];

const state = {
  sessions: [],
  activeSessionId: null,
  activeSession: null,
  activeRecordingId: null,
  systemStatus: null,
  activeTab: "summary",
  isStreaming: false,
  isResummarizing: false,
  lastContextMode: "full",
  theme: "light",
  confirmCallback: null,
  pollTimer: null,
  fetchingSuggestionsFor: new Set(),
  attemptedSuggestionsKeys: new Set(),
  podcastBusy: false,
  podcastStartedAt: null,
  podcastElapsedTimer: null,
  cancellingPodcastId: null,
  transcriptRecFilter: "all",
};

const audioPlayers = new Map();
const podcastAudioPlayers = new Map();
const podcastFormState = new Map();
const sessionQuizzes = new Map();
const renderMemo = new Map();
const customSelectControllers = new Map();
let activeDropdownMenu = null;
let activeDropdownWrapper = null;

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

function makeTimestampPillHtml(tsStr) {
  const clean = tsStr.trim();
  const seconds = parseTimestampToSeconds(clean);
  return `<button type="button" class="chat-ts-pill" data-seconds="${seconds}" data-label="${escapeHtml(clean)}" title="Ir al momento ${escapeHtml(clean)}"><span class="ts-play-tri">▶</span>${escapeHtml(clean)}</button>`;
}

function findMatchingSourceByRef(refText) {
  const sources = state.activeSession?.sources || [];
  if (!sources.length) return { source: null, locator: "" };
  const raw = String(refText || "").trim().replace(/^\[|\]$/g, "");
  const parts = raw.split(/\s*[·|]\s*/);
  const namePart = (parts[0] || "").trim().toLowerCase();
  const locator = (parts[1] || "").trim();
  const matched =
    sources.find((s) => (s.filename || "").toLowerCase() === namePart) ||
    sources.find((s) => namePart && (s.filename || "").toLowerCase().includes(namePart));
  if (matched) return { source: matched, locator };
  return { source: null, locator };
}

function makeSourcePillHtml(source, label, locator = "") {
  const icon = source?.source_type === "image" ? "🖼" : "📄";
  return `<button type="button" class="chat-src-pill" data-source-id="${escapeHtml(source?.id || "")}" data-locator="${escapeHtml(locator)}" title="Abrir fuente: ${escapeHtml(label)}"><span>${icon}</span><span>${escapeHtml(label)}</span></button>`;
}

function makeCitationPillHtml(refValue) {
  const clean = String(refValue || "").trim();
  if (!clean) return "";
  if (/^\d{1,2}:\d{2}(?::\d{2})?$/.test(clean)) {
    return makeTimestampPillHtml(clean);
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

  const tsSingle = "\\d{1,2}:\\d{2}(?::\\d{2})?";
  const bracketRe = new RegExp(`\\[(${tsSingle}(?:\\s*[-–,]\\s*${tsSingle})*)\\]`, "g");
  const sources = state.activeSession?.sources || [];
  const docRefRe = /\[([^\[\]\n]{3,90}?\.(?:pdf|pptx|docx|txt|md|png|jpg|jpeg|webp|bmp)(?:\s*[·|]\s*(?:p\.|diap\.|sec\.)\s*\d+)?)\]/gi;

  const parts = masked.split(/(<[^>]+>)/g);
  for (let i = 0; i < parts.length; i += 2) {
    parts[i] = parts[i].replace(bracketRe, (_, inner) => {
      const matches = inner.match(new RegExp(tsSingle, "g"));
      if (!matches || matches.length === 0) return `[${inner}]`;
      return matches.map(makeTimestampPillHtml).join(" ");
    });
    if (sources.length > 0) {
      parts[i] = parts[i].replace(docRefRe, (full, inner) => {
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

// --- Telemetry, Context Health & Hero Status ---

async function refreshSystemStatus(options = {}) {
  try {
    const data = await api("/api/system/status");
    state.systemStatus = data;
    renderTelemetry(data);
    if (data.settings?.theme && !localStorage.getItem("transcript_theme")) {
      applyTheme(data.settings.theme, false);
    }
    if (options.syncSettingsModal && isModalOpen("settings-modal") && !activeDropdownMenu) {
      populateSettingsFromStatus(data);
    }
  } catch (_) {}
}

function renderTelemetry(data) {
  const hw = data.hardware || {};
  const res = data.resources || {};
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
  } else {
    gpuLabel.textContent = "GPU";
    vramText.textContent = "Modo CPU";
    vramBar.style.width = "0%";
    gpuTemp.textContent = "CPU Local";
  }

  document.getElementById("ram-text").textContent = `${hw.ram_used_gb} / ${hw.ram_total_gb} GB`;
  document.getElementById("ram-bar").style.width = `${Math.min(100, hw.ram_percent || 0)}%`;
  const activeLabel = getActiveModelLabel();
  document.getElementById("active-model-name").textContent = activeLabel;

  const pill = document.getElementById("model-state-pill");
  const pillText = document.getElementById("model-state-text");
  const vramChip = document.getElementById("topbar-vram-pill");
  const telTrigger = document.getElementById("btn-telemetry-popover");

  const vramPct = Math.round(hw.vram_percent || 0);
  if (vramChip) {
    vramChip.textContent = hw.gpu_available ? `VRAM ${vramPct}%` : "CPU";
  }

  if (res.phase === "podcast" || res.podcast_active) {
    pill.className = "mini-status active";
    const podPct = Math.max(8, Math.round((res.podcast_progress || 0) * 100));
    pillText.textContent = `Podcast ${podPct}%`;
  } else if (res.phase === "whisper") {
    pill.className = "mini-status active";
    pillText.textContent = `Whisper ${(res.active_device || "").toUpperCase()}`;
  } else if (res.phase === "llm") {
    pill.className = "mini-status active";
    pillText.textContent = `IA ${res.active_model || activeLabel}`;
  } else if (res.phase === "unloading") {
    pill.className = "mini-status active";
    pillText.textContent = "Liberando...";
  } else if (runningOllama.length > 0) {
    pill.className = "mini-status active";
    pillText.textContent = "En VRAM";
  } else {
    pill.className = "mini-status idle";
    pillText.textContent = "VRAM libre";
  }

  if (telTrigger) {
    telTrigger.classList.toggle("podcast-live", Boolean(res.podcast_active || res.phase === "podcast"));
  }

  syncPodcastBusyUI(Boolean(res.podcast_active || res.phase === "podcast"));
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
    state.transcriptRecFilter = "all";
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

    audio.addEventListener("timeupdate", () => syncPlayerCardDOM(rec.id));
    audio.addEventListener("play", () => syncPlayerCardDOM(rec.id));
    audio.addEventListener("pause", () => syncPlayerCardDOM(rec.id));
    audio.addEventListener("ended", () => syncPlayerCardDOM(rec.id));
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
    info.appendChild(meta);

    const rightBtns = document.createElement("div");
    rightBtns.className = "rec-controls-right";

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
      retransBtn.title = "Re-transcribir con la configuración actual de Whisper";
      retransBtn.innerHTML = ICON_RETRANSCRIBE;
      retransBtn.onclick = (e) => {
        e.stopPropagation();
        const model = state.systemStatus?.settings?.whisper_model || "large-v3-turbo";
        openConfirm(
          "Re-transcribir grabación",
          `¿Volver a transcribir "${rec.filename}" usando Whisper (${model}) y la mejora de audio para clases? Se actualizarán su transcripción y resumen.`,
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
          audio.pause();
          audioPlayers.delete(rec.id);
          await api(`/api/sessions/${rec.session_id}/recordings/${rec.id}`, {
            method: "DELETE",
          });
          await selectSession(rec.session_id);
          await loadSessions(false);
          showToast("Grabación eliminada.");
        }
      );
    };

    rightBtns.appendChild(speedBtn);
    rightBtns.appendChild(removeBtn);

    top.appendChild(chk);
    top.appendChild(playBtn);
    top.appendChild(info);
    top.appendChild(rightBtns);
    card.appendChild(top);

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
    info.appendChild(meta);

    const rightBtns = document.createElement("div");
    rightBtns.className = "rec-controls-right";

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
          await api(`/api/sessions/${src.session_id}/sources/${src.id}`, {
            method: "DELETE",
          });
          await selectSession(src.session_id);
          await loadSessions(false);
          showToast("Fuente eliminada.");
        }
      );
    };
    rightBtns.appendChild(removeBtn);

    top.appendChild(chk);
    top.appendChild(iconEl);
    top.appendChild(info);
    top.appendChild(rightBtns);
    card.appendChild(top);

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
    audio.currentTime = Math.max(0, Number(seconds) || 0);
    audio.play().catch(() => {});
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
    (r) => r.status === "ready" || r.duration_sec > 0
  );
  if (recordings.length === 0) return null;
  if (recordings.length === 1) return recordings[0].id;

  const contextContainer = triggerEl.closest("p, li, .msg-bubble, .content-card");
  const contextText = (contextContainer?.textContent || "").toLowerCase();

  for (let i = 0; i < recordings.length; i += 1) {
    const rec = recordings[i];
    if (
      contextText.includes((rec.filename || "").toLowerCase()) ||
      contextText.includes(`grabación ${i + 1}`)
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

function buildSkeletonHtml(statusLabel) {
  return `
    <div class="summary-skeleton-card">
      <div class="skeleton-banner">
        <div class="think-dots" aria-hidden="true"><span></span><span></span><span></span></div>
        <span class="think-label">${escapeHtml(statusLabel)}</span>
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
      const modelName = getActiveModelLabel();
      const stageMsg = `${workingItem.stage || "Procesando"} (${Math.round((workingItem.progress || 0) * 100)}%) — ${modelName} preparará el resumen...`;
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

  pane.innerHTML = `
    <div class="content-card combined ${revealClass}">
      <div class="content-card-header">
        <span>Resumen general de la sesión</span>
        <span class="card-badge lime">${totalActive} fuente(s) activa(s)</span>
      </div>
      ${chipsHtml ? `<div class="summary-sources-strip">${chipsHtml}</div>` : ""}
      <div class="md">${renderMarkdown(summaryMarkdown || "_Sin resumen generado aún._", true)}</div>
    </div>
  `;
}

function renderTranscriptTab(session) {
  const pane = document.getElementById("tab-transcript");
  const query = (document.getElementById("transcript-search").value || "").trim().toLowerCase();
  const recordings = (session.recordings || []).filter((r) => r.transcript_text && r.enabled !== false);
  const sources = (session.sources || []).filter((s) => s.extracted_text && s.enabled !== false);

  if (state.transcriptRecFilter !== "all" && !recordings.some((r) => r.id === state.transcriptRecFilter)) {
    state.transcriptRecFilter = "all";
  }

  const contentSig = [
    session.id,
    query,
    state.transcriptRecFilter,
    recordings.map((r) => `${r.id}:${(r.transcript_text || "").length}:${(r.segments || []).length}`).join(","),
    sources.map((s) => s.id).join(","),
  ].join("|");
  if (!memoChanged("content-pane", contentSig)) return;

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

  if (recordings.length >= 2) {
    const filterBar = document.createElement("div");
    filterBar.className = "transcript-filter-bar";

    const allBtn = document.createElement("button");
    allBtn.type = "button";
    allBtn.className = `quiz-choice-pill ${state.transcriptRecFilter === "all" ? "active" : ""}`;
    allBtn.textContent = `Todos los audios (${recordings.length})`;
    allBtn.onclick = () => {
      state.transcriptRecFilter = "all";
      renderTranscriptTab(session);
    };
    filterBar.appendChild(allBtn);

    for (const r of recordings) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `quiz-choice-pill ${state.transcriptRecFilter === r.id ? "active" : ""}`;
      btn.textContent = `🎙 ${r.filename}`;
      btn.onclick = () => {
        state.transcriptRecFilter = r.id;
        renderTranscriptTab(session);
      };
      filterBar.appendChild(btn);
    }
    pane.appendChild(filterBar);
  }

  const visibleRecordings = state.transcriptRecFilter === "all"
    ? recordings
    : recordings.filter((r) => r.id === state.transcriptRecFilter);

  const highlightMatch = (el, text) => {
    if (query) {
      const safe = escapeHtml(text);
      const re = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "gi");
      el.innerHTML = safe.replace(re, '<mark class="hl">$1</mark>');
    } else {
      el.textContent = text;
    }
  };

  for (const rec of visibleRecordings) {
    const card = document.createElement("div");
    card.className = "content-card";

    const header = document.createElement("div");
    header.className = "content-card-header";
    header.innerHTML = `<span>🎙 ${escapeHtml(rec.filename)}</span><span class="card-badge">${formatDuration(rec.duration_sec)} · Clic en tiempo para reproducir</span>`;
    card.appendChild(header);

    const segs = rec.segments || [];
    let matchedAny = false;
    if (segs.length > 0) {
      for (const seg of segs) {
        const text = seg.text || "";
        if (query && !text.toLowerCase().includes(query)) continue;
        matchedAny = true;

        const row = document.createElement("div");
        row.className = "segment-row";

        const tsBtn = document.createElement("button");
        tsBtn.className = "timestamp-btn";
        tsBtn.textContent = formatDuration(seg.start);
        tsBtn.onclick = () => seekRecording(rec.id, seg.start, { pulse: true });

        const textEl = document.createElement("div");
        textEl.className = "segment-text";
        highlightMatch(textEl, text);

        row.appendChild(tsBtn);
        row.appendChild(textEl);
        card.appendChild(row);
      }
    } else if (!query || rec.transcript_text.toLowerCase().includes(query)) {
      matchedAny = true;
      const p = document.createElement("div");
      p.className = "segment-text";
      highlightMatch(p, rec.transcript_text);
      card.appendChild(p);
    }

    if (!matchedAny) {
      const emptyMatch = document.createElement("div");
      emptyMatch.className = "rec-empty";
      emptyMatch.textContent = "Sin coincidencias en este audio para la búsqueda actual.";
      card.appendChild(emptyMatch);
    }

    pane.appendChild(card);
  }
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
  qs.loading = true;
  qs.questions = null;
  qs.answers = {};
  qs.drafts = {};
  qs.revealedHints = {};
  qs.examSubmitted = false;
  qs.submittingExam = false;
  qs.elapsedSec = 0;
  qs.startedAt = null;
  renderQuizTab(session);

  try {
    const res = await api(`/api/sessions/${session.id}/quiz/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
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
    showToast(err.message, true);
  } finally {
    qs.loading = false;
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
  qs.drafts[q.id] = clean;
  qs.answers[q.id] = { type: "open", userAnswer: clean, grading: true, result: null };
  updateSingleQuizCard(session, q);

  try {
    const result = await api(`/api/sessions/${session.id}/quiz/grade`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
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
    showToast(err.message, true);
  } finally {
    updateSingleQuizCard(session, q);
  }
}

async function submitFullExam(session) {
  const qs = getQuizState(session.id);
  if (!qs.questions || qs.submittingExam || qs.examSubmitted) return;

  qs.submittingExam = true;
  stopQuizTimer(qs);
  const submitBtn = document.querySelector("#tab-quiz .quiz-submit-exam-btn");
  if (submitBtn) {
    submitBtn.disabled = true;
    submitBtn.textContent = "Calificando examen...";
  }

  try {
    for (const q of qs.questions) {
      if (q.type === "open") {
        const draft = (qs.drafts[q.id] || "").trim();
        if (draft && !qs.answers[q.id]?.result) {
          const result = await api(`/api/sessions/${session.id}/quiz/grade`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
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
    showToast(err.message, true);
    if (submitBtn) {
      submitBtn.disabled = false;
      submitBtn.textContent = "Entregar y calificar examen";
    }
  } finally {
    qs.submittingExam = false;
  }
}

function renderQuizTab(session) {
  const pane = document.getElementById("tab-quiz");
  const resetBtn = document.getElementById("btn-reset-quiz");
  if (!pane) return;

  const readyRecs = (session?.recordings || []).filter((r) => r.status === "ready");
  const readySrcs = (session?.sources || []).filter((s) => s.status === "ready");
  const qs = getQuizState(session?.id || "none");
  const selectedTotal = qs.selectedRecIds.size + qs.selectedSrcIds.size;

  const quizSig = [
    session?.id || "",
    readyRecs.length,
    readySrcs.length,
    qs.loading ? 1 : 0,
    qs.count,
    qs.mode,
    qs.difficulty,
    qs.questionStyle,
    qs.includeHints ? 1 : 0,
    qs.examMode ? 1 : 0,
    qs.timeLimitMin,
    qs.examSubmitted ? 1 : 0,
    selectedTotal,
    Array.from(qs.selectedRecIds).sort().join(","),
    Array.from(qs.selectedSrcIds).sort().join(","),
    qs.questions ? qs.questions.length : -1,
    JSON.stringify(qs.answers),
  ].join("|");
  if (!memoChanged("quiz-pane", quizSig)) return;

  if (!session || (readyRecs.length === 0 && readySrcs.length === 0)) {
    if (resetBtn) resetBtn.classList.add("hidden");
    pane.innerHTML = `<div class="chat-placeholder"><h3>Estudio de exámenes</h3><p>Cuando al menos una grabación o fuente documental esté lista podrás personalizar aquí simulacros de examen basados en tus materiales.</p></div>`;
    return;
  }

  if (qs.loading) {
    if (resetBtn) resetBtn.classList.add("hidden");
    const modelName = getActiveModelLabel();
    pane.innerHTML = buildSkeletonHtml(`${modelName} está diseñando tu examen personalizado...`);
    return;
  }

  if (!qs.questions || qs.questions.length === 0) {
    if (resetBtn) resetBtn.classList.add("hidden");
    pane.innerHTML = "";

    const sourceChipsHtml = [
      ...readyRecs.map((r) => {
        const sel = qs.selectedRecIds.has(r.id);
        return `<button type="button" class="podcast-source-chip ${sel ? "selected" : ""}" data-quiz-kind="rec" data-quiz-item="${escapeHtml(r.id)}">🎙 ${escapeHtml(r.filename)}</button>`;
      }),
      ...readySrcs.map((s) => {
        const sel = qs.selectedSrcIds.has(s.id);
        const ico = s.source_type === "image" ? "🖼" : "📄";
        return `<button type="button" class="podcast-source-chip ${sel ? "selected" : ""}" data-quiz-kind="src" data-quiz-item="${escapeHtml(s.id)}">${ico} ${escapeHtml(s.filename)}</button>`;
      }),
    ].join("");

    const setup = document.createElement("div");
    setup.className = "quiz-setup-card";
    setup.innerHTML = `
      <div class="quiz-setup-header">
        <h2>Generador de examen personalizado</h2>
        <p>Elige las fuentes, el nivel de exigencia, el enfoque temático y si deseas retroalimentación inmediata o simulación cronometrada de examen.</p>
      </div>
      <div class="quiz-config-grid">
        <div class="quiz-config-row">
          <div class="quiz-config-label">Fuentes a evaluar (${selectedTotal} seleccionada(s))</div>
          <div class="podcast-sources-grid">${sourceChipsHtml}</div>
        </div>
        <div class="quiz-config-row">
          <div class="quiz-config-label">Tema o enfoque específico (opcional)</div>
          <textarea class="podcast-focus-input quiz-focus-input" placeholder="Ej. Enfócate en definiciones clave, fórmulas del capítulo 3, diferencias entre algoritmos o casos clínicos...">${escapeHtml(qs.focus)}</textarea>
        </div>
        <div class="quiz-config-row">
          <div class="quiz-config-label">Cantidad de preguntas</div>
          <div class="quiz-pill-group" data-quiz-group="count">
            <button type="button" class="quiz-choice-pill ${qs.count === 3 ? "active" : ""}" data-val="3">3 preguntas</button>
            <button type="button" class="quiz-choice-pill ${qs.count === 5 ? "active" : ""}" data-val="5">5 preguntas</button>
            <button type="button" class="quiz-choice-pill ${qs.count === 8 ? "active" : ""}" data-val="8">8 preguntas</button>
            <button type="button" class="quiz-choice-pill ${qs.count === 10 ? "active" : ""}" data-val="10">10 preguntas</button>
            <button type="button" class="quiz-choice-pill ${qs.count === 15 ? "active" : ""}" data-val="15">15 preguntas</button>
          </div>
        </div>
        <div class="quiz-config-row">
          <div class="quiz-config-label">Formato de preguntas</div>
          <div class="quiz-pill-group" data-quiz-group="mode">
            <button type="button" class="quiz-choice-pill ${qs.mode === "mixed" ? "active" : ""}" data-val="mixed">Mixto completo</button>
            <button type="button" class="quiz-choice-pill ${qs.mode === "mcq" ? "active" : ""}" data-val="mcq">Opción múltiple</button>
            <button type="button" class="quiz-choice-pill ${qs.mode === "true_false" ? "active" : ""}" data-val="true_false">Verdadero / Falso</button>
            <button type="button" class="quiz-choice-pill ${qs.mode === "open" ? "active" : ""}" data-val="open">Respuesta abierta</button>
          </div>
        </div>
        <div class="field-row">
          <div class="quiz-config-row">
            <div class="quiz-config-label">Dificultad</div>
            <div class="quiz-pill-group" data-quiz-group="difficulty">
              <button type="button" class="quiz-choice-pill ${qs.difficulty === "easy" ? "active" : ""}" data-val="easy">Básico</button>
              <button type="button" class="quiz-choice-pill ${qs.difficulty === "medium" ? "active" : ""}" data-val="medium">Intermedio</button>
              <button type="button" class="quiz-choice-pill ${qs.difficulty === "hard" ? "active" : ""}" data-val="hard">Difícil / Parcial</button>
            </div>
          </div>
          <div class="quiz-config-row">
            <div class="quiz-config-label">Estilo de preguntas</div>
            <div class="quiz-pill-group" data-quiz-group="questionStyle">
              <button type="button" class="quiz-choice-pill ${qs.questionStyle === "conceptual" ? "active" : ""}" data-val="conceptual">Conceptual</button>
              <button type="button" class="quiz-choice-pill ${qs.questionStyle === "practical" ? "active" : ""}" data-val="practical">Casos prácticos</button>
              <button type="button" class="quiz-choice-pill ${qs.questionStyle === "tricky" ? "active" : ""}" data-val="tricky">Trampas</button>
            </div>
          </div>
        </div>
        <div class="field-row">
          <div class="quiz-config-row">
            <div class="quiz-config-label">Modalidad de evaluación</div>
            <div class="quiz-pill-group" data-quiz-group="examMode">
              <button type="button" class="quiz-choice-pill ${!qs.examMode ? "active" : ""}" data-val="false">Feedback inmediato</button>
              <button type="button" class="quiz-choice-pill ${qs.examMode ? "active" : ""}" data-val="true">Modo examen real</button>
            </div>
          </div>
          <div class="quiz-config-row">
            <div class="quiz-config-label">Temporizador y pistas</div>
            <div class="quiz-pill-group" data-quiz-group="timeLimitMin">
              <button type="button" class="quiz-choice-pill ${qs.timeLimitMin === 0 ? "active" : ""}" data-val="0">Sin límite</button>
              <button type="button" class="quiz-choice-pill ${qs.timeLimitMin === 5 ? "active" : ""}" data-val="5">5 min</button>
              <button type="button" class="quiz-choice-pill ${qs.timeLimitMin === 10 ? "active" : ""}" data-val="10">10 min</button>
              <button type="button" class="quiz-choice-pill ${qs.timeLimitMin === 15 ? "active" : ""}" data-val="15">15 min</button>
              <button type="button" class="quiz-choice-pill quiz-hint-toggle ${qs.includeHints ? "active" : ""}" data-val="toggle">💡 Pistas</button>
            </div>
          </div>
        </div>
      </div>
      <button type="button" id="btn-generate-quiz" class="quiz-start-btn" ${state.podcastBusy || selectedTotal === 0 ? "disabled" : ""}>
        <span>Generar examen personalizado</span>
        <span class="cta-circle" aria-hidden="true">→</span>
      </button>
    `;

    setup.querySelectorAll(".podcast-source-chip").forEach((chip) => {
      chip.onclick = () => {
        const target = chip.dataset.quizKind === "rec" ? qs.selectedRecIds : qs.selectedSrcIds;
        const id = chip.dataset.quizItem;
        if (target.has(id)) target.delete(id);
        else target.add(id);
        renderQuizTab(session);
      };
    });

    setup.querySelector(".quiz-focus-input").addEventListener("input", (e) => {
      qs.focus = e.target.value;
    });

    setup.querySelectorAll('[data-quiz-group="count"] .quiz-choice-pill').forEach((b) => {
      b.onclick = () => {
        qs.count = Number(b.dataset.val);
        renderQuizTab(session);
      };
    });
    setup.querySelectorAll('[data-quiz-group="mode"] .quiz-choice-pill').forEach((b) => {
      b.onclick = () => {
        qs.mode = b.dataset.val;
        renderQuizTab(session);
      };
    });
    setup.querySelectorAll('[data-quiz-group="difficulty"] .quiz-choice-pill').forEach((b) => {
      b.onclick = () => {
        qs.difficulty = b.dataset.val;
        renderQuizTab(session);
      };
    });
    setup.querySelectorAll('[data-quiz-group="questionStyle"] .quiz-choice-pill').forEach((b) => {
      b.onclick = () => {
        qs.questionStyle = b.dataset.val;
        renderQuizTab(session);
      };
    });
    setup.querySelectorAll('[data-quiz-group="examMode"] .quiz-choice-pill').forEach((b) => {
      b.onclick = () => {
        qs.examMode = b.dataset.val === "true";
        renderQuizTab(session);
      };
    });
    setup.querySelectorAll('[data-quiz-group="timeLimitMin"] .quiz-choice-pill').forEach((b) => {
      b.onclick = () => {
        if (b.dataset.val === "toggle") {
          qs.includeHints = !qs.includeHints;
        } else {
          qs.timeLimitMin = Number(b.dataset.val);
        }
        renderQuizTab(session);
      };
    });

    setup.querySelector("#btn-generate-quiz").onclick = () => generateQuizForSession(session);
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
    st ? Array.from(st.selectedRecIds).sort().join(",") : "",
    st ? Array.from(st.selectedSrcIds).sort().join(",") : "",
    st?.style || "",
    st?.length || "",
    st?.language || "",
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
    const banner = document.createElement("div");
    banner.className = "podcast-lock-banner";
    banner.innerHTML = `<span>🎙</span><div><strong>Podcast Studio 100% local (Kokoro-82M ONNX):</strong> se dedica el equipo en modo exclusivo (1.° Guion con Ollama → 2.° Liberar VRAM → 3.° Voces neuronales locales) con opción de cancelar en cualquier momento.</div>`;
    pane.appendChild(banner);

    const studio = document.createElement("div");
    studio.className = "podcast-studio-card";

    const selectedTotal = st.selectedRecIds.size + st.selectedSrcIds.size;
    const sourceChipsHtml = [
      ...readyRecs.map((r) => {
        const sel = st.selectedRecIds.has(r.id);
        return `<button type="button" class="podcast-source-chip ${sel ? "selected" : ""}" data-pod-kind="rec" data-pod-item="${escapeHtml(r.id)}">🎙 ${escapeHtml(r.filename)}</button>`;
      }),
      ...readySrcs.map((s) => {
        const sel = st.selectedSrcIds.has(s.id);
        const ico = s.source_type === "image" ? "🖼" : "📄";
        return `<button type="button" class="podcast-source-chip ${sel ? "selected" : ""}" data-pod-kind="src" data-pod-item="${escapeHtml(s.id)}">${ico} ${escapeHtml(s.filename)}</button>`;
      }),
    ].join("");

    studio.innerHTML = `
      <div class="quiz-setup-header">
        <h2>Crear episodio de Podcast local</h2>
        <p>Elige qué fuentes incluir, indica el enfoque o temas clave y selecciona el estilo de conversación entre dos conductores.</p>
      </div>
      <div class="quiz-config-grid">
        <div class="quiz-config-row">
          <div class="quiz-config-label">Fuentes a incluir (${selectedTotal} seleccionada(s))</div>
          <div class="podcast-sources-grid">${sourceChipsHtml}</div>
        </div>
        <div class="quiz-config-row">
          <div class="quiz-config-label">Temas o enfoque específico (opcional)</div>
          <textarea class="podcast-focus-input" placeholder="Ej. Enfóquense en los conceptos que entran al examen, comparen las fórmulas del pizarrón con lo que explicó el profesor y den ejemplos prácticos...">${escapeHtml(st.focus)}</textarea>
        </div>
        <div class="quiz-config-row">
          <div class="quiz-config-label">Estilo del diálogo</div>
          <div class="quiz-pill-group" data-pod-group="style">
            <button type="button" class="quiz-choice-pill ${st.style === "conversation" ? "active" : ""}" data-val="conversation">Conversación natural</button>
            <button type="button" class="quiz-choice-pill ${st.style === "exam_review" ? "active" : ""}" data-val="exam_review">Repaso de examen</button>
            <button type="button" class="quiz-choice-pill ${st.style === "debate" ? "active" : ""}" data-val="debate">Debate analítico</button>
            <button type="button" class="quiz-choice-pill ${st.style === "step_by_step" ? "active" : ""}" data-val="step_by_step">Paso a paso</button>
          </div>
        </div>
        <div class="quiz-config-row">
          <div class="quiz-config-label">Duración e idioma</div>
          <div class="field-row">
            <div class="quiz-pill-group" data-pod-group="length">
              <button type="button" class="quiz-choice-pill ${st.length === "short" ? "active" : ""}" data-val="short">Corto (~2 min)</button>
              <button type="button" class="quiz-choice-pill ${st.length === "medium" ? "active" : ""}" data-val="medium">Medio (~4 min)</button>
              <button type="button" class="quiz-choice-pill ${st.length === "long" ? "active" : ""}" data-val="long">Largo (~7 min)</button>
            </div>
            <div class="quiz-pill-group" data-pod-group="language">
              <button type="button" class="quiz-choice-pill ${st.language === "es" ? "active" : ""}" data-val="es">Español</button>
              <button type="button" class="quiz-choice-pill ${st.language === "en" ? "active" : ""}" data-val="en">English</button>
            </div>
          </div>
        </div>
      </div>
      <button type="button" class="quiz-start-btn pod-start-btn" ${selectedTotal === 0 ? "disabled" : ""}>
        <span>Generar podcast local</span>
        <span class="cta-circle" aria-hidden="true">→</span>
      </button>
    `;

    studio.querySelectorAll(".podcast-source-chip").forEach((chip) => {
      chip.onclick = () => {
        const kind = chip.dataset.podKind;
        const id = chip.dataset.podItem;
        const targetSet = kind === "rec" ? st.selectedRecIds : st.selectedSrcIds;
        if (targetSet.has(id)) targetSet.delete(id);
        else targetSet.add(id);
        renderPodcastTab(session);
      };
    });

    const focusEl = studio.querySelector(".podcast-focus-input");
    focusEl.addEventListener("input", () => {
      st.focus = focusEl.value;
    });

    ["style", "length", "language"].forEach((group) => {
      studio.querySelectorAll(`[data-pod-group="${group}"] .quiz-choice-pill`).forEach((b) => {
        b.onclick = () => {
          st[group] = b.dataset.val;
          renderPodcastTab(session);
        };
      });
    });

    studio.querySelector(".pod-start-btn").onclick = () => generatePodcastForSession(session);
    pane.appendChild(studio);
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
      const fdAudio = new FormData();
      audioFiles.forEach((f) => fdAudio.append("files", f));
      await api(`/api/sessions/${state.activeSessionId}/recordings`, {
        method: "POST",
        body: fdAudio,
      });
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

  state.isStreaming = true;
  document.getElementById("btn-send-chat").disabled = true;
  const inputEl = document.getElementById("chat-input");
  inputEl.value = "";
  inputEl.style.height = "auto";

  appendChatBubble("user", question);
  const assistantBubble = appendChatBubble("assistant", "", true);
  const scroll = document.getElementById("center-scroll");

  let accumulated = "";
  let thinkingAccum = "";
  let thinkingBoxEl = null;
  let thinkingContentEl = null;
  let answerContainerEl = null;

  try {
    const resp = await fetch(`/api/sessions/${state.activeSessionId}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
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
            accumulated += payload.text;
            answerContainerEl.innerHTML =
              renderMarkdown(accumulated, true) + '<span class="stream-cursor"></span>';
            scroll.scrollTop = scroll.scrollHeight;
          } else if (payload.type === "done") {
            if (answerContainerEl) {
              answerContainerEl.innerHTML = renderMarkdown(accumulated, true);
            }
          } else if (payload.type === "error") {
            throw new Error(payload.detail || "Error en la respuesta del modelo.");
          }
        }
      }
    }

    if (answerContainerEl) {
      answerContainerEl.innerHTML = renderMarkdown(accumulated, true);
    }

    const fresh = await api(`/api/sessions/${state.activeSessionId}`);
    state.activeSession = fresh;
    renderContextMetrics(fresh.context_metrics);
    updateChatControls(fresh);
    renderWorkspaceSubtitle(fresh);
    await loadSessions(false);
  } catch (err) {
    assistantBubble.innerHTML = `<span style="color:#fca5a5;">${escapeHtml(err.message)}</span>`;
    showToast(err.message, true);
  } finally {
    state.isStreaming = false;
    document.getElementById("btn-send-chat").disabled = false;
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
    opt.textContent = w.label;
    whisperSelect.appendChild(opt);
  }
  whisperSelect.value = settings.whisper_model || "large-v3-turbo";

  document.getElementById("set-whisper-device").value = settings.whisper_device || "auto";
  document.getElementById("set-whisper-lang").value = settings.whisper_language || "auto";
  document.getElementById("set-num-ctx").value = String(settings.num_ctx || 8192);
  document.getElementById("set-idle-unload").value = String(settings.idle_unload_sec || 180);
  document.getElementById("set-classroom-enhance").checked = settings.classroom_audio_enhance !== false;
  document.getElementById("set-enable-thinking").checked = Boolean(settings.enable_thinking);

  [
    "set-ollama-model",
    "set-whisper-model",
    "set-whisper-device",
    "set-whisper-lang",
    "set-num-ctx",
    "set-idle-unload",
  ].forEach(refreshCustomSelect);
}

function openSettingsModal() {
  if (state.podcastBusy) {
    showToast("Ajustes bloqueados mientras se genera el podcast en modo exclusivo.", true);
    return;
  }
  populateSettingsFromStatus(state.systemStatus || {});
  document.getElementById("btn-open-settings").classList.add("active");
  openModal("settings-modal");
  refreshSystemStatus({ syncSettingsModal: true });
}

function closeSettingsModal() {
  closeActiveDropdown();
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
    enable_thinking: document.getElementById("set-enable-thinking").checked,
  };
  await api("/api/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  closeSettingsModal();
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
  document.getElementById("btn-toggle-sidebar").onclick = () => {
    studioBody.classList.toggle("left-collapsed");
    setTimeout(updateTabIndicator, 250);
  };
  document.getElementById("btn-toggle-sources").onclick = () => {
    studioBody.classList.toggle("right-collapsed");
    setTimeout(updateTabIndicator, 250);
  };

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
    const recs = (state.activeSession.recordings || []).filter(
      (r) => r.transcript_text && r.enabled !== false
    );
    const selected = state.transcriptRecFilter === "all"
      ? recs
      : recs.filter((r) => r.id === state.transcriptRecFilter);
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
    if (!state.activeSessionId || state.isResummarizing) return;
    const btn = document.getElementById("btn-resummarize");
    const prevHtml = btn.innerHTML;
    state.isResummarizing = true;
    btn.disabled = true;
    btn.innerHTML = `<span class="btn-spinner"></span><span>Regenerando...</span>`;

    if (state.activeTab !== "summary") {
      switchTab("summary");
    }
    const modelName = getActiveModelLabel();
    document.getElementById("tab-summary").innerHTML = buildSkeletonHtml(
      `${modelName} está recreando el resumen...`
    );

    try {
      const fresh = await api(`/api/sessions/${state.activeSessionId}/resummarize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      state.activeSession = fresh;
      renderSummaryTab(fresh, true);
      showToast("Resumen actualizado.");
    } catch (err) {
      if (state.activeSession) renderSummaryTab(state.activeSession);
      showToast(err.message, true);
    } finally {
      state.isResummarizing = false;
      btn.disabled = false;
      btn.innerHTML = prevHtml;
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

  document.getElementById("chat-form").onsubmit = (e) => {
    e.preventDefault();
    sendChatQuestion(chatInput.value);
  };

  document.getElementById("btn-clear-chat").onclick = handleClearChatHistory;

  document.getElementById("btn-unload-vram").onclick = triggerUnloadVram;

  document.getElementById("btn-open-settings").onclick = openSettingsModal;
  document.getElementById("btn-close-settings").onclick = closeSettingsModal;
  document.getElementById("btn-cancel-settings").onclick = closeSettingsModal;
  document.getElementById("settings-form").onsubmit = (e) =>
    saveSettings(e).catch((err) => showToast(err.message, true));

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

  initAllCustomSelects();
  initEvents();
  switchTab("summary");
  await refreshSystemStatus();
  await loadSessions(true);
  setInterval(refreshSystemStatus, 3000);
}

window.addEventListener("DOMContentLoaded", initApp);
