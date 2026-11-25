/* TransCrypting Frontend — Bento Lime/Obsidian/Porcelain (100% local) */

const TAB_ORDER = ["summary", "transcript", "chat", "quiz"];

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
};

const audioPlayers = new Map();
const sessionQuizzes = new Map();
const customSelectControllers = new Map();
let activeDropdownMenu = null;
let activeDropdownWrapper = null;

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
  return `<button type="button" class="chat-ts-pill" data-seconds="${seconds}" data-label="${clean}" title="Ir al momento ${clean}"><span class="ts-play-tri">▶</span>${clean}</button>`;
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

  const parts = masked.split(/(<[^>]+>)/g);
  for (let i = 0; i < parts.length; i += 2) {
    parts[i] = parts[i].replace(bracketRe, (_, inner) => {
      const matches = inner.match(new RegExp(tsSingle, "g"));
      if (!matches || matches.length === 0) return `[${inner}]`;
      return matches.map(makeTimestampPillHtml).join(" ");
    });
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
  const heroTitle = document.getElementById("hero-status-title");
  const heroFill = document.getElementById("hero-progress-fill");
  const heroSub = document.getElementById("hero-substatus");

  const vramPct = Math.round(hw.vram_percent || 0);

  if (res.phase === "whisper") {
    pill.className = "mini-status active";
    pillText.textContent = `Whisper ${(res.active_device || "").toUpperCase()}`;
    heroTitle.textContent = "Transcribiendo audio...";
    heroFill.style.width = `${Math.max(25, vramPct)}%`;
    heroSub.textContent = `${res.active_model} · VRAM ${vramPct}%`;
  } else if (res.phase === "llm") {
    pill.className = "mini-status active";
    pillText.textContent = "Modelo activo";
    heroTitle.textContent = `Analizando con ${res.active_model || activeLabel}`;
    heroFill.style.width = `${Math.max(35, vramPct)}%`;
    heroSub.textContent = `${res.active_model} · VRAM ${vramPct}%`;
  } else if (res.phase === "unloading") {
    pill.className = "mini-status active";
    pillText.textContent = "Liberando...";
    heroTitle.textContent = "Liberando memoria...";
    heroFill.style.width = "15%";
    heroSub.textContent = "Vaciando VRAM y RAM";
  } else if (runningOllama.length > 0) {
    pill.className = "mini-status active";
    pillText.textContent = "En VRAM";
    heroTitle.textContent = "Listo para preguntar";
    heroFill.style.width = `${Math.max(20, vramPct)}%`;
    heroSub.textContent = `En caché: ${runningOllama[0].name} · VRAM ${vramPct}%`;
  } else {
    pill.className = "mini-status idle";
    pillText.textContent = "VRAM libre";
    heroTitle.textContent = "Listo para analizar";
    heroFill.style.width = `${Math.max(8, vramPct)}%`;
    heroSub.textContent = `En espera · VRAM ${vramPct}% · 100% local`;
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

function switchTab(tabName) {
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

  requestAnimationFrame(updateTabIndicator);

  if (tabName === "chat") {
    const scroll = document.getElementById("center-scroll");
    scroll.scrollTop = scroll.scrollHeight;
    if (state.activeSession) {
      ensureDynamicSuggestions(state.activeSession);
    }
  } else if (tabName === "quiz" && state.activeSession) {
    renderQuizTab(state.activeSession);
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
    meta.textContent = `${s.recording_count || 0} audio(s) · ${s.message_count || 0} msg`;

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
}

async function createNewSession() {
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
}

async function selectSession(sessionId) {
  if (state.activeSessionId !== sessionId) {
    stopAllOtherAudio(null);
    state.activeRecordingId = null;
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

  const processing = (session.recordings || []).some((r) =>
    ["queued", "preparing", "transcribing", "summarizing"].includes(r.status)
  );
  if (processing) {
    state.pollTimer = setTimeout(async () => {
      if (state.activeSessionId === session.id) {
        const fresh = await api(`/api/sessions/${session.id}`);
        state.activeSession = fresh;
        renderRecordings(fresh.recordings || []);
        renderStatsTiles(fresh.recordings || []);
        renderContextMetrics(fresh.context_metrics);
        if (!state.isResummarizing) {
          renderSummaryTab(fresh);
        }
        renderTranscriptTab(fresh);
        if (!getQuizState(fresh.id).questions) {
          renderQuizTab(fresh);
        }
        renderWorkspaceSubtitle(fresh);
        updateChatControls(fresh);
        loadSessions(false).catch(() => {});
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
    renderStatsTiles([]);
    renderRecordings([]);
    renderContextMetrics(null);
    return;
  }

  emptyEl.classList.add("hidden");
  activeEl.classList.remove("hidden");

  document.getElementById("session-title").textContent = session.title;
  renderRecordings(session.recordings || []);
  renderStatsTiles(session.recordings || []);
  renderContextMetrics(session.context_metrics);
  if (!state.isResummarizing) {
    renderSummaryTab(session);
  }
  renderTranscriptTab(session);
  renderQuizTab(session);
  renderWorkspaceSubtitle(session);
  renderChatMessages(session.messages || []);
  updateChatControls(session);
  requestAnimationFrame(updateTabIndicator);
}

function renderStatsTiles(recordings) {
  const count = recordings.length;
  const totalSec = recordings.reduce((acc, r) => acc + (Number(r.duration_sec) || 0), 0);
  document.getElementById("stat-audio-count").textContent = String(count);
  document.getElementById("stat-audio-duration").textContent = `${formatDuration(totalSec)} total`;
}

function renderWorkspaceSubtitle(session) {
  const sub = document.getElementById("session-subtitle");
  const readyCount = (session.recordings || []).filter((r) => r.status === "ready").length;
  const totalCount = (session.recordings || []).length;
  const modelName = getActiveModelLabel();
  const mode = session.context_metrics?.mode || state.lastContextMode;
  const modeLabel = mode === "retrieval" ? "RAG Protegido" : "Contexto completo";

  if (totalCount === 0) {
    sub.textContent = "Adjunta una grabación para comenzar";
  } else if (readyCount === 0) {
    sub.textContent = `Procesando ${totalCount} grabación(es)...`;
  } else {
    sub.textContent = `${modelName} · ${readyCount} audio(s) en contexto · ${modeLabel}`;
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

function renderRecordings(recordings) {
  const list = document.getElementById("recordings-list");
  list.innerHTML = "";

  if (!recordings || recordings.length === 0) {
    const empty = document.createElement("div");
    empty.className = "rec-empty";
    empty.textContent = "Aún no hay grabaciones en esta conversación.";
    list.appendChild(empty);
    return;
  }

  if (!recordings.some((r) => r.id === state.activeRecordingId)) {
    state.activeRecordingId = recordings[0].id;
  }

  for (const rec of recordings) {
    const audio = getOrCreateAudio(rec);
    const isActive = rec.id === state.activeRecordingId;
    const isWorking = ["queued", "preparing", "transcribing", "summarizing"].includes(rec.status);

    const card = document.createElement("div");
    card.className = `rec-card ${isActive ? "active" : ""}`;
    card.dataset.recId = rec.id;
    card.dataset.durationSec = String(rec.duration_sec || 0);

    card.onclick = () => {
      if (state.activeRecordingId !== rec.id) {
        state.activeRecordingId = rec.id;
        renderRecordings(recordings);
      }
    };

    const top = document.createElement("div");
    top.className = "rec-top";

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
      renderRecordings(recordings);
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

    top.appendChild(playBtn);
    top.appendChild(info);
    top.appendChild(rightBtns);
    card.appendChild(top);

    if (isWorking) {
      const stageLine = document.createElement("div");
      stageLine.style.fontSize = "11.5px";
      stageLine.style.opacity = "0.85";
      stageLine.textContent = rec.stage || "Procesando...";
      card.appendChild(stageLine);

      const bar = document.createElement("div");
      bar.className = "scrubber-track";
      const fill = document.createElement("div");
      fill.className = "scrubber-fill working-shimmer";
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
  const readyRecs = recordings.filter((r) => r.status === "ready");
  const workingRec = recordings.find((r) =>
    ["queued", "preparing", "transcribing", "summarizing"].includes(r.status)
  );

  if (recordings.length === 0) {
    pane.innerHTML = `<div class="chat-placeholder"><h3>Sin grabaciones</h3><p>Adjunta una grabación desde el botón verde lima o arrástrala aquí para generar su transcripción y resumen.</p></div>`;
    return;
  }

  if (readyRecs.length === 0) {
    const modelName = getActiveModelLabel();
    const stageMsg = workingRec
      ? `${workingRec.stage || "Procesando"} (${Math.round((workingRec.progress || 0) * 100)}%) — ${modelName} preparará el resumen...`
      : `${modelName} está generando el resumen...`;
    pane.innerHTML = buildSkeletonHtml(stageMsg);
    return;
  }

  const revealClass = animateReveal ? "summary-reveal" : "";
  const parts = [];
  if (readyRecs.length >= 2 && session.combined_summary) {
    parts.push(`
      <div class="content-card combined ${revealClass}">
        <div class="content-card-header">
          <span>Resumen integrado de la conversación</span>
          <span class="card-badge lime">${readyRecs.length} grabaciones</span>
        </div>
        <div class="md">${renderMarkdown(session.combined_summary, true)}</div>
      </div>
    `);
  }

  for (const rec of readyRecs) {
    parts.push(`
      <div class="content-card ${revealClass}">
        <div class="content-card-header">
          <span>${escapeHtml(rec.filename)}</span>
          <span class="card-badge">${formatDuration(rec.duration_sec)} · ${escapeHtml(rec.whisper_model || "whisper")}</span>
        </div>
        <div class="md">${renderMarkdown(rec.summary_text || "_Sin resumen._", true)}</div>
      </div>
    `);
  }

  pane.innerHTML = parts.join("");
}

function renderTranscriptTab(session) {
  const pane = document.getElementById("tab-transcript");
  const query = (document.getElementById("transcript-search").value || "").trim().toLowerCase();
  const recordings = (session.recordings || []).filter((r) => r.transcript_text);

  if (recordings.length === 0) {
    pane.innerHTML = `<div class="chat-placeholder"><h3>Transcripción pendiente</h3><p>El texto aparecerá aquí con marcas de tiempo clicables cuando Whisper termine.</p></div>`;
    return;
  }

  pane.innerHTML = "";
  for (const rec of recordings) {
    const card = document.createElement("div");
    card.className = "content-card";

    const header = document.createElement("div");
    header.className = "content-card-header";
    header.innerHTML = `<span>${escapeHtml(rec.filename)}</span><span class="card-badge">Clic en tiempo para reproducir</span>`;
    card.appendChild(header);

    const segs = rec.segments || [];
    if (segs.length > 0) {
      for (const seg of segs) {
        const text = seg.text || "";
        if (query && !text.toLowerCase().includes(query)) continue;

        const row = document.createElement("div");
        row.className = "segment-row";

        const tsBtn = document.createElement("button");
        tsBtn.className = "timestamp-btn";
        tsBtn.textContent = formatDuration(seg.start);
        tsBtn.onclick = () => seekRecording(rec.id, seg.start, { pulse: true });

        const textEl = document.createElement("div");
        textEl.className = "segment-text";
        if (query) {
          const safe = escapeHtml(text);
          const re = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "gi");
          textEl.innerHTML = safe.replace(re, '<mark class="hl">$1</mark>');
        } else {
          textEl.textContent = text;
        }

        row.appendChild(tsBtn);
        row.appendChild(textEl);
        card.appendChild(row);
      }
    } else {
      const p = document.createElement("div");
      p.className = "segment-text";
      p.textContent = rec.transcript_text;
      card.appendChild(p);
    }

    pane.appendChild(card);
  }
}

// --- Interactive Exam / Quiz Mode ---

function getQuizState(sessionId) {
  let qs = sessionQuizzes.get(sessionId);
  if (!qs) {
    qs = { count: 5, mode: "mixed", questions: null, answers: {}, drafts: {}, loading: false };
    sessionQuizzes.set(sessionId, qs);
  }
  return qs;
}

async function generateQuizForSession(session) {
  if (!session) return;
  const qs = getQuizState(session.id);
  if (qs.loading) return;

  qs.loading = true;
  qs.questions = null;
  qs.answers = {};
  qs.drafts = {};
  renderQuizTab(session);

  try {
    const res = await api(`/api/sessions/${session.id}/quiz/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ num_questions: qs.count, mode: qs.mode }),
    });
    qs.questions = res.questions || [];
    qs.answers = {};
    qs.drafts = {};
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
  let scoreSum = 0;

  for (const q of questions) {
    const ans = qs.answers[q.id];
    if (!ans) continue;
    if (ans.type === "mcq") {
      answeredCount += 1;
      scoreSum += ans.isCorrect ? 100 : 0;
    } else if (ans.type === "open" && !ans.grading && ans.result) {
      answeredCount += 1;
      scoreSum += Number(ans.result.score || 0);
    }
  }

  const progressPct = Math.round((answeredCount / Math.max(1, total)) * 100);
  const avgScore = answeredCount > 0 ? Math.round(scoreSum / answeredCount) : 0;
  return { total, answeredCount, progressPct, avgScore };
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

  const top = document.createElement("div");
  top.className = "quiz-card-top";
  top.innerHTML = `
    <span class="quiz-q-num">Pregunta ${idx + 1} de ${total}</span>
    <span class="card-badge">${q.type === "mcq" ? "Opción múltiple" : "Respuesta abierta"}</span>
  `;
  card.appendChild(top);

  const qText = document.createElement("div");
  qText.className = "quiz-q-text";
  qText.textContent = q.question;
  card.appendChild(qText);

  if (q.type === "mcq") {
    const optsWrap = document.createElement("div");
    optsWrap.className = "quiz-options";

    (q.options || []).forEach((optText, optIdx) => {
      const optBtn = document.createElement("button");
      optBtn.type = "button";
      optBtn.className = "quiz-option";
      if (ans) {
        optBtn.disabled = true;
        if (optIdx === q.correct_index) {
          optBtn.classList.add("correct");
        } else if (optIdx === ans.selectedIndex && !ans.isCorrect) {
          optBtn.classList.add("wrong");
        }
      }
      optBtn.innerHTML = `<span class="quiz-option-letter">${letters[optIdx] || optIdx + 1}</span><span>${escapeHtml(optText)}</span>`;
      if (!ans) {
        optBtn.onclick = () => submitMcqAnswer(session, q, optIdx);
      }
      optsWrap.appendChild(optBtn);
    });
    card.appendChild(optsWrap);

    if (ans) {
      const fb = document.createElement("div");
      fb.className = "quiz-feedback";
      const tsPill = q.timestamp ? ` ${makeTimestampPillHtml(q.timestamp)}` : "";
      const verdictHead = ans.isCorrect ? "✓ Correcto" : "✕ Incorrecto";
      const expl = q.explanation ? escapeHtml(q.explanation) : `La opción correcta es ${letters[q.correct_index]}.`;
      fb.innerHTML = `<strong>${verdictHead}</strong><div>${expl}${tsPill}</div>`;
      card.appendChild(fb);
    }
  } else {
    const openWrap = document.createElement("div");
    openWrap.className = "quiz-open-wrap";

    const textarea = document.createElement("textarea");
    textarea.className = "quiz-open-input";
    textarea.placeholder = "Escribe tu respuesta basada en lo explicado en clase...";
    textarea.value = ans ? ans.userAnswer || "" : qs.drafts[q.id] || "";
    textarea.disabled = Boolean(ans);
    if (!ans) {
      textarea.addEventListener("input", () => {
        qs.drafts[q.id] = textarea.value;
      });
    }

    openWrap.appendChild(textarea);

    if (!ans) {
      const actions = document.createElement("div");
      actions.className = "quiz-open-actions";
      const evalBtn = document.createElement("button");
      evalBtn.type = "button";
      evalBtn.className = "pill-btn dark sm";
      evalBtn.textContent = "Evaluar respuesta";
      evalBtn.onclick = () => submitOpenAnswer(session, q, textarea.value);
      actions.appendChild(evalBtn);
      openWrap.appendChild(actions);
    } else if (ans.grading) {
      const loadingBox = document.createElement("div");
      loadingBox.className = "quiz-grade-box";
      loadingBox.innerHTML = `
        <div class="thinking-indicator">
          <div class="think-dots" aria-hidden="true"><span></span><span></span><span></span></div>
          <span class="think-label">Evaluando tu respuesta con la grabación...</span>
        </div>
      `;
      openWrap.appendChild(loadingBox);
    } else if (ans.result) {
      const res = ans.result;
      const gradeBox = document.createElement("div");
      gradeBox.className = `quiz-grade-box ${res.verdict}`;
      const verdictMap = {
        correct: "Correcta",
        partial: "Parcialmente correcta",
        incorrect: "Por mejorar",
      };
      const tsVal = res.timestamp || q.timestamp || "";
      const tsPill = tsVal ? ` ${makeTimestampPillHtml(tsVal)}` : "";
      const missingHtml = res.missing_points
        ? `<div><strong>Te faltó mencionar:</strong> ${escapeHtml(res.missing_points)}</div>`
        : "";
      const expectedHtml = q.expected_answer
        ? `<div><strong>Referencia de clase:</strong> ${escapeHtml(q.expected_answer)}${tsPill}</div>`
        : tsPill
        ? `<div><strong>Momento en el audio:</strong>${tsPill}</div>`
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
    }

    card.appendChild(openWrap);
  }

  return card;
}

function buildQuizSummaryNode(session, qs, avgScore) {
  const sumCard = document.createElement("div");
  sumCard.className = "quiz-summary-card";
  sumCard.innerHTML = `
    <div>
      <div style="font-size:13px; opacity:0.78;">Examen completado</div>
      <div style="font-size:20px; font-weight:700;">Calificación final: ${avgScore} / 100</div>
    </div>
    <button type="button" class="pill-btn dark-muted">Generar otro examen</button>
  `;
  sumCard.querySelector("button").onclick = () => {
    qs.questions = null;
    qs.answers = {};
    qs.drafts = {};
    renderQuizTab(session);
  };
  return sumCard;
}

function updateQuizProgressAndSummary(session) {
  const pane = document.getElementById("tab-quiz");
  const qs = getQuizState(session.id);
  if (!pane || !qs.questions) return;

  const { total, answeredCount, progressPct, avgScore } = computeQuizStats(qs);
  const progCard = pane.querySelector(".quiz-progress-card");
  if (progCard) {
    const spans = progCard.querySelectorAll(".quiz-progress-top span");
    if (spans[0]) spans[0].textContent = `Progreso del examen: ${answeredCount} / ${total}`;
    if (spans[1]) {
      spans[1].textContent = answeredCount > 0
        ? `Promedio: ${avgScore}/100`
        : "Selecciona o redacta tus respuestas";
    }
    const fill = progCard.querySelector(".quiz-progress-fill");
    if (fill) fill.style.width = `${progressPct}%`;
  }

  const existingSummary = pane.querySelector(".quiz-summary-card");
  if (answeredCount === total && !existingSummary) {
    pane.appendChild(buildQuizSummaryNode(session, qs, avgScore));
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
}

function submitMcqAnswer(session, q, selectedIndex) {
  const qs = getQuizState(session.id);
  if (qs.answers[q.id]) return;
  const isCorrect = selectedIndex === q.correct_index;
  qs.answers[q.id] = {
    type: "mcq",
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

function renderQuizTab(session) {
  const pane = document.getElementById("tab-quiz");
  const resetBtn = document.getElementById("btn-reset-quiz");
  if (!pane) return;

  const readyRecs = (session?.recordings || []).filter((r) => r.status === "ready");
  if (!session || readyRecs.length === 0) {
    if (resetBtn) resetBtn.classList.add("hidden");
    pane.innerHTML = `<div class="chat-placeholder"><h3>Examen de repaso</h3><p>Cuando tus grabaciones estén listas podrás generar aquí preguntas de opción múltiple y abiertas basadas en lo que dijo el profesor en clase.</p></div>`;
    return;
  }

  const qs = getQuizState(session.id);

  if (qs.loading) {
    if (resetBtn) resetBtn.classList.add("hidden");
    const modelName = getActiveModelLabel();
    pane.innerHTML = buildSkeletonHtml(`${modelName} está diseñando tu examen de repaso...`);
    return;
  }

  if (!qs.questions || qs.questions.length === 0) {
    if (resetBtn) resetBtn.classList.add("hidden");
    pane.innerHTML = "";
    const setup = document.createElement("div");
    setup.className = "quiz-setup-card";
    setup.innerHTML = `
      <div class="quiz-setup-header">
        <h2>Simulador de examen de clase</h2>
        <p>Pon a prueba lo que aprendiste con preguntas creadas únicamente a partir de tus grabaciones, con corrección instantánea y enlace directo al minuto exacto de la clase.</p>
      </div>
      <div class="quiz-config-grid">
        <div class="quiz-config-row">
          <div class="quiz-config-label">Cantidad de preguntas</div>
          <div class="quiz-pill-group" data-quiz-group="count">
            <button type="button" class="quiz-choice-pill ${qs.count === 3 ? "active" : ""}" data-val="3">3 preguntas</button>
            <button type="button" class="quiz-choice-pill ${qs.count === 5 ? "active" : ""}" data-val="5">5 preguntas</button>
            <button type="button" class="quiz-choice-pill ${qs.count === 8 ? "active" : ""}" data-val="8">8 preguntas</button>
          </div>
        </div>
        <div class="quiz-config-row">
          <div class="quiz-config-label">Formato</div>
          <div class="quiz-pill-group" data-quiz-group="mode">
            <button type="button" class="quiz-choice-pill ${qs.mode === "mixed" ? "active" : ""}" data-val="mixed">Mixto (opción múltiple + abierta)</button>
            <button type="button" class="quiz-choice-pill ${qs.mode === "mcq" ? "active" : ""}" data-val="mcq">Solo opción múltiple</button>
            <button type="button" class="quiz-choice-pill ${qs.mode === "open" ? "active" : ""}" data-val="open">Solo respuesta abierta</button>
          </div>
        </div>
      </div>
      <button type="button" id="btn-generate-quiz" class="quiz-start-btn">
        <span>Generar examen de repaso</span>
        <span class="cta-circle" aria-hidden="true">→</span>
      </button>
    `;

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
    setup.querySelector("#btn-generate-quiz").onclick = () => generateQuizForSession(session);
    pane.appendChild(setup);
    return;
  }

  if (resetBtn) resetBtn.classList.remove("hidden");

  const { total, answeredCount, progressPct, avgScore } = computeQuizStats(qs);
  pane.innerHTML = "";

  const progCard = document.createElement("div");
  progCard.className = "quiz-progress-card";
  progCard.innerHTML = `
    <div class="quiz-progress-top">
      <span>Progreso del examen: ${answeredCount} / ${total}</span>
      <span>${answeredCount > 0 ? `Promedio: ${avgScore}/100` : "Selecciona o redacta tus respuestas"}</span>
    </div>
    <div class="quiz-progress-track">
      <div class="quiz-progress-fill" style="width:${progressPct}%"></div>
    </div>
  `;
  pane.appendChild(progCard);

  qs.questions.forEach((q, idx) => {
    pane.appendChild(buildQuizQuestionCard(session, q, idx, total, true));
  });

  if (answeredCount === total) {
    pane.appendChild(buildQuizSummaryNode(session, qs, avgScore));
  }
}

// --- Upload Recordings ---

async function uploadFiles(fileList) {
  if (!fileList || fileList.length === 0) return;

  if (!state.activeSessionId) {
    await createNewSession();
  }

  const formData = new FormData();
  for (const f of fileList) {
    formData.append("files", f);
  }

  try {
    showToast(`Subiendo ${fileList.length} archivo(s) localmente...`);
    await api(`/api/sessions/${state.activeSessionId}/recordings`, {
      method: "POST",
      body: formData,
    });
    await selectSession(state.activeSessionId);
    await loadSessions(false);
  } catch (err) {
    showToast(err.message, true);
  }
}

// --- Dynamic AI Quick Prompts ---

async function ensureDynamicSuggestions(session) {
  if (!session) return;
  const readyCount = (session.recordings || []).filter((r) => r.status === "ready").length;
  if (readyCount === 0) return;
  if (Array.isArray(session.suggested_prompts) && session.suggested_prompts.length > 0) return;
  if (state.fetchingSuggestionsFor.has(session.id) || state.isStreaming) return;

  state.fetchingSuggestionsFor.add(session.id);
  const quick = document.getElementById("quick-prompts");
  quick.classList.remove("hidden");
  quick.innerHTML = `
    <div class="prompt-chip-skel"></div>
    <div class="prompt-chip-skel"></div>
    <div class="prompt-chip-skel"></div>
  `;

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
  }
}

function renderQuickPrompts(session) {
  const quick = document.getElementById("quick-prompts");
  const readyCount = (session?.recordings || []).filter((r) => r.status === "ready").length;
  if (!session || readyCount === 0) {
    quick.classList.add("hidden");
    quick.innerHTML = "";
    return;
  }

  quick.classList.remove("hidden");
  if (state.fetchingSuggestionsFor.has(session.id)) {
    return;
  }

  const suggestions = (session.suggested_prompts && session.suggested_prompts.length > 0)
    ? session.suggested_prompts
    : [
        {
          label: "Puntos clave",
          prompt: "¿Cuáles son los puntos más importantes de la grabación?",
        },
        {
          label: "Ponme a prueba",
          prompt: "Hazme 5 preguntas tipo examen basadas en lo que se dijo en la clase.",
        },
        {
          label: readyCount >= 2 ? "Comparar audios" : "Tareas y fechas",
          prompt: readyCount >= 2
            ? "Si hay varias grabaciones, ¿qué cambia o se agrega entre una y otra?"
            : "¿Qué tareas, fechas o acuerdos se mencionaron?",
        },
      ];

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
    if (!pill) return;
    const seconds = Number(pill.dataset.seconds || 0);
    const recId = resolveRecordingForTimestamp(pill, seconds);
    if (recId) {
      seekRecording(recId, seconds, { pulse: true });
    }
  });

  document.getElementById("transcript-search").oninput = () => {
    if (state.activeSession) renderTranscriptTab(state.activeSession);
  };

  document.getElementById("btn-copy-transcript").onclick = async () => {
    if (!state.activeSession) return;
    const text = (state.activeSession.recordings || [])
      .filter((r) => r.transcript_text)
      .map((r) => `=== ${r.filename} ===\n${r.transcript_text}`)
      .join("\n\n");
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
    qs.questions = null;
    qs.answers = {};
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
  document.getElementById("btn-hero-unload").onclick = triggerUnloadVram;

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
  });

  ["settings-modal", "models-guide-modal", "confirm-modal", "rename-modal"].forEach((id) => {
    document.getElementById(id).addEventListener("click", (e) => {
      if (e.target.id === id) {
        if (id === "settings-modal") closeSettingsModal();
        else if (id === "models-guide-modal") closeModal("models-guide-modal");
        else if (id === "confirm-modal") closeConfirm();
        else if (id === "rename-modal") closeRenameModal();
      }
    });
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (activeDropdownMenu) {
        closeActiveDropdown();
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
