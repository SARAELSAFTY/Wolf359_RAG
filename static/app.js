const API_BASE = ""; // same-origin, served by FastAPI

const CHARACTER_COLORS = {
  MINKOWSKI: "var(--minkowski)",
  EIFFEL: "var(--eiffel)",
  HILBERT: "var(--hilbert)",
};

const messagesEls = {
  summarizer: document.getElementById("messages-summarizer"),
  groupchat: document.getElementById("messages-groupchat"),
};
const formEl = document.getElementById("chat-form");
const queryInput = document.getElementById("query-input");
const sendBtn = document.getElementById("send-btn");
const clearBtn = document.getElementById("clear-btn");
const modeButtons = document.querySelectorAll(".mode-btn");
const tokenUsageEl = document.getElementById("token-usage");

let currentMode = "summarizer";
let messagesEl = messagesEls.summarizer;
const usageByMode = { summarizer: null, groupchat: null };

function setMode(mode) {
  currentMode = mode;
  modeButtons.forEach((btn) => btn.classList.toggle("active", btn.dataset.mode === mode));
  queryInput.placeholder = mode === "groupchat"
    ? 'Ask the crew... (or address someone, e.g. "Hilbert, what happened?")'
    : "Ask something about Wolf 359...";
  Object.entries(messagesEls).forEach(([name, el]) => {
    el.hidden = name !== mode;
  });
  messagesEl = messagesEls[mode];
  messagesEl.scrollTop = messagesEl.scrollHeight;
  renderTokenUsage(mode);
}

function renderTokenUsage(mode = currentMode) {
  const usage = usageByMode[mode];
  if (!usage) {
    tokenUsageEl.textContent = "";
    tokenUsageEl.classList.remove("over-limit");
    tokenUsageEl.title = "";
    return;
  }

  const suffix = usage.over_limit ? " (over limit)" : "";
  tokenUsageEl.textContent = `${usage.session_tokens.toLocaleString()} / ${usage.token_limit.toLocaleString()} tokens${suffix}`;
  tokenUsageEl.classList.toggle("over-limit", usage.over_limit);
  tokenUsageEl.title = usage.over_limit
    ? "Session token budget exceeded; responses will continue."
    : "Tokens used in this session";
}

function updateTokenUsage(mode, usage) {
  if (!usage) return;
  usageByMode[mode] = usage;
  if (mode === currentMode) renderTokenUsage();
}

modeButtons.forEach((btn) => {
  btn.addEventListener("click", () => setMode(btn.dataset.mode));
});

// Separate session IDs per mode, since each mode's history has a
// different shape (see core/history.py) and switching modes mid-session
// shouldn't mix a summarizer Q&A thread with a group-chat thread.
function getSessionId(mode) {
  const key = `wolf359_session_id_${mode}`;
  let id = sessionStorage.getItem(key);
  if (!id) {
    id = crypto.randomUUID();
    sessionStorage.setItem(key, id);
  }
  return id;
}

function appendMessage(role, text, className = role, container = messagesEl) {
  const div = document.createElement("div");
  div.className = `msg ${className}`;
  div.textContent = text;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return div;
}

function appendCharacterMessage(character, text, container = messagesEl) {
  const div = document.createElement("div");
  div.className = "msg character";
  div.style.setProperty("--char-color", CHARACTER_COLORS[character] || "var(--accent)");

  const label = document.createElement("div");
  label.className = "char-label";
  label.textContent = character;

  const body = document.createElement("div");
  body.textContent = text;

  div.appendChild(label);
  div.appendChild(body);
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return div;
}

function appendSources(container, sources) {
  if (!sources || sources.length === 0) return;
  const details = document.createElement("details");
  details.className = "sources";
  const summary = document.createElement("summary");
  summary.textContent = `${sources.length} source${sources.length > 1 ? "s" : ""}`;
  details.appendChild(summary);

  for (const s of sources) {
    const item = document.createElement("div");
    item.className = "source-item";
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `${s.character ?? "Unknown"} — ep.${s.episode ?? "?"} — ${s.title ?? ""} (score ${s.rerank_score.toFixed(2)})`;
    const text = document.createElement("div");
    text.textContent = s.text;
    item.appendChild(meta);
    item.appendChild(text);
    details.appendChild(item);
  }
  container.appendChild(details);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function appendLoading(text = "Thinking...", container = messagesEl) {
  const div = document.createElement("div");
  div.className = "loading";
  div.textContent = text;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return div;
}

async function submitSummarizer(query, container) {
  const loadingEl = appendLoading("Thinking...", container);
  try {
    const response = await fetch(`${API_BASE}/summarize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: getSessionId("summarizer"), query }),
    });
    loadingEl.remove();

    if (!response.ok) {
      const errBody = await response.json().catch(() => ({}));
      appendMessage("error", `Request failed: ${errBody.detail || response.statusText}`, "error", container);
      return;
    }

    const data = await response.json();
    updateTokenUsage("summarizer", data.token_usage);
    const assistantEl = appendMessage("assistant", data.answer, "assistant", container);
    appendSources(assistantEl, data.sources);
  } catch (err) {
    loadingEl.remove();
    appendMessage("error", `Couldn't reach the server. Is it running? (${err.message})`, "error", container);
  }
}

async function submitGroupChat(query, container) {
  const loadingEl = appendLoading("The crew is responding...", container);
  try {
    const response = await fetch(`${API_BASE}/groupchat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: getSessionId("groupchat"), query }),
    });
    loadingEl.remove();

    if (!response.ok) {
      const errBody = await response.json().catch(() => ({}));
      appendMessage("error", `Request failed: ${errBody.detail || response.statusText}`, "error", container);
      return;
    }

    const data = await response.json();
    updateTokenUsage("groupchat", data.token_usage);
    let lastEl = null;
    for (let i = 0; i < data.replies.length; i++) {
      if (i > 0) await sleep(600);
      lastEl = appendCharacterMessage(data.replies[i].character, data.replies[i].text, container);
    }
    if (lastEl) appendSources(lastEl, data.sources);
  } catch (err) {
    loadingEl.remove();
    appendMessage("error", `Couldn't reach the server. Is it running? (${err.message})`, "error", container);
  }
}

formEl.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = queryInput.value.trim();
  if (!query) return;

  const requestMode = currentMode;
  const targetEl = messagesEl;
  appendMessage("user", query, "user", targetEl);
  queryInput.value = "";
  sendBtn.disabled = true;

  if (requestMode === "summarizer") {
    await submitSummarizer(query, targetEl);
  } else {
    await submitGroupChat(query, targetEl);
  }

  sendBtn.disabled = false;
  queryInput.focus();
});

clearBtn.addEventListener("click", async () => {
  const sessionId = getSessionId(currentMode);
  try {
    await fetch(`${API_BASE}/session/${sessionId}`, { method: "DELETE" });
  } catch (err) {
    console.warn("Failed to clear server-side history:", err);
  }
  sessionStorage.removeItem(`wolf359_session_id_${currentMode}`);
  messagesEl.innerHTML = "";
  usageByMode[currentMode] = null;
  renderTokenUsage();
});

// Clear both modes' server-side history when the tab/window closes.
window.addEventListener("pagehide", (event) => {
  if (event.persisted) return;

  const modes = ["summarizer", "groupchat"];
  const sessionIds = modes
    .map((mode) => sessionStorage.getItem(`wolf359_session_id_${mode}`))
    .filter(Boolean);
  if (sessionIds.length === 0) return;

  const payload = JSON.stringify({ session_ids: sessionIds });
  let sent = false;
  if (navigator.sendBeacon) {
    sent = navigator.sendBeacon(
      `${API_BASE}/session/clear`,
      new Blob([payload], { type: "application/json" })
    );
  }
  if (!sent) {
    fetch(`${API_BASE}/session/clear`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
      keepalive: true,
    }).catch(() => {});
  }

  modes.forEach((mode) => sessionStorage.removeItem(`wolf359_session_id_${mode}`));
});

// initialize
setMode("summarizer");
