"use strict";

const form = document.getElementById("ask-form");
const field = document.getElementById("question");
const goBtn = document.getElementById("ask-go");
const goLabel = goBtn.querySelector(".ask__go-label");

const alphaControl = document.getElementById("alpha-control");
const alphaInput = document.getElementById("alpha");
const alphaValue = document.getElementById("alpha-value");
const kValue = document.getElementById("k-value");

const result = document.getElementById("result");
const answerBody = document.getElementById("answer-body");
const gaugeNeedle = document.getElementById("gauge-needle");
const gaugeScore = document.getElementById("gauge-score");
const guardrail = document.getElementById("guardrail");
const guardrailVal = document.getElementById("guardrail-val");
const readingNote = document.getElementById("reading-note");
const sourcesList = document.getElementById("sources");
const sourceCount = document.getElementById("source-count");
const rankBy = document.getElementById("rank-by");

const RANK_LABEL = { hybrid: "fusion", dense: "cosine similarity", sparse: "BM25 score" };

const DEMO_CORPUS_TEXT = "6 documents · 21 chunks · dense FAISS + sparse BM25";
const MAX_UPLOAD_BYTES = 5 * 1024 * 1024;

// upload elements
const corpusbar = document.querySelector(".corpusbar");
const corpusText = document.getElementById("corpus-text");
const uploadBtn = document.getElementById("upload-btn");
const resetBtn = document.getElementById("reset-btn");
const fileInput = document.getElementById("file-input");
const uploadNote = document.getElementById("upload-note");

let corpusId = null; // null => shared demo corpus

// --- gauge dial: draw the 0..5 ticks once ---
(function drawTicks() {
  const ticks = document.getElementById("gauge-ticks");
  const cx = 100, cy = 108, rOuter = 82, rInner = 71;
  for (let i = 0; i <= 5; i++) {
    const angle = Math.PI - (i / 5) * Math.PI; // 180deg (left) -> 0deg (right)
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", cx + rInner * Math.cos(angle));
    line.setAttribute("y1", cy - rInner * Math.sin(angle));
    line.setAttribute("x2", cx + rOuter * Math.cos(angle));
    line.setAttribute("y2", cy - rOuter * Math.sin(angle));
    if (i === 0 || i === 5) line.classList.add("is-major");
    ticks.appendChild(line);
  }
})();

function setNeedle(score) {
  const deg = -90 + (score / 5) * 180;
  gaugeNeedle.style.transform = `rotate(${deg}deg)`;
}

// --- controls ---
function syncMode() {
  const mode = form.querySelector('input[name="mode"]:checked').value;
  alphaControl.dataset.disabled = mode !== "hybrid";
  alphaInput.disabled = mode !== "hybrid";
}
form.querySelectorAll('input[name="mode"]').forEach((r) => r.addEventListener("change", syncMode));
syncMode();

alphaInput.addEventListener("input", () => {
  alphaValue.textContent = Number(alphaInput.value).toFixed(2);
});

document.querySelectorAll(".stepper__btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const next = Number(kValue.textContent) + Number(btn.dataset.step);
    kValue.textContent = Math.min(10, Math.max(1, next));
  });
});

// --- upload: bring your own document ---
function setNote(text, tone) {
  uploadNote.hidden = !text;
  uploadNote.textContent = text || "";
  if (tone) uploadNote.dataset.tone = tone;
  else delete uploadNote.dataset.tone;
}

function useDemoCorpus() {
  corpusId = null;
  corpusbar.dataset.custom = "false";
  corpusText.textContent = DEMO_CORPUS_TEXT;
  resetBtn.hidden = true;
  setNote("", null);
}

uploadBtn.addEventListener("click", () => fileInput.click());
resetBtn.addEventListener("click", useDemoCorpus);

fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (!file) return;

  if (file.size > MAX_UPLOAD_BYTES) {
    setNote("That file is over the 5 MB limit.", "error");
    fileInput.value = "";
    return;
  }

  setNote(`Indexing ${file.name}…`, "busy");
  uploadBtn.disabled = true;

  const body = new FormData();
  body.append("file", file);
  try {
    const res = await fetch("/api/upload", { method: "POST", body });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || "Upload failed.");

    corpusId = data.corpus_id;
    corpusbar.dataset.custom = "true";
    corpusText.textContent = `${data.filename} · ${data.chunks} chunks · your document`;
    resetBtn.hidden = false;
    setNote(`Indexed ${data.filename}. Ask it anything below.`, null);
  } catch (err) {
    setNote(err.message || "Upload failed.", "error");
  } finally {
    uploadBtn.disabled = false;
    fileInput.value = "";
  }
});

// Enter submits; Shift+Enter for a newline.
field.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

// --- rendering ---
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

function formatAnswer(text) {
  return text
    .split(/\n{2,}/)
    .map((para) => {
      const safe = escapeHtml(para.trim()).replace(/`([^`]+)`/g, "<code>$1</code>");
      return `<p>${safe.replace(/\n/g, "<br>")}</p>`;
    })
    .join("");
}

function renderSources(sources) {
  sourcesList.innerHTML = "";
  sources.forEach((s, i) => {
    const li = document.createElement("li");
    const details = document.createElement("details");
    details.className = "source";
    details.style.animationDelay = `${i * 60}ms`;

    const loc = s.page ? `${s.type} · p${s.page}` : s.type;
    const isImage = s.type === "image";
    details.innerHTML = `
      <summary>
        <span class="source__rank">${String(s.rank).padStart(2, "0")}</span>
        <span class="source__id">${escapeHtml(s.chunk_id)}</span>
        <span class="source__tags">
          <span class="tag${isImage ? " is-image" : ""}">${escapeHtml(loc)}</span>
          <span class="source__chevron" aria-hidden="true">›</span>
        </span>
      </summary>
      <p class="source__text">${escapeHtml(s.text)}</p>`;
    li.appendChild(details);
    sourcesList.appendChild(li);
  });
}

function renderResult(data) {
  answerBody.className = "paper__body";
  answerBody.innerHTML = formatAnswer(data.answer);

  const score = data.faithfulness.score;
  gaugeScore.textContent = score;
  setNeedle(score);

  if (data.flagged) {
    guardrail.dataset.state = "flag";
    guardrailVal.textContent = "flagged";
    const claims = data.faithfulness.unsupported_claims || [];
    readingNote.textContent = claims.length
      ? `Unsupported: ${claims.join("; ")}`
      : data.faithfulness.reasoning;
  } else {
    guardrail.dataset.state = "ok";
    guardrailVal.textContent = "grounded";
    readingNote.textContent = "";
  }

  rankBy.textContent = RANK_LABEL[data.mode] || data.mode;
  sourceCount.textContent = data.sources.length;
  renderSources(data.sources);
}

function showPaperMessage(text, muted) {
  result.hidden = false;
  answerBody.className = muted ? "paper__body is-muted" : "paper__body";
  answerBody.innerHTML = `<p>${escapeHtml(text)}</p>`;
  gaugeScore.textContent = "–";
  setNeedle(0);
  guardrail.dataset.state = "idle";
  guardrailVal.textContent = "—";
  readingNote.textContent = "";
  sourcesList.innerHTML = "";
  sourceCount.textContent = "0";
}

// --- submit ---
let busy = false;

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (busy) return;

  const question = field.value.trim();
  if (!question) return;

  busy = true;
  goBtn.disabled = true;
  goLabel.textContent = "Consulting…";
  showPaperMessage("Retrieving from the index and grounding the answer…", true);

  const payload = {
    question,
    mode: form.querySelector('input[name="mode"]:checked').value,
    k: Number(kValue.textContent),
    alpha: Number(alphaInput.value),
    corpus_id: corpusId,
  };

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      if (res.status === 404) useDemoCorpus(); // uploaded corpus expired — fall back
      throw new Error(detail.detail || "Something went wrong retrieving the answer.");
    }
    renderResult(await res.json());
  } catch (err) {
    showPaperMessage(err.message || "The request failed. Try again.", true);
    guardrail.dataset.state = "idle";
  } finally {
    busy = false;
    goBtn.disabled = false;
    goLabel.textContent = "Ask";
    result.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
});
