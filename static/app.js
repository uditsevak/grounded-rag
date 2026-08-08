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
  };

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
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
