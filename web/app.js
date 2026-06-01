// mimOE Health Triage — behavior layer.
// Talks to the FastAPI backend (server.py); no clinical logic lives here.

const $ = (id) => document.getElementById(id);

const els = {
  symptom: $("symptom"),
  patientId: $("patientId"),
  triageBtn: $("triageBtn"),
  examples: $("examples"),
  statusDot: $("statusDot"),
  statusText: $("statusText"),
  status: $("status"),
  empty: $("emptyState"),
  result: $("result"),
  urgencyBadge: $("urgencyBadge"),
  escalateFlag: $("escalateFlag"),
  action: $("action"),
  reasoning: $("reasoning"),
  usedVitals: $("usedVitals"),
  vitalsRow: $("vitalsRow"),
  vitals: $("vitals"),
  source: $("source"),
  toast: $("toast"),
};

// Normal reference ranges, used only to flag out-of-range vitals in the UI.
const NORMAL = {
  heart_rate_bpm: [60, 100],
  spo2_percent: [95, 100],
  respiratory_rate_bpm: [12, 20],
  systolic_bp_mmhg: [90, 120],
  temperature_c: [36.1, 37.2],
};
const VITAL_LABEL = {
  heart_rate_bpm: ["HR", "bpm"],
  spo2_percent: ["SpO₂", "%"],
  respiratory_rate_bpm: ["RR", "bpm"],
  systolic_bp_mmhg: ["SBP", "mmHg"],
  temperature_c: ["Temp", "°C"],
};
const SOURCE_LABEL = {
  json: "JSON (Tier 1)",
  "nl-classified": "NL-classified (Tier 2)",
  "fail-safe": "Fail-safe (Tier 3)",
};

let toastTimer = null;
function showToast(msg) {
  els.toast.textContent = msg;
  els.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (els.toast.hidden = true), 4000);
}

async function loadConfig() {
  try {
    const res = await fetch("/api/config");
    const cfg = await res.json();
    els.status.classList.add("status--ok");
    els.statusText.textContent = `${cfg.model} · on-device`;
    els.status.title = cfg.base_url;
  } catch {
    els.status.classList.add("status--down");
    els.statusText.textContent = "endpoint unreachable";
  }
}

function renderVitals(vitals) {
  if (!vitals) {
    els.vitalsRow.hidden = true;
    return;
  }
  els.vitalsRow.hidden = false;
  els.vitals.innerHTML = "";
  for (const key of Object.keys(VITAL_LABEL)) {
    if (!(key in vitals)) continue;
    const [label, unit] = VITAL_LABEL[key];
    const [lo, hi] = NORMAL[key];
    const value = vitals[key];
    const alert = value < lo || value > hi;
    const pill = document.createElement("span");
    pill.className = "vitals__pill" + (alert ? " vitals__pill--alert" : "");
    pill.textContent = `${label} ${value}${unit}`;
    els.vitals.appendChild(pill);
  }
}

function renderResult(data) {
  const r = data.result;

  const badge = els.urgencyBadge;
  badge.textContent = r.urgency;
  badge.className = "badge badge--" + r.urgency;

  els.escalateFlag.hidden = !r.escalate_to_human;
  els.action.textContent = r.recommended_action;
  els.reasoning.textContent = r.reasoning;

  els.usedVitals.textContent = data.used_vitals ? "Yes — tool called" : "No";
  renderVitals(data.vitals);
  els.source.textContent = SOURCE_LABEL[data.source] || data.source;

  els.empty.hidden = true;
  els.result.hidden = false;
}

async function runTriage() {
  const symptom = els.symptom.value.trim();
  if (!symptom) {
    showToast("Please describe the symptoms first.");
    els.symptom.focus();
    return;
  }

  els.triageBtn.disabled = true;
  els.triageBtn.classList.add("is-loading");

  try {
    const res = await fetch("/api/triage", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        symptom,
        patient_id: els.patientId.value.trim() || "demo-patient",
      }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    renderResult(await res.json());
  } catch (err) {
    showToast("Triage failed — is the server running? " + err.message);
  } finally {
    els.triageBtn.disabled = false;
    els.triageBtn.classList.remove("is-loading");
  }
}

// Wiring -------------------------------------------------------------------
els.triageBtn.addEventListener("click", runTriage);

els.examples.addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (chip) els.symptom.value = chip.dataset.symptom;
});

// Cmd/Ctrl + Enter submits from the textarea.
els.symptom.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") runTriage();
});

loadConfig();
