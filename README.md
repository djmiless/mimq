# mimOE Health Triage Agent

A small, on-device AI agent that triages a patient symptom report and returns a
**structured, typed result** — urgency level, recommended action, an
escalate-to-human flag, and reasoning. It connects to a local
[mimOE](https://mimik.com) inference endpoint (OpenAI-compatible) using the
**BYO Framework** approach with **LangChain**, and runs **entirely on-device —
no cloud fallback**.

This is a deliberately compact agent. The interesting part is not its size but
the design decisions behind making it actually work against a tiny edge model.

---

## Why this agent

It maps directly to two things:

- **Miles CAM** — my overdose-detection project, where an edge device must make
  a fast, private, structured urgency decision and know when to escalate to a
  human.
- **mimik's health-platform vision** — edge-native, privacy-first inference with
  structured, machine-consumable outputs rather than free text.

A triage agent is the smallest honest example of that pattern: read sensor data
on-device, reason locally, and emit a typed decision a downstream system can act
on — without any data leaving the device.

---

## What it does

For each symptom report the agent runs a three-step loop:

1. **DECIDE** — the model decides whether it needs vital signs before triaging.
2. **ACT** — if so, it calls the `get_vital_signs` tool (simulated on-device
   sensor read).
3. **TRIAGE** — it returns a validated `TriageResult` (Pydantic):

```python
class TriageResult(BaseModel):
    urgency: Literal["low", "medium", "high"]
    recommended_action: str
    escalate_to_human: bool
    reasoning: str
```

### Example

```
[case 2] patient=patient-acute-07
  symptom: Sudden crushing chest pain spreading to my left arm and short of breath.
  [decision] read vital signs? YES
  [tool]     get_vital_signs -> HR=74bpm, SpO2=96%, RR=11bpm, SBP=83mmHg, Temp=36.5C
  --- TriageResult (source: nl-classified) ---
  urgency            : HIGH
  recommended_action : Seek emergency care now — call emergency services.
  escalate_to_human  : True  (ESCALATE TO HUMAN)
  reasoning          : The symptoms ... are consistent with a serious cardiac/respiratory event requiring immediate medical attention.
```

---

## The key design decision: structured output from a 360M model

The target model is `smollm-360m` served by mimOE. Before writing the agent I
probed the endpoint, and found two things that shape the entire design:

- It **does not support OpenAI function-calling**. Passing `tools` is ignored —
  the model never emits `tool_calls`, it just answers in prose.
- It **does not honor JSON mode** (`response_format: json_object`) or a fixed
  output format. Asked for JSON it returns prose or echoes the schema; asked for
  one word it writes three paragraphs.

That means the "obvious" LangChain path — `.bind_tools()` and
`.with_structured_output()` — **silently fails on this model**, because both are
built on function-calling under the hood. A naive implementation would look
correct and never produce a valid object.

So structured output is produced through a **capability ladder**, with a
clinical **safety override** on top:

| Tier | Strategy | When it fires |
|------|----------|---------------|
| 1 | Parse a JSON object from the response (`PydanticOutputParser` + tolerant extractor) | Capable models that *can* emit JSON |
| 2 | Classify the model's **natural-language** assessment into the schema (negation-aware keyword scoring) | Tiny models like `smollm-360m` |
| 3 | Rule-based **fail-safe** that always escalates to a human | Model unreachable or empty |

**Safety override:** deterministic rules — red-flag symptoms (e.g. "chest pain",
"overdose", "unresponsive") and abnormal vital signs — may only **raise**
urgency and **force escalation**, never lower them. *The model proposes; the
safety rules can override upward.* For a health system that is the correct
asymmetry: a false "high" wastes a clinician's minute, a false "low" can kill.

Tier 2 is what actually runs against `smollm-360m` today. The model genuinely
does the clinical reasoning (which small models are reasonably good at); the
code is responsible only for turning that prose into a typed, validated object.
Tier 1 means the same agent upgrades automatically — point `MODEL_NAME` at a
capable model and it will start producing JSON directly with no code changes.

### Other choices

- **LangChain + `ChatOpenAI(base_url=...)`** — minimal code to talk to any
  OpenAI-compatible endpoint; this is the "BYO Framework" connection point.
- **Pydantic schema** — the agent's contract with downstream systems is a typed
  object, not a string.
- **`temperature=0.1`** — clinical outputs should be as consistent as possible.
- **`max_tokens=256`** — tiny models can loop/repeat indefinitely; capping output
  bounds latency and prevents runaway generations.
- **`max_retries=0`, no cloud client** — strictly on-device. If the local model
  is unreachable the agent fails safe (Tier 3); it never reaches out to a cloud
  provider.

---

## Project layout

```
.
├── agent.py         # core agent: DECIDE → ACT → TRIAGE loop (run_triage / triage_symptom)
├── schemas.py       # Pydantic TriageResult output schema
├── tools.py         # get_vital_signs tool (simulated sensor data) + safety helpers
├── server.py        # thin FastAPI layer: serves the UI, exposes /api/triage
├── web/             # browser frontend (separation of concerns)
│   ├── index.html   #   structure
│   ├── styles.css   #   presentation (light/dark, responsive)
│   └── app.js       #   behavior (calls the API; no clinical logic here)
├── .env             # local config (gitignored)
├── .env.example     # config template
├── requirements.txt # langchain, langchain-openai, pydantic, python-dotenv, fastapi, uvicorn
├── .gitignore       # excludes .env
└── README.md
```

**Separation of concerns.** The clinical logic lives only in `agent.py` /
`schemas.py` / `tools.py`. `server.py` owns HTTP and nothing else. `web/` owns
presentation and is split into structure / style / behavior. The same
`run_triage()` powers both the CLI and the web API — the UI is a pure client of
the agent, not a reimplementation of it.

---

## Setup & run

```bash
# 1. configure
cp .env.example .env       # then edit with your endpoint details

# 2. install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. run the built-in demo (three contrasting cases)
python agent.py

# 4. or triage a single report
python agent.py "I have a fever of 39C and a stiff neck"
```

### Web frontend (optional)

A small, modern browser UI for interactive testing — symptom box, example
presets, a color-coded urgency result, and the full agent trace (whether vitals
were read, the vital-sign pills with out-of-range values flagged, and which
output tier produced the result).

```bash
uvicorn server:app --port 8000     # then open http://127.0.0.1:8000
```

It is a vanilla HTML/CSS/JS frontend (no build step, no node_modules) served by
FastAPI — deliberately lightweight to keep the focus on the agent. It supports
light/dark themes, is responsive, and falls back gracefully if the endpoint is
unreachable.

`.env` values:

```
MIMOE_BASE_URL=http://10.0.0.8:8083/mimik-ai/openai/v1
MIMOE_API_KEY=1234
MODEL_NAME=smollm-360m
```

---

## Demo recipes (normal vs. abnormal vitals)

`get_vital_signs` derives a patient's vitals **deterministically from the
patient id** (the id is hashed, then mapped into the reference ranges). You
don't edit numbers — you pick an id that hashes to the reading you want, and the
same id always returns the same vitals. Abnormal vitals trip the safety override,
which can raise urgency and force escalation, so the *same symptom* can triage
differently depending on the id.

**Patient ids with normal vitals (all five in range):**

| Patient id        | Vitals |
|-------------------|--------|
| `patient-calm-01` | HR 60, SpO₂ 95%, RR 18, SBP 101, Temp 37.0 |
| `healthy-01`      | HR 91, SpO₂ 100%, RR 17, SBP 100, Temp 36.8 |
| `healthy-03`      | HR 68, SpO₂ 98%, RR 12, SBP 119, Temp 37.0 |

**Patient ids with abnormal vitals (≥1 out of range → safety override fires):**

| Patient id         | Vitals | Out of range |
|--------------------|--------|--------------|
| `patient-dizzy-03` | HR 76, SpO₂ **93%**, RR 14, SBP 119, Temp 37.1 | low SpO₂ |
| `patient-acute-07` | HR 74, SpO₂ 96%, RR **11**, SBP **83**, Temp 36.5 | low RR + low BP |
| `demo-patient`     | HR **102**, SpO₂ 97%, RR 17, SBP **121**, Temp 36.5 | high HR + high BP |

You can confirm any id's vitals yourself:

```bash
python -c "from tools import get_vital_signs_raw, format_vitals; \
print(format_vitals(get_vital_signs_raw('patient-dizzy-03')))"
```

### Use cases to demonstrate each behavior

Run with `PATIENT_ID=<id> python agent.py "<symptom>"`, or set the **Patient ID**
field in the web UI.

The safety override works as a **floor**, never a ceiling: abnormal vitals force
urgency to **at least MEDIUM**, and red-flag symptoms to **at least HIGH**. The
model may independently rate a case *higher* than the floor — it just can't go
lower.

1. **Benign symptom + normal vitals → LOW, no escalation**
   `PATIENT_ID=patient-calm-01 python agent.py "Mild runny nose and slight sore throat since this morning."`

2. **Safety override floor — benign symptom + abnormal vitals → at least MEDIUM**
   `PATIENT_ID=patient-dizzy-03 python agent.py "I feel a bit dizzy when I stand up quickly but it passes."`
   (SpO₂ 93% is out of range, so a would-be LOW is raised to MEDIUM or higher.)

3. **Red-flag symptom → HIGH + escalate, regardless of vitals**
   `PATIENT_ID=patient-acute-07 python agent.py "Sudden crushing chest pain spreading to my left arm and short of breath."`

4. **Same symptom, vitals raise the floor** (run both, compare urgency)
   - normal:   `PATIENT_ID=healthy-03 python agent.py "I feel a bit dizzy when I stand up quickly."`
   - abnormal: `PATIENT_ID=patient-dizzy-03 python agent.py "I feel a bit dizzy when I stand up quickly."`
   The abnormal-vitals run is always ≥ the normal run — the clearest proof the
   `get_vital_signs` tool actually changes the decision.

5. **Fail-safe tier — model unreachable → escalates, never crashes, never calls the cloud**
   `MIMOE_BASE_URL=http://127.0.0.1:9/dead/v1 python agent.py "crushing chest pain"`

> Note: vitals are fully deterministic, but `smollm-360m`'s natural-language
> classification is not perfectly reproducible — exact urgency for borderline
> cases can vary between runs. The deterministic *floors* above always hold.

---

## Limitations & honesty

- This is **not** a medical device and gives **no** medical advice. It is a
  technical demonstration of an edge triage agent.
- On `smollm-360m`, urgency comes from Tier-2 classification of the model's
  prose plus deterministic safety rules — not from the model emitting a label
  it cannot reliably format. That is a feature of small-model engineering, and
  it is stated plainly rather than hidden behind a function-calling call that
  would silently degrade.
- `get_vital_signs` returns **simulated** data, deterministically derived from
  the patient id so demo runs are reproducible.
