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
├── agent.py         # main agent: connect to mimOE, run the DECIDE → ACT → TRIAGE loop
├── schemas.py       # Pydantic TriageResult output schema
├── tools.py         # get_vital_signs tool (simulated sensor data) + safety helpers
├── .env             # local config (gitignored)
├── .env.example     # config template
├── requirements.txt # langchain, langchain-openai, pydantic, python-dotenv
├── .gitignore       # excludes .env
└── README.md
```

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

`.env` values:

```
MIMOE_BASE_URL=http://10.0.0.8:8083/mimik-ai/openai/v1
MIMOE_API_KEY=1234
MODEL_NAME=smollm-360m
```

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
