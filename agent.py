"""mimOE on-device health triage agent (BYO Framework: LangChain).

Flow for one symptom report:

    1. DECIDE  - the model decides whether it needs vital signs before triaging.
    2. ACT     - if so, call the get_vital_signs tool (simulated sensor read).
    3. TRIAGE  - the model returns a structured TriageResult (Pydantic).

Everything runs against a local mimOE OpenAI-compatible endpoint. There is no
cloud fallback: if the local model is unreachable or its output cannot be
parsed, the agent fails *safe* by escalating to a human rather than guessing.

Design note: smollm-360m (the model mimOE is serving here) does not support
OpenAI function-calling or JSON mode, so `.bind_tools()` / `.with_structured_output()`
are not reliable against it. Instead we use LangChain's ChatOpenAI to talk to
the endpoint and a prompt-and-parse strategy (PydanticOutputParser + a tolerant
JSON extractor + a rule-based fail-safe). See README for the full rationale.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Optional, Tuple

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from schemas import TriageResult
from tools import (
    format_vitals,
    get_vital_signs_raw,
    vitals_are_abnormal,
)

load_dotenv()

BASE_URL = os.getenv("MIMOE_BASE_URL", "http://10.0.0.8:8083/mimik-ai/openai/v1")
API_KEY = os.getenv("MIMOE_API_KEY", "1234")
MODEL_NAME = os.getenv("MODEL_NAME", "smollm-360m")

# Low temperature: clinical outputs should be as consistent as possible.
TEMPERATURE = 0.1


def build_llm() -> ChatOpenAI:
    """Construct a ChatOpenAI client pointed at the local mimOE endpoint."""
    return ChatOpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        timeout=60,
        max_retries=0,  # on-device only; do not silently retry/fall back
        max_tokens=256,  # cap output: tiny models can loop/repeat indefinitely
    )


# --------------------------------------------------------------------------- #
# Step 1: tool-use decision (model decides whether to read vitals)
# --------------------------------------------------------------------------- #
_DECISION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a triage router. Decide if live vital signs (heart rate, "
            "SpO2, blood pressure, temperature) would help judge how urgent a "
            "symptom report is. Reply with EXACTLY one word: YES or NO.",
        ),
        ("user", "Symptom report: {symptom}\nNeed vital signs? Answer YES or NO:"),
    ]
)


def decide_use_vitals(llm: ChatOpenAI, symptom: str) -> bool:
    """Ask the model whether to call get_vital_signs. Fail safe = True.

    A tiny model is unreliable, so we read the first yes/no token and, if the
    answer is unclear, default to gathering vitals (more information is the
    safer choice for triage).
    """
    try:
        text = (_DECISION_PROMPT | llm).invoke({"symptom": symptom}).content.lower()
    except Exception:
        return True
    if re.search(r"\bno\b", text) and not re.search(r"\byes\b", text):
        return False
    return True


# --------------------------------------------------------------------------- #
# Step 3: structured triage
#
# Structured output is produced through a capability ladder, because a 360M
# model cannot reliably emit JSON or follow a fixed label format:
#
#   Tier 1  parse a JSON object if the model emitted one      (capable models)
#   Tier 2  classify the model's natural-language assessment  (tiny models)
#   Tier 3  rule-based fail-safe                               (model unusable)
#
# A clinical SAFETY OVERRIDE sits on top of Tiers 1-2: deterministic rules
# (red-flag symptoms, abnormal vital signs) may only RAISE urgency and force
# escalation, never lower them. The model proposes; safety rules override
# upward. That keeps the model in the loop while making dangerous misses hard.
# --------------------------------------------------------------------------- #
_parser = PydanticOutputParser(pydantic_object=TriageResult)

# Ask for free-form reasoning, which the model does do, rather than JSON.
_ASSESS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an on-device clinical triage assistant. In 2-3 short "
            "sentences, state how urgent the situation is (low, medium, or "
            "high) and what the patient should do right now. Be concise and "
            "conservative.",
        ),
        ("user", "Symptom report: {symptom}\nVital signs: {vitals}"),
    ]
)

# Red-flag symptoms force HIGH urgency + escalation regardless of model output.
_RED_FLAGS = (
    "chest pain", "not breathing", "can't breathe", "cannot breathe",
    "unconscious", "unresponsive", "overdose", "seizure", "stroke",
    "severe bleeding", "blue lips", "anaphylaxis", "choking",
)

_NEGATORS = ("not ", "no ", "n't", "without ", "isn't", "rule out", "rather than")
_HIGH_PHRASES = (
    "emergency", "911", "immediately", "right away", "life-threatening",
    "heart attack", "stroke", "urgent", "severe", "call for help",
    "as soon as possible", "seek immediate",
)
_LOW_PHRASES = (
    "mild", "not serious", "not severe", "common cold", "self-care",
    "monitor", "rest", "over-the-counter", "no cause for concern",
    "manageable", "not an emergency", "usually harmless",
)

_RANK = {"low": 0, "medium": 1, "high": 2}
_UNRANK = {0: "low", 1: "medium", 2: "high"}


def _negated(text: str, idx: int) -> bool:
    """True if a negator appears in the ~15 chars before position idx."""
    window = text[max(0, idx - 15) : idx]
    return any(neg in window for neg in _NEGATORS)


def _count_signal(text: str, phrases: Tuple[str, ...]) -> int:
    """Count non-negated occurrences of any phrase in text."""
    total = 0
    for phrase in phrases:
        start = 0
        while True:
            idx = text.find(phrase, start)
            if idx == -1:
                break
            if not _negated(text, idx):
                total += 1
            start = idx + len(phrase)
    return total


def _classify_assessment(text: str) -> str:
    """Map the model's free-text assessment to low/medium/high."""
    low = text.lower()
    high_hits = _count_signal(low, _HIGH_PHRASES)
    low_hits = _count_signal(low, _LOW_PHRASES)
    if high_hits > low_hits:
        return "high"
    if low_hits > high_hits:
        return "low"
    return "medium"


def _safety_floor(symptom: str, vitals: Optional[dict]) -> Tuple[str, list]:
    """Minimum urgency forced by deterministic clinical rules, plus reasons."""
    symptom_l = symptom.lower()
    reasons = []
    floor = "low"
    if any(flag in symptom_l for flag in _RED_FLAGS):
        floor = "high"
        reasons.append("red-flag symptom present")
    if vitals is not None and vitals_are_abnormal(vitals):
        floor = _UNRANK[max(_RANK[floor], _RANK["medium"])]
        reasons.append("abnormal vital signs")
    return floor, reasons


def _first_sentences(text: str, n: int = 2) -> str:
    """Trim a noisy response down to its first n sentences."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    cleaned = [p.strip() for p in parts if p.strip()]
    return " ".join(cleaned[:n]) if cleaned else text.strip()[:200]


def _action_for(urgency: str) -> str:
    return {
        "high": "Seek emergency care now — call emergency services.",
        "medium": "Contact a clinician today for assessment.",
        "low": "Self-care and monitor; seek care if symptoms worsen.",
    }[urgency]


def _extract_json(text: str) -> Optional[dict]:
    """Pull the first balanced {...} block out of a noisy model response."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _apply_safety_override(
    result: TriageResult, symptom: str, vitals: Optional[dict]
) -> TriageResult:
    """Raise urgency / force escalation per deterministic safety rules."""
    floor, reasons = _safety_floor(symptom, vitals)
    if _RANK[floor] > _RANK[result.urgency]:
        result.urgency = floor
        result.reasoning = (
            f"{result.reasoning} [safety override: {', '.join(reasons)}]"
        )
        result.recommended_action = _action_for(result.urgency)
    if result.urgency == "high":
        result.escalate_to_human = True
    return result


def _rule_based_fallback(symptom: str, vitals: Optional[dict]) -> TriageResult:
    """Safe deterministic triage used only when the model is unreachable.

    For a health system, never returning a result is not acceptable, and a
    confidently-wrong "low" is dangerous. So the fallback errs toward caution
    and always escalates to a human.
    """
    floor, reasons = _safety_floor(symptom, vitals)
    urgency = _UNRANK[max(_RANK[floor], _RANK["medium"])]
    detail = f" ({'; '.join(reasons)})" if reasons else ""
    return TriageResult(
        urgency=urgency,
        recommended_action="Have a clinician review this report now.",
        escalate_to_human=True,
        reasoning=(
            "Fail-safe triage: the local model was unreachable or returned "
            f"nothing, so the agent is escalating to a human{detail}."
        ),
    )


def triage_symptom(
    symptom: str,
    patient_id: str = "demo-patient",
    llm: Optional[ChatOpenAI] = None,
    verbose: bool = True,
) -> Tuple[TriageResult, str]:
    """Run the full triage loop for one symptom report.

    Returns (result, source) where source is one of:
    "json" (Tier 1), "nl-classified" (Tier 2), or "fail-safe" (Tier 3).
    """
    llm = llm or build_llm()

    # Step 1: decide whether to read vitals.
    use_vitals = decide_use_vitals(llm, symptom)
    vitals = None
    vitals_str = "not measured"
    if use_vitals:
        # Step 2: call the tool (simulated on-device sensor read).
        vitals = get_vital_signs_raw(patient_id)
        vitals_str = format_vitals(vitals)
    if verbose:
        print(f"  [decision] read vital signs? {'YES' if use_vitals else 'NO'}")
        if use_vitals:
            print(f"  [tool]     get_vital_signs -> {vitals_str}")

    # Step 3: ask the model, then structure its answer.
    try:
        raw = (_ASSESS_PROMPT | llm).invoke(
            {"symptom": symptom, "vitals": vitals_str}
        ).content
    except Exception as exc:
        if verbose:
            print(f"  [warn]     model call failed: {exc}")
        return _rule_based_fallback(symptom, vitals), "fail-safe"

    if not raw or not raw.strip():
        return _rule_based_fallback(symptom, vitals), "fail-safe"

    # Tier 1: the model emitted a JSON object we can validate directly.
    data = _extract_json(raw)
    if data is not None:
        try:
            # Validate through LangChain's PydanticOutputParser so the same
            # parser whose format instructions a capable model would follow is
            # the one that validates the result.
            result = _parser.parse(json.dumps(data))
            return _apply_safety_override(result, symptom, vitals), "json"
        except Exception:
            pass

    # Tier 2: classify the model's natural-language assessment.
    urgency = _classify_assessment(raw)
    result = TriageResult(
        urgency=urgency,
        recommended_action=_action_for(urgency),
        escalate_to_human=(urgency == "high"),
        reasoning=_first_sentences(raw),
    )
    return _apply_safety_override(result, symptom, vitals), "nl-classified"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
_DEMO_CASES = [
    ("I have a mild runny nose and a slight sore throat since this morning.",
     "patient-calm-01"),
    ("Sudden crushing chest pain spreading to my left arm and I feel short of "
     "breath.", "patient-acute-07"),
    ("I feel a bit dizzy when I stand up quickly but it passes.",
     "patient-dizzy-03"),
]


def _run_demo() -> None:
    print(f"mimOE health triage agent")
    print(f"endpoint : {BASE_URL}")
    print(f"model    : {MODEL_NAME}  (on-device, no cloud fallback)\n")
    llm = build_llm()
    for i, (symptom, pid) in enumerate(_DEMO_CASES, 1):
        print(f"[case {i}] patient={pid}")
        print(f"  symptom: {symptom}")
        result, source = triage_symptom(symptom, pid, llm=llm)
        print(f"  --- TriageResult (source: {source}) ---")
        print(result.pretty())
        print()


def main() -> None:
    if len(sys.argv) > 1:
        symptom = " ".join(sys.argv[1:])
        pid = os.getenv("PATIENT_ID", "demo-patient")
        print(f"symptom: {symptom}\n")
        result, source = triage_symptom(symptom, pid)
        print(f"\n--- TriageResult (source: {source}) ---")
        print(result.pretty())
        # Also emit machine-readable JSON for downstream consumers.
        print("\nJSON:")
        print(result.model_dump_json(indent=2))
    else:
        _run_demo()


if __name__ == "__main__":
    main()
