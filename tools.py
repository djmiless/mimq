"""Tools available to the triage agent.

For this assignment `get_vital_signs` returns *simulated* sensor data, standing
in for the on-device sensors a real edge deployment (e.g. a wearable feeding
Miles CAM) would expose. It is registered both as a plain callable and as a
LangChain `@tool` so it can be wired into either a manual loop or a LangChain
agent executor.

Why a manual loop instead of `.bind_tools()`: the target model, smollm-360m
served by mimOE, does not implement OpenAI function-calling. It ignores the
`tools` parameter and never emits `tool_calls`. So the agent decides whether to
call this tool via a small prompted decision step (see agent.py) and then
invokes the plain function directly.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict

from langchain_core.tools import tool

# Plausible adult reference ranges, used only to make the simulation realistic
# and to drive the rule-based fail-safe in agent.py.
_NORMAL = {
    "heart_rate_bpm": (60, 100),
    "spo2_percent": (95, 100),
    "respiratory_rate_bpm": (12, 20),
    "systolic_bp_mmhg": (90, 120),
    "temperature_c": (36.1, 37.2),
}


def _seeded_vitals(patient_id: str) -> Dict[str, Any]:
    """Deterministically derive a vitals snapshot from the patient id.

    Deterministic (not random) so a demo run is reproducible and so the same
    patient id always yields the same reading. Some ids land outside the normal
    range on purpose, to exercise the high-urgency / escalation path.
    """
    digest = hashlib.sha256(patient_id.encode("utf-8")).digest()

    def pick(lo: float, hi: float, byte: int, spread: float = 0.25) -> float:
        # Map a byte (0-255) into [lo - spread*range, hi + spread*range] so
        # roughly the tails fall outside normal.
        rng = hi - lo
        frac = digest[byte] / 255.0
        value = (lo - spread * rng) + frac * (rng * (1 + 2 * spread))
        return round(value, 1)

    return {
        "patient_id": patient_id,
        "heart_rate_bpm": int(pick(60, 100, 0)),
        "spo2_percent": int(min(100, pick(95, 100, 1))),
        "respiratory_rate_bpm": int(pick(12, 20, 2)),
        "systolic_bp_mmhg": int(pick(90, 120, 3)),
        "temperature_c": pick(36.1, 37.2, 4),
    }


def get_vital_signs_raw(patient_id: str = "demo-patient") -> Dict[str, Any]:
    """Return a simulated vital-signs snapshot for a patient (plain function)."""
    return _seeded_vitals(patient_id)


@tool
def get_vital_signs(patient_id: str = "demo-patient") -> Dict[str, Any]:
    """Read the patient's current vital signs from on-device sensors.

    Returns heart rate (bpm), SpO2 (%), respiratory rate (bpm), systolic blood
    pressure (mmHg) and temperature (C). Use before triaging when the symptom
    report alone is not enough to judge urgency.
    """
    return _seeded_vitals(patient_id)


def format_vitals(vitals: Dict[str, Any]) -> str:
    """Compact one-line rendering for injection into the model prompt."""
    return (
        f"HR={vitals['heart_rate_bpm']}bpm, "
        f"SpO2={vitals['spo2_percent']}%, "
        f"RR={vitals['respiratory_rate_bpm']}bpm, "
        f"SBP={vitals['systolic_bp_mmhg']}mmHg, "
        f"Temp={vitals['temperature_c']}C"
    )


def vitals_are_abnormal(vitals: Dict[str, Any]) -> bool:
    """True if any vital is outside its normal reference range."""
    checks = {
        "heart_rate_bpm": vitals["heart_rate_bpm"],
        "spo2_percent": vitals["spo2_percent"],
        "respiratory_rate_bpm": vitals["respiratory_rate_bpm"],
        "systolic_bp_mmhg": vitals["systolic_bp_mmhg"],
        "temperature_c": vitals["temperature_c"],
    }
    for key, value in checks.items():
        lo, hi = _NORMAL[key]
        if value < lo or value > hi:
            return True
    return False
