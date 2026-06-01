"""Pydantic output schema for the mimOE health triage agent.

The agent's whole contract with the rest of the system is this schema: a typed,
validated object instead of free-text. Downstream code (an alerting service, a
nurse dashboard, the Miles CAM overdose-detection pipeline) can rely on these
fields existing and being well-formed.
"""

from typing import Literal

from pydantic import BaseModel, Field

UrgencyLevel = Literal["low", "medium", "high"]


class TriageResult(BaseModel):
    """Structured result of triaging a single symptom report."""

    urgency: UrgencyLevel = Field(
        description="Clinical urgency: 'low', 'medium', or 'high'."
    )
    recommended_action: str = Field(
        description="Short, concrete next step for the patient or caregiver."
    )
    escalate_to_human: bool = Field(
        description="True if a human clinician must review this case now."
    )
    reasoning: str = Field(
        description="Brief justification for the urgency and action."
    )

    def pretty(self) -> str:
        """Human-readable rendering for the CLI."""
        flag = "ESCALATE TO HUMAN" if self.escalate_to_human else "no escalation"
        return (
            f"  urgency            : {self.urgency.upper()}\n"
            f"  recommended_action : {self.recommended_action}\n"
            f"  escalate_to_human  : {self.escalate_to_human}  ({flag})\n"
            f"  reasoning          : {self.reasoning}"
        )
