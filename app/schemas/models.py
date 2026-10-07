from typing import Literal

from pydantic import BaseModel, Field


class SpecialistReport(BaseModel):
    """
    Standard structured output produced
    by specialist agents.
    """

    agent_name: Literal[
        "behavior_agent",
        "business_agent",
        "experiment_agent",
    ]

    conclusion: str = Field(
        description=(
            "Direct analytical conclusion."
        )
    )

    key_findings: list[str] = Field(
        default_factory=list,
        description=(
            "Important findings derived "
            "from analytical tools."
        ),
    )

    evidence: list[str] = Field(
        default_factory=list,
        description=(
            "Numerical evidence supporting "
            "the conclusion."
        ),
    )

    recommendation: str | None = Field(
        default=None,
        description=(
            "Business recommendation "
            "when appropriate."
        ),
    )

    limitations: list[str] = Field(
        default_factory=list,
        description=(
            "Important caveats or limitations."
        ),
    )

SpecialistName = Literal[
    "behavior_agent",
    "business_agent",
    "experiment_agent",
]


class CoordinatorDecision(BaseModel):
    """
    Routing decision produced by the Coordinator Agent.
    """

    intent_summary: str = Field(
        description=(
            "Short summary of what the user "
            "is asking the system to analyze."
        )
    )

    selected_agents: list[SpecialistName] = Field(
        min_length=1,
        max_length=3,
        description=(
            "Specialist agents required to "
            "answer the user query."
        ),
    )

    routing_rationale: str = Field(
        description=(
            "Brief explanation of why these "
            "specialists are required. "
            "Do not provide hidden reasoning "
            "or step-by-step chain of thought."
        )
    )

    requires_synthesis: bool = Field(
        description=(
            "True when results from multiple "
            "specialists must be combined."
        )
    )

class DecisionReport(BaseModel):
    """
    Final synthesis produced by the Decision Agent.
    """

    conclusion: str = Field(
        description=(
            "Direct answer to the user's original question."
        )
    )

    key_evidence: list[str] = Field(
        default_factory=list,
        description=(
            "Most important evidence taken from "
            "specialist reports."
        ),
    )

    recommendation: str | None = Field(
        default=None,
        description=(
            "Final business recommendation when appropriate."
        ),
    )

    confidence: Literal[
        "high",
        "medium",
        "low",
    ] = Field(
        description=(
            "Confidence based on consistency and "
            "completeness of available evidence."
        )
    )

    specialist_contributions: list[str] = Field(
        default_factory=list,
        description=(
            "Short description of how each specialist "
            "contributed to the decision."
        ),
    )

    limitations: list[str] = Field(
        default_factory=list,
        description=(
            "Important limitations, uncertainty, "
            "or missing evidence."
        ),
    )

    decision_label: Literal[
        "rollout",
        "do_not_rollout",
        "monitor",
        "not_applicable",
    ] = Field(
        default="not_applicable",
        description=(
            "Machine-readable final action label. "
            "Use rollout or do_not_rollout only when the "
            "user asks for an experiment rollout decision. "
            "Use monitor for a wait-and-observe decision. "
            "Use not_applicable for descriptive or analytical questions."
        ),
    )

class CriticReport(BaseModel):
    """
    Verification result produced by the Critic Agent.
    """

    verdict: Literal[
        "pass",
        "revise",
    ]

    evidence_consistent: bool = Field(
        description=(
            "Whether the final decision is consistent "
            "with specialist evidence."
        )
    )

    unsupported_claims: list[str] = Field(
        default_factory=list,
        description=(
            "Claims not supported by specialist reports."
        ),
    )

    missing_considerations: list[str] = Field(
        default_factory=list,
        description=(
            "Material evidence or limitations that "
            "the Decision Agent failed to consider."
        ),
    )

    revision_instructions: list[str] = Field(
        default_factory=list,
        description=(
            "Specific instructions for correcting "
            "the decision when verdict is revise."
        ),
    )

    summary: str = Field(
        description=(
            "Short verification summary."
        )
    )