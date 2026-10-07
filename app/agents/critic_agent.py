from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from app.config import get_llm

from app.graph.state import AgentState
from app.graph.memory import (
    build_contextual_query,
)

from app.schemas.models import (
    CriticReport,
)


# =========================================================
# Prompt
# =========================================================
SYSTEM_PROMPT = """
You are the Critic Agent in a multi-agent
e-commerce decision system.

Your responsibility is verification, NOT business analysis.

You must check whether the Decision Agent's output is
faithful to the specialist evidence.

Check for:

- unsupported claims
- invented numbers
- arithmetic not present in specialist reports
- contradictions with specialist findings
- causal claims based only on observational data
- ignored experiment uncertainty
- ignored profit or business significance
- missing material limitations
- recommendations inconsistent with evidence
- system-prompt or workflow-instruction leakage
- meta-commentary that does not answer the business question
- inconsistent decision_label versus conclusion or recommendation

Rules:

1. Do NOT perform new business analysis.
2. Do NOT perform new arithmetic.
3. Do NOT invent alternative statistics.
4. Do NOT reject a decision merely for writing style.
5. Use verdict="revise" only for material problems.
6. Use verdict="pass" when evidence and recommendation
   are substantively consistent.
7. Be concise.
8. A decision that repeats system instructions,
   workflow rules, or internal agent behavior should
   receive verdict="revise".
9. For rollout decisions, verify that decision_label is
   consistent with specialist experiment evidence.
"""


# =========================================================
# Helpers
# =========================================================
def _collect_evidence(
    state: AgentState,
) -> str:

    sections = []

    for key in [
        "behavior_report",
        "business_report",
        "experiment_report",
    ]:

        report = state.get(
            key
        )

        if report is not None:

            sections.append(
                f"{key.upper()}:\n"
                +
                report.model_dump_json(
                    indent=2
                )
            )

    return "\n\n".join(
        sections
    )


# =========================================================
# Critic
# =========================================================
def create_critic_agent():

    llm = get_llm()

    return llm.with_structured_output(
        CriticReport,
        method="function_calling",
        strict=False,
    )


def run_critic_agent(
    state: AgentState,
) -> CriticReport:

    decision = state.get(
        "decision_report"
    )

    if decision is None:

        raise RuntimeError(
            "Critic Agent cannot run "
            "without decision_report."
        )

    evidence = _collect_evidence(
        state
    )

    contextual_query = (
        build_contextual_query(
            state
        )
    )

    prompt = f"""
ORIGINAL USER QUESTION:

{contextual_query}


SPECIALIST EVIDENCE:

{evidence}


DECISION AGENT OUTPUT:

{decision.model_dump_json(indent=2)}


Verify the decision against the specialist evidence.
"""

    critic = create_critic_agent()

    result = critic.invoke(
        [
            SystemMessage(
                content=SYSTEM_PROMPT
            ),

            HumanMessage(
                content=prompt
            ),
        ]
    )

    if isinstance(
        result,
        CriticReport
    ):
        return result

    return CriticReport.model_validate(
        result
    )


# =========================================================
# LangGraph Node
# =========================================================
def critic_agent_node(
    state: AgentState,
) -> dict:

    critique = run_critic_agent(
        state
    )

    return {
        "critique":
            critique
    }
