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
    CoordinatorDecision,
)


# =========================================================
# System Prompt
# =========================================================
SYSTEM_PROMPT = """
You are the Coordinator Agent in a multi-agent
e-commerce growth decision system.

Your responsibility is ONLY:

1. Understand the user's analytical request.
2. Determine which specialist agents are required.
3. Route the request to the minimum sufficient set
   of specialist agents.

You do NOT perform the business analysis yourself.

When multiple specialists are selected, order selected_agents
in a logical execution order so that later specialists can
use earlier specialist outputs as context when useful.

Available specialists:

-----------------------------------------
behavior_agent
-----------------------------------------

Use for:

- repeat customers
- customer loyalty
- customer segments
- category preferences
- delivery experience
- reviews
- behavioral differences

Examples:

"How are repeat customers different?"
"Does late delivery relate to poor reviews?"
"What categories do repeat customers prefer?"


-----------------------------------------
business_agent
-----------------------------------------

Use for:

- revenue
- order volume
- customers
- AOV
- revenue trends
- top / bottom categories
- month-over-month category performance
- declining categories

Examples:

"Which categories lost the most revenue?"
"What was monthly revenue?"
"How did AOV change?"


-----------------------------------------
experiment_agent
-----------------------------------------

Use for:

- A/B tests
- EXP001 / EXP002 / EXP003
- control vs treatment
- conversion lift
- p-values
- statistical significance
- revenue per user
- profit per user
- experiment rollout decisions

Examples:

"Should EXP001 be rolled out?"
"Is EXP002 statistically significant?"


-----------------------------------------
Multi-specialist queries
-----------------------------------------

Some questions require multiple specialists.

Example:

"Repeat customers prefer which categories,
and are those categories growing in revenue?"

Requires:

behavior_agent
business_agent


Example:

"EXP001 improves conversion. Does this align
with the overall business trend?"

Requires:

experiment_agent
business_agent


Rules:

1. Select only agents that are actually needed.
2. One query may require 1, 2, or 3 specialists.
3. Do not select all agents by default.
4. Do not perform numerical analysis yourself.
5. Do not invent business conclusions.
6. routing_rationale must be short and operational.
7. requires_synthesis should be true whenever
   multiple specialists are selected.
"""


# =========================================================
# Coordinator
# =========================================================
def create_coordinator():
    """
    Create structured-output coordinator.
    """

    llm = get_llm()

    return llm.with_structured_output(
        CoordinatorDecision,
        method="function_calling",
        strict=False,
    )


def run_coordinator(
    query: str,
) -> CoordinatorDecision:
    """
    Classify the user query and select
    the required specialist agents.
    """

    coordinator = create_coordinator()

    result = coordinator.invoke(
        [
            SystemMessage(
                content=SYSTEM_PROMPT
            ),
            HumanMessage(
                content=query
            ),
        ]
    )

    if isinstance(
        result,
        CoordinatorDecision
    ):
        decision = result

    else:
        decision = (
            CoordinatorDecision
            .model_validate(
                result
            )
        )

    # Ensure synthesis flag is consistent.
    decision = decision.model_copy(
        update={
            "requires_synthesis":
                len(
                    decision.selected_agents
                )
                > 1
        }
    )

    return decision


# =========================================================
# LangGraph Node
# =========================================================
def coordinator_node(
    state: AgentState,
) -> dict:
    """
    LangGraph node.

    Reads:
        user_query

    Writes:
        coordinator_decision
        selected_agents
        completed_agents
    """

    query = build_contextual_query(
        state
    )

    decision = run_coordinator(
        query
    )

    return {
        "coordinator_decision":
            decision,

        "selected_agents":
            decision.selected_agents,

        "completed_agents":
            [],

        "errors":
            [],

        "behavior_report":
            None,

        "business_report":
            None,

        "experiment_report":
            None,

        "decision_report":
            None,

        "critique":
            None,

        "revision_count":
            0,
    }


# =========================================================
# Manual Test
# =========================================================
if __name__ == "__main__":

    test_queries = [
        (
            "How are repeat customers "
            "different from one-time customers?"
        ),

        (
            "Which categories experienced "
            "the largest revenue decline "
            "in June 2018?"
        ),

        (
            "Should we roll out EXP001?"
        ),

        (
            "Which categories do repeat "
            "customers prefer, and are those "
            "categories performing well "
            "in revenue?"
        ),
    ]

    for query in test_queries:

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"QUERY: {query}"
        )

        print(
            "=" * 70
        )

        decision = run_coordinator(
            query
        )

        print(
            decision.model_dump_json(
                indent=2
            )
        )
