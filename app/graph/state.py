import operator
from typing import (
    Annotated,
    NotRequired,
    TypedDict,
)

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from app.schemas.models import (
    CoordinatorDecision,
    CriticReport,
    DecisionReport,
    SpecialistName,
    SpecialistReport,
)


class AgentState(TypedDict):
    """
    Shared working memory for one multi-agent task.
    """

    # =========================================================
    # Input
    # =========================================================
    user_query: str

    thread_id: NotRequired[str]

    messages: NotRequired[
        Annotated[
            list[AnyMessage],
            add_messages,
        ]
    ]

    # =========================================================
    # Coordinator
    # =========================================================
    coordinator_decision: NotRequired[
        CoordinatorDecision
    ]

    selected_agents: NotRequired[
        list[SpecialistName]
    ]

    completed_agents: NotRequired[
        list[SpecialistName]
    ]

    # =========================================================
    # Specialist Reports
    # =========================================================
    behavior_report: NotRequired[
        SpecialistReport | None
    ]

    business_report: NotRequired[
        SpecialistReport | None
    ]

    experiment_report: NotRequired[
        SpecialistReport | None
    ]

    # =========================================================
    # Decision + Critic
    # =========================================================
    decision_report: NotRequired[
        DecisionReport | None
    ]

    critique: NotRequired[
        CriticReport | None
    ]

    revision_count: NotRequired[int]

    # =========================================================
    # Errors
    # =========================================================
    errors: NotRequired[
        list[str]
    ]

    # =========================================================
    # Performance Profiling
    # =========================================================
    run_id: str

    latency_trace: Annotated[
        list[dict],
        operator.add,
    ]