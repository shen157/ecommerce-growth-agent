from typing import Literal

from app.graph.state import AgentState


RouterDestination = Literal[
    "behavior_agent_node",
    "business_agent_node",
    "experiment_agent_node",
    "fast_decision_agent_node",
    "parallel_specialists_node",
    "decision_agent_node",
]


AGENT_TO_NODE = {
    "behavior_agent":
        "behavior_agent_node",

    "business_agent":
        "business_agent_node",

    "experiment_agent":
        "experiment_agent_node",
}


def route_next_specialist(
    state: AgentState,
) -> RouterDestination:
    """
    Route specialists using dependency-aware execution.

    Independent multi-agent tasks run in parallel.

    Tasks where Business depends on findings first
    discovered by Behavior remain sequential.
    """

    selected_agents = state.get(
        "selected_agents",
        [],
    )

    completed_agents = set(
        state.get(
            "completed_agents",
            [],
        )
    )

    if not selected_agents:
        return "decision_agent_node"

    # =========================================================
    # Dependency-aware parallel routing
    # =========================================================
    if (
        len(selected_agents) > 1
        and not completed_agents
    ):

        query = (
            state.get(
                "user_query",
                "",
            )
            .lower()
        )

        selected_set = set(
            selected_agents
        )

        behavior_business_pair = {
            "behavior_agent",
            "business_agent",
        }.issubset(
            selected_set
        )

        dependency_phrases = (
            "preferred categories",
            "their preferred categories",
            "those categories",
            "category preferences",
            "categories do repeat customers",
            "repeat customers prefer",
        )

        requires_behavior_first = (
            behavior_business_pair
            and any(
                phrase in query
                for phrase
                in dependency_phrases
            )
        )

        # Independent specialists can run concurrently.
        if not requires_behavior_first:
            return "parallel_specialists_node"

        # Otherwise fall through to the normal sequential
        # specialist execution below.

    # =========================================================
    # Sequential specialist execution
    # =========================================================
    for agent_name in selected_agents:

        if (
            agent_name
            not in completed_agents
        ):

            return AGENT_TO_NODE[
                agent_name
            ]

    # =========================================================
    # Single-agent fast path
    # =========================================================
    if (
        len(selected_agents) == 1
        and selected_agents[0]
        in {
            "behavior_agent",
            "business_agent",
        }
    ):
        return "fast_decision_agent_node"

    # Experiments and multi-agent synthesis still use
    # the full Decision Agent.
    return "decision_agent_node"


def validate_routing_state(
    state: AgentState,
) -> None:
    """
    Validate Coordinator routing output.
    """

    selected_agents = state.get(
        "selected_agents",
        []
    )

    valid_agents = set(
        AGENT_TO_NODE.keys()
    )

    invalid_agents = [
        agent
        for agent in selected_agents
        if agent not in valid_agents
    ]

    if invalid_agents:

        raise ValueError(
            "Unknown specialist agents: "
            f"{invalid_agents}"
        )

CriticDestination = Literal[
    "revise",
    "finish",
]


MAX_REVISIONS = 1


def route_after_critic(
    state: AgentState,
) -> CriticDestination:
    """
    Decide whether Decision Agent must revise.

    Maximum one revision is allowed to avoid
    uncontrolled reflection loops.
    """

    critique = state.get(
        "critique"
    )

    revision_count = state.get(
        "revision_count",
        0
    )

    if critique is None:
        return "finish"

    if (
        critique.verdict
        ==
        "revise"
        and
        revision_count
        <
        MAX_REVISIONS
    ):
        return "revise"

    return "finish"