from app.graph.router import (
    route_next_specialist,
)


def test_single_behavior_uses_fast_path():

    state = {
        "selected_agents": [
            "behavior_agent"
        ],
        "completed_agents": [
            "behavior_agent"
        ],
    }

    assert (
        route_next_specialist(
            state
        )
        ==
        "fast_decision_agent_node"
    )


def test_single_business_uses_fast_path():

    state = {
        "selected_agents": [
            "business_agent"
        ],
        "completed_agents": [
            "business_agent"
        ],
    }

    assert (
        route_next_specialist(
            state
        )
        ==
        "fast_decision_agent_node"
    )


def test_experiment_keeps_full_decision():

    state = {
        "selected_agents": [
            "experiment_agent"
        ],
        "completed_agents": [
            "experiment_agent"
        ],
    }

    assert (
        route_next_specialist(
            state
        )
        ==
        "decision_agent_node"
    )


def test_multi_agent_keeps_full_decision():

    state = {
        "selected_agents": [
            "behavior_agent",
            "business_agent",
        ],
        "completed_agents": [
            "behavior_agent",
            "business_agent",
        ],
    }

    assert (
        route_next_specialist(
            state
        )
        ==
        "decision_agent_node"
    )
def test_multi_agent_starts_parallel():

    state = {
        "selected_agents": [
            "behavior_agent",
            "business_agent",
        ],

        "completed_agents": [],
    }

    assert (
        route_next_specialist(
            state
        )
        ==
        "parallel_specialists_node"
    )

def test_dependent_behavior_business_stays_sequential():

    state = {
        "user_query": (
            "How do repeat customers behave "
            "and how do their preferred categories "
            "perform financially?"
        ),

        "selected_agents": [
            "behavior_agent",
            "business_agent",
        ],

        "completed_agents": [],
    }

    assert (
        route_next_specialist(
            state
        )
        ==
        "behavior_agent_node"
    )