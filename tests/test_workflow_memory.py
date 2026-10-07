from langgraph.checkpoint.memory import InMemorySaver

import app.graph.workflow as workflow_module
import app.agents.coordinator as coordinator_module
from app.schemas.models import (
    CoordinatorDecision,
    CriticReport,
    DecisionReport,
    SpecialistReport,
)


def test_same_thread_retains_multi_turn_context(
    monkeypatch,
):
    coordinator_queries = []

    def fake_run_coordinator(query):
        coordinator_queries.append(
            query
        )

        return CoordinatorDecision(
            intent_summary="Test request",
            selected_agents=[
                "behavior_agent"
            ],
            routing_rationale=(
                "Behavior context is required."
            ),
            requires_synthesis=False,
        )

    def fake_behavior(state):
        return {
            "behavior_report":
                SpecialistReport(
                    agent_name=(
                        "behavior_agent"
                    ),
                    conclusion=(
                        "Behavior result"
                    ),
                ),
            "completed_agents": [
                "behavior_agent"
            ],
        }

    def fake_decision(state):
        return {
            "decision_report":
                DecisionReport(
                    conclusion=(
                        "Answer for: "
                        + state[
                            "user_query"
                        ]
                    ),
                    confidence="high",
                ),
            "revision_count":
                state.get(
                    "revision_count",
                    0,
                ),
        }

    def fake_critic(state):
        return {
            "critique":
                CriticReport(
                    verdict="pass",
                    evidence_consistent=True,
                    summary="Evidence is consistent.",
                )
        }

    monkeypatch.setattr(
        coordinator_module,
        "run_coordinator",
        fake_run_coordinator,
    )
    monkeypatch.setattr(
        workflow_module,
        "behavior_agent_node",
        fake_behavior,
    )
    monkeypatch.setattr(
        workflow_module,
        "decision_agent_node",
        fake_decision,
    )
    monkeypatch.setattr(
        workflow_module,
        "fast_single_specialist_decision_node",
        fake_decision,
    )
    monkeypatch.setattr(
        workflow_module,
        "critic_agent_node",
        fake_critic,
    )

    app = workflow_module.build_workflow(
        checkpointer=InMemorySaver()
    )

    first = workflow_module.run_workflow(
        "Which categories are strongest?",
        thread_id="customer-analysis",
        app=app,
    )
    second = workflow_module.run_workflow(
        "How about repeat customers?",
        thread_id="customer-analysis",
        app=app,
    )

    assert first["thread_id"] == (
        "customer-analysis"
    )
    assert len(first["messages"]) == 2
    assert len(second["messages"]) == 4

    second_turn_context = (
        coordinator_queries[1]
    )

    assert (
        "Which categories are strongest?"
        in second_turn_context
    )
    assert (
        "Answer for: Which categories are strongest?"
        in second_turn_context
    )
    assert (
        "How about repeat customers?"
        in second_turn_context
    )


def test_different_thread_does_not_share_context(
    monkeypatch,
):
    coordinator_queries = []

    def fake_run_coordinator(query):
        coordinator_queries.append(
            query
        )

        return CoordinatorDecision(
            intent_summary="Test request",
            selected_agents=[
                "behavior_agent"
            ],
            routing_rationale="Behavior only.",
            requires_synthesis=False,
        )

    def fake_behavior(state):
        return {
            "completed_agents": [
                "behavior_agent"
            ]
        }

    def fake_decision(state):
        return {
            "decision_report":
                DecisionReport(
                    conclusion="Thread-local answer",
                    confidence="high",
                )
        }

    def fake_critic(state):
        return {
            "critique":
                CriticReport(
                    verdict="pass",
                    evidence_consistent=True,
                    summary="Pass.",
                )
        }

    monkeypatch.setattr(
        coordinator_module,
        "run_coordinator",
        fake_run_coordinator,
    )
    monkeypatch.setattr(
        workflow_module,
        "behavior_agent_node",
        fake_behavior,
    )
    monkeypatch.setattr(
        workflow_module,
        "decision_agent_node",
        fake_decision,
    )
    monkeypatch.setattr(
        workflow_module,
        "fast_single_specialist_decision_node",
        fake_decision,
    )
    monkeypatch.setattr(
        workflow_module,
        "critic_agent_node",
        fake_critic,
    )

    app = workflow_module.build_workflow(
        checkpointer=InMemorySaver()
    )

    workflow_module.run_workflow(
        "First thread question",
        thread_id="thread-a",
        app=app,
    )
    result = workflow_module.run_workflow(
        "Second thread question",
        thread_id="thread-b",
        app=app,
    )

    assert len(result["messages"]) == 2
    assert (
        "First thread question"
        not in coordinator_queries[1]
    )
