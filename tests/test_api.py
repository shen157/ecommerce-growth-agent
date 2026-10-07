from fastapi.testclient import TestClient

from app import main as main_module
from app.schemas.models import (
    CoordinatorDecision,
    CriticReport,
    DecisionReport,
)


client = TestClient(main_module.app)


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_returns_structured_workflow_result(monkeypatch):
    captured = {}
    captured_log = {}

    def fake_run_workflow(query, thread_id=None):
        captured["query"] = query
        captured["thread_id"] = thread_id
        return {
            "thread_id": thread_id,
            "run_id": "run-current",
            "coordinator_decision": CoordinatorDecision(
                intent_summary="Analyze customer behavior",
                selected_agents=["behavior_agent"],
                routing_rationale="Behavior analysis is required.",
                requires_synthesis=False,
            ),
            "completed_agents": ["behavior_agent"],
            "decision_report": DecisionReport(
                conclusion="Repeat customers spend more overall.",
                confidence="high",
            ),
            "critique": CriticReport(
                verdict="pass",
                evidence_consistent=True,
                summary="The conclusion is supported.",
            ),
            "errors": [],
            "latency_trace": [
                {
                    "run_id": "run-previous",
                    "node": "coordinator",
                    "seconds": 9.0,
                },
                {
                    "run_id": "run-current",
                    "node": "behavior_agent",
                    "seconds": 1.25,
                },
            ],
            "workflow_latency_seconds": 1.5,
        }

    monkeypatch.setattr(
        main_module,
        "run_workflow",
        fake_run_workflow,
    )

    def fake_log_workflow_summary(event, query, result, **fields):
        captured_log.update(
            {
                "event": event,
                "query": query,
                "result": result,
                **fields,
            }
        )

    monkeypatch.setattr(
        main_module,
        "log_workflow_summary",
        fake_log_workflow_summary,
    )

    response = client.post(
        "/analyze",
        json={
            "query": "  Compare repeat customers  ",
            "thread_id": "customer-123",
        },
    )

    assert response.status_code == 200
    assert captured == {
        "query": "Compare repeat customers",
        "thread_id": "customer-123",
    }

    body = response.json()
    assert body["thread_id"] == "customer-123"
    assert body["run_id"] == "run-current"
    assert body["selected_agents"] == ["behavior_agent"]
    assert body["completed_agents"] == ["behavior_agent"]
    assert body["decision"]["confidence"] == "high"
    assert body["critic_verdict"] == "pass"
    assert body["errors"] == []
    assert body["latency_trace"] == [
        {
            "run_id": "run-current",
            "node": "behavior_agent",
            "seconds": 1.25,
        }
    ]
    assert body["workflow_latency_seconds"] == 1.5
    assert captured_log["event"] == "api.analyze.completed"
    assert captured_log["query"] == "Compare repeat customers"
    assert captured_log["result"]["run_id"] == "run-current"
    assert captured_log["http_status"] == 200


def test_analyze_rejects_blank_query(monkeypatch):
    def unexpected_run_workflow(*args, **kwargs):
        raise AssertionError("run_workflow should not be called")

    monkeypatch.setattr(
        main_module,
        "run_workflow",
        unexpected_run_workflow,
    )

    response = client.post(
        "/analyze",
        json={"query": "   "},
    )

    assert response.status_code == 422


def test_analyze_returns_safe_500_when_workflow_fails(monkeypatch):
    captured_log = {}

    def failing_run_workflow(query, thread_id=None):
        raise RuntimeError("provider secret should not leak")

    def fake_log_error_event(event, **fields):
        captured_log["event"] = event
        captured_log.update(fields)

    monkeypatch.setattr(
        main_module,
        "run_workflow",
        failing_run_workflow,
    )
    monkeypatch.setattr(
        main_module,
        "log_error_event",
        fake_log_error_event,
    )

    response = client.post(
        "/analyze",
        json={"query": "Analyze revenue"},
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Workflow analysis failed"
    }
    assert captured_log["event"] == "api.analyze.failed"
    assert captured_log["query"] == "Analyze revenue"
    assert captured_log["thread_id"] is None
    assert isinstance(captured_log["error"], RuntimeError)
    assert captured_log["http_status"] == 500
