import json

from app.observability.logging import (
    build_workflow_summary,
    log_error_event,
    log_workflow_summary,
)
from app.schemas.models import (
    CoordinatorDecision,
    CriticReport,
)


def _workflow_result(secret=""):
    return {
        "thread_id": "thread-123",
        "run_id": "run-123",
        "coordinator_decision": CoordinatorDecision(
            intent_summary="Analyze customers",
            selected_agents=["behavior_agent", "business_agent"],
            routing_rationale="Two perspectives are required.",
            requires_synthesis=True,
        ),
        "completed_agents": ["behavior_agent", "business_agent"],
        "critique": CriticReport(
            verdict="pass",
            evidence_consistent=True,
            summary="Supported by the data.",
        ),
        "revision_count": 1,
        "errors": [f"safe warning {secret}"],
        "latency_trace": [
            {"run_id": "old-run", "node": "coordinator", "seconds": 9},
            {"run_id": "run-123", "node": "coordinator", "seconds": 0.1},
            {"run_id": "run-123", "node": "behavior_agent", "seconds": 0.2},
            {"run_id": "run-123", "node": "behavior_agent", "seconds": 0.3},
            {"run_id": "run-123", "node": "parallel_specialists", "seconds": 0.4},
            {"run_id": "run-123", "node": "decision_agent", "seconds": 0.5},
            {"run_id": "run-123", "node": "critic_agent", "seconds": 0.6},
        ],
        "workflow_latency_seconds": 2.25,
    }


def test_workflow_summary_has_required_fields_and_aggregated_latency(capsys):
    query = "Compare repeat and one-time customers"
    summary = build_workflow_summary(query, _workflow_result())

    required_fields = {
        "run_id",
        "thread_id",
        "query_summary",
        "query_length",
        "selected_agents",
        "completed_agents",
        "critic_verdict",
        "revision_count",
        "errors",
        "error_count",
        "node_latency_seconds",
        "workflow_latency_seconds",
    }
    assert required_fields <= summary.keys()
    assert summary["query_length"] == len(query)
    assert summary["critic_verdict"] == "pass"
    assert summary["revision_count"] == 1
    assert summary["error_count"] == 1
    assert summary["node_latency_seconds"] == {
        "coordinator": 0.1,
        "behavior": 0.5,
        "business": 0.0,
        "experiment": 0.0,
        "parallel": 0.4,
        "decision": 0.5,
        "critic": 0.6,
    }

    log_workflow_summary(
        "workflow.completed",
        query,
        _workflow_result(),
    )
    logged = json.loads(capsys.readouterr().out)
    assert logged["event"] == "workflow.completed"
    assert required_fields <= logged.keys()
    assert logged["timestamp"].endswith("+00:00")


def test_error_log_is_structured_and_redacts_api_key(monkeypatch, capsys):
    secret = "unit-test-token-123456"
    monkeypatch.setenv("TOKENHUB_API_KEY", secret)

    payload = log_error_event(
        "api.analyze.failed",
        query=f"Analyze revenue with TOKENHUB_API_KEY={secret}",
        thread_id="thread-error",
        run_id="run-error",
        error=RuntimeError(f"Provider rejected {secret}"),
        workflow_latency_seconds=0.25,
        http_status=500,
    )

    output = capsys.readouterr().out
    logged = json.loads(output)
    assert logged == payload
    assert logged["event"] == "api.analyze.failed"
    assert logged["error_type"] == "RuntimeError"
    assert logged["error_count"] == 1
    assert logged["http_status"] == 500
    assert secret not in output
    assert "[REDACTED]" in output
