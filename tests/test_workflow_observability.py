import pytest

from app.graph import workflow as workflow_module


class SuccessfulWorkflowApp:
    def invoke(self, state, config):
        assert config["configurable"]["thread_id"] == "thread-direct"
        return {
            **state,
            "selected_agents": ["behavior_agent"],
            "completed_agents": ["behavior_agent"],
        }


class FailingWorkflowApp:
    def invoke(self, state, config):
        raise RuntimeError("provider failed")


def test_run_workflow_logs_completed_summary(monkeypatch):
    captured = {}

    def fake_log_workflow_summary(event, query, result, **fields):
        captured.update(
            {
                "event": event,
                "query": query,
                "result": result,
                **fields,
            }
        )

    monkeypatch.setattr(
        workflow_module,
        "log_workflow_summary",
        fake_log_workflow_summary,
    )

    result = workflow_module.run_workflow(
        "Analyze customers",
        thread_id="thread-direct",
        app=SuccessfulWorkflowApp(),
    )

    assert captured["event"] == "workflow.completed"
    assert captured["query"] == "Analyze customers"
    assert captured["result"] is result
    assert result["workflow_latency_seconds"] >= 0


def test_run_workflow_logs_failure_and_reraises(monkeypatch):
    captured = {}

    def fake_log_error_event(event, **fields):
        captured["event"] = event
        captured.update(fields)

    monkeypatch.setattr(
        workflow_module,
        "log_error_event",
        fake_log_error_event,
    )

    with pytest.raises(RuntimeError, match="provider failed"):
        workflow_module.run_workflow(
            "Analyze customers",
            thread_id="thread-direct",
            app=FailingWorkflowApp(),
        )

    assert captured["event"] == "workflow.failed"
    assert captured["query"] == "Analyze customers"
    assert captured["thread_id"] == "thread-direct"
    assert len(captured["run_id"]) == 12
    assert isinstance(captured["error"], RuntimeError)
    assert captured["workflow_latency_seconds"] >= 0
