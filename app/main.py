from collections.abc import Mapping
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.graph.workflow import run_workflow
from app.observability.logging import (
    log_error_event,
    log_workflow_summary,
)
from app.schemas.models import DecisionReport, SpecialistName


class HealthResponse(BaseModel):
    status: Literal["ok"]


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1)
    thread_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
    )


class LatencyEvent(BaseModel):
    run_id: str
    node: str
    seconds: float = Field(ge=0)


class AnalyzeResponse(BaseModel):
    thread_id: str
    run_id: str
    selected_agents: list[SpecialistName]
    completed_agents: list[SpecialistName]
    decision: DecisionReport | None
    critic_verdict: Literal["pass", "revise"] | None
    errors: list[str]
    latency_trace: list[LatencyEvent]
    workflow_latency_seconds: float = Field(ge=0)


app = FastAPI(
    title="Ecommerce Growth Agent API",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    try:
        result = run_workflow(
            query=request.query,
            thread_id=request.thread_id,
        )
        response = _build_analyze_response(result)
        log_workflow_summary(
            "api.analyze.completed",
            request.query,
            result,
            http_status=200,
        )
        return response
    except Exception as exc:
        log_error_event(
            "api.analyze.failed",
            query=request.query,
            thread_id=request.thread_id,
            error=exc,
            http_status=500,
        )
        raise HTTPException(
            status_code=500,
            detail="Workflow analysis failed",
        ) from exc


def _build_analyze_response(result: Mapping) -> AnalyzeResponse:
    run_id = str(result["run_id"])
    coordinator = result.get("coordinator_decision")
    selected_agents = result.get("selected_agents")

    if selected_agents is None and coordinator is not None:
        selected_agents = (
            coordinator.get("selected_agents", [])
            if isinstance(coordinator, Mapping)
            else coordinator.selected_agents
        )

    critique = result.get("critique")
    critic_verdict = None
    if critique is not None:
        critic_verdict = (
            critique.get("verdict")
            if isinstance(critique, Mapping)
            else critique.verdict
        )

    latency_trace = [
        event
        for event in result.get("latency_trace", [])
        if event.get("run_id") == run_id
    ]

    return AnalyzeResponse(
        thread_id=str(result["thread_id"]),
        run_id=run_id,
        selected_agents=selected_agents or [],
        completed_agents=result.get("completed_agents", []),
        decision=result.get("decision_report"),
        critic_verdict=critic_verdict,
        errors=result.get("errors", []),
        latency_trace=latency_trace,
        workflow_latency_seconds=result.get(
            "workflow_latency_seconds",
            0.0,
        ),
    )
