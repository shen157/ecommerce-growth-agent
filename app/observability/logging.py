"""Structured JSON logging without an external monitoring dependency."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any


LOGGER_NAME = "ecommerce_growth_agent.observability"
QUERY_SUMMARY_LIMIT = 160

_SENSITIVE_NAME = re.compile(
    r"(?:api[_-]?key|authorization|password|secret|token)",
    re.IGNORECASE,
)
_ASSIGNED_SECRET = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:API[_-]?KEY|AUTHORIZATION|PASSWORD|SECRET|TOKEN)"
    r"[A-Z0-9_]*)\s*[:=]\s*[^\s,;]+"
)
_BEARER_SECRET = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{8,}")
_KEY_LIKE_SECRET = re.compile(r"\b(?:sk|rk|pk)-[A-Za-z0-9_-]{8,}\b")

_NODE_NAMES = {
    "coordinator": "coordinator",
    "behavior_agent": "behavior",
    "business_agent": "business",
    "experiment_agent": "experiment",
    "parallel_specialists": "parallel",
    "decision_agent": "decision",
    "fast_decision": "decision",
    "critic_agent": "critic",
}
_STANDARD_NODES = (
    "coordinator",
    "behavior",
    "business",
    "experiment",
    "parallel",
    "decision",
    "critic",
)


class JsonFormatter(logging.Formatter):
    """Render only the pre-sanitized structured event as one JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(
            record,
            "structured_event",
            {"event": record.getMessage()},
        )
        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )


class StdoutHandler(logging.StreamHandler):
    """Use the current stdout so Docker and test capture see each event."""

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stdout
        super().emit(record)


def _get_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)

    if not any(
        getattr(handler, "_observability_json", False)
        for handler in logger.handlers
    ):
        handler = StdoutHandler()
        handler._observability_json = True  # type: ignore[attr-defined]
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


logger = _get_logger()


def _known_secret_values() -> list[str]:
    return [
        value
        for name, value in os.environ.items()
        if value and len(value) >= 4 and _SENSITIVE_NAME.search(name)
    ]


def redact_text(value: Any) -> str:
    """Remove common credential forms and configured secret values."""
    text = str(value)

    for secret in _known_secret_values():
        text = text.replace(secret, "[REDACTED]")

    text = _ASSIGNED_SECRET.sub(
        lambda match: f"{match.group(1)}=[REDACTED]",
        text,
    )
    text = _BEARER_SECRET.sub("Bearer [REDACTED]", text)
    return _KEY_LIKE_SECRET.sub("[REDACTED]", text)


def sanitize(value: Any, key: str | None = None) -> Any:
    """Recursively sanitize values before they reach the logging system."""
    if key is not None and _SENSITIVE_NAME.search(key):
        return "[REDACTED]"

    if isinstance(value, Mapping):
        return {
            str(item_key): sanitize(item_value, str(item_key))
            for item_key, item_value in value.items()
        }

    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [sanitize(item) for item in value]

    if isinstance(value, str):
        return redact_text(value)

    return value


def log_event(
    event: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> dict[str, Any]:
    """Emit one sanitized JSON event to stdout and return its payload."""
    payload = sanitize(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
    )
    logger.log(
        level,
        event,
        extra={"structured_event": payload},
    )
    return payload


def aggregate_node_latency(
    latency_trace: Sequence[Mapping[str, Any]] | None,
    run_id: str | None = None,
) -> dict[str, float]:
    """Sum repeated node timings into a stable, dashboard-friendly shape."""
    totals = {node: 0.0 for node in _STANDARD_NODES}

    for item in latency_trace or []:
        if run_id is not None and str(item.get("run_id")) != run_id:
            continue

        raw_name = str(item.get("node", "unknown"))
        node_name = _NODE_NAMES.get(raw_name, raw_name)

        try:
            seconds = max(0.0, float(item.get("seconds", 0.0)))
        except (TypeError, ValueError):
            seconds = 0.0

        totals[node_name] = totals.get(node_name, 0.0) + seconds

    return {
        node: round(seconds, 6)
        for node, seconds in totals.items()
    }


def _mapping_or_attribute(
    value: Any,
    name: str,
    default: Any = None,
) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def build_workflow_summary(
    query: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the common successful workflow/API summary fields."""
    run_id = str(result.get("run_id", "")) or None
    coordinator = result.get("coordinator_decision")
    selected_agents = result.get("selected_agents")

    if selected_agents is None:
        selected_agents = _mapping_or_attribute(
            coordinator,
            "selected_agents",
            [],
        )

    critique = result.get("critique")
    errors = list(result.get("errors") or [])
    query_summary = " ".join(query.split())[:QUERY_SUMMARY_LIMIT]

    return sanitize(
        {
            "run_id": run_id,
            "thread_id": result.get("thread_id"),
            "query_summary": query_summary,
            "query_length": len(query),
            "selected_agents": list(selected_agents or []),
            "completed_agents": list(
                result.get("completed_agents") or []
            ),
            "critic_verdict": _mapping_or_attribute(
                critique,
                "verdict",
            ),
            "revision_count": int(result.get("revision_count") or 0),
            "errors": errors,
            "error_count": len(errors),
            "node_latency_seconds": aggregate_node_latency(
                result.get("latency_trace"),
                run_id,
            ),
            "workflow_latency_seconds": round(
                max(
                    0.0,
                    float(result.get("workflow_latency_seconds") or 0.0),
                ),
                6,
            ),
        }
    )


def log_workflow_summary(
    event: str,
    query: str,
    result: Mapping[str, Any],
    **fields: Any,
) -> dict[str, Any]:
    """Log a successful workflow summary."""
    return log_event(
        event,
        **build_workflow_summary(query, result),
        **fields,
    )


def log_error_event(
    event: str,
    *,
    query: str,
    thread_id: str | None,
    error: Exception,
    run_id: str | None = None,
    workflow_latency_seconds: float = 0.0,
    **fields: Any,
) -> dict[str, Any]:
    """Log a safe error event without a traceback or secret-bearing repr."""
    query_summary = " ".join(query.split())[:QUERY_SUMMARY_LIMIT]
    return log_event(
        event,
        level=logging.ERROR,
        run_id=run_id,
        thread_id=thread_id,
        query_summary=query_summary,
        query_length=len(query),
        selected_agents=[],
        completed_agents=[],
        critic_verdict=None,
        revision_count=0,
        errors=[redact_text(error)],
        error_count=1,
        error_type=type(error).__name__,
        node_latency_seconds={node: 0.0 for node in _STANDARD_NODES},
        workflow_latency_seconds=round(
            max(0.0, workflow_latency_seconds),
            6,
        ),
        **fields,
    )
