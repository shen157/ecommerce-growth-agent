from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from functools import wraps
from time import perf_counter
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)

from langgraph.checkpoint.memory import (
    InMemorySaver,
)

from langgraph.checkpoint.serde.jsonplus import (
    JsonPlusSerializer,
)

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from app.schemas.models import (
    CoordinatorDecision,
    CriticReport,
    DecisionReport,
    SpecialistReport,
)

from app.agents.behavior_agent import (
    run_behavior_agent,
)

from app.agents.business_agent import (
    run_business_agent,
)

from app.agents.coordinator import (
    coordinator_node,
)

from app.agents.critic_agent import (
    critic_agent_node,
)

from app.agents.decision_agent import (
    decision_agent_node,
    fast_single_specialist_decision_node,
)

from app.agents.experiment_agent import (
    run_experiment_agent,
)

from app.graph.router import (
    route_after_critic,
    route_next_specialist,
)

from app.graph.state import (
    AgentState,
)

from app.graph.memory import (
    build_contextual_query,
)

from app.observability.logging import (
    log_error_event,
    log_workflow_summary,
)


# =========================================================
# Helpers
# =========================================================
def _mark_completed(
    state: AgentState,
    agent_name: str,
) -> list:

    completed = list(
        state.get(
            "completed_agents",
            [],
        )
    )

    if agent_name not in completed:
        completed.append(
            agent_name
        )

    return completed


def _existing_context(
    state: AgentState,
) -> str:

    context = []

    behavior = state.get(
        "behavior_report"
    )

    business = state.get(
        "business_report"
    )

    experiment = state.get(
        "experiment_report"
    )

    if behavior is not None:

        context.append(
            "BEHAVIOR AGENT CONTEXT:\n"
            +
            behavior.model_dump_json(
                indent=2
            )
        )

    if business is not None:

        context.append(
            "BUSINESS AGENT CONTEXT:\n"
            +
            business.model_dump_json(
                indent=2
            )
        )

    if experiment is not None:

        context.append(
            "EXPERIMENT AGENT CONTEXT:\n"
            +
            experiment.model_dump_json(
                indent=2
            )
        )

    return "\n\n".join(
        context
    )


def _build_specialist_query(
    state: AgentState,
    target_agent: str,
) -> str:

    existing_context = (
        _existing_context(
            state
        )
    )

    contextual_query = (
        build_contextual_query(
            state
        )
    )

    if not existing_context:
        return contextual_query

    return f"""
ORIGINAL USER QUESTION:

{contextual_query}


PREVIOUS SPECIALIST CONTEXT:

{existing_context}


You are now acting as:

{target_agent}


Use previous specialist output only as context.
For quantitative claims in your own domain,
use your own analytical tools.

Do not invent missing metrics.
""".strip()


# =========================================================
# Performance Profiling
# =========================================================
def _timed_node(
    node_name: str,
    node_func,
):

    @wraps(node_func)
    def wrapper(
        state: AgentState,
    ) -> dict:

        start = perf_counter()

        result = node_func(
            state
        )

        elapsed = (
            perf_counter()
            -
            start
        )

        result = dict(
            result
        )

        existing_trace = list(
    result.get(
        "latency_trace",
        [],
    )
)

        existing_trace.append(
            {
                "run_id":
                    state["run_id"],

                "node":
                    node_name,

                "seconds":
                    elapsed,
            }
        )

        result[
            "latency_trace"
        ] = existing_trace

        return result

    return wrapper


# =========================================================
# Specialist Nodes
# =========================================================
def behavior_agent_node(
    state: AgentState,
) -> dict:

    query = _build_specialist_query(
        state,
        "behavior_agent",
    )

    errors = list(
        state.get(
            "errors",
            [],
        )
    )

    try:

        report = run_behavior_agent(
            query
        )

        return {
            "behavior_report":
                report,

            "completed_agents":
                _mark_completed(
                    state,
                    "behavior_agent",
                ),
        }

    except Exception as exc:

        errors.append(
            "behavior_agent failed: "
            f"{type(exc).__name__}: {exc}"
        )

        return {
            "completed_agents":
                _mark_completed(
                    state,
                    "behavior_agent",
                ),

            "errors":
                errors,
        }


def business_agent_node(
    state: AgentState,
) -> dict:

    query = _build_specialist_query(
        state,
        "business_agent",
    )

    errors = list(
        state.get(
            "errors",
            [],
        )
    )

    try:

        report = run_business_agent(
            query
        )

        return {
            "business_report":
                report,

            "completed_agents":
                _mark_completed(
                    state,
                    "business_agent",
                ),
        }

    except Exception as exc:

        errors.append(
            "business_agent failed: "
            f"{type(exc).__name__}: {exc}"
        )

        return {
            "completed_agents":
                _mark_completed(
                    state,
                    "business_agent",
                ),

            "errors":
                errors,
        }


def experiment_agent_node(
    state: AgentState,
) -> dict:

    query = _build_specialist_query(
        state,
        "experiment_agent",
    )

    errors = list(
        state.get(
            "errors",
            [],
        )
    )

    try:

        report = run_experiment_agent(
            query
        )

        return {
            "experiment_report":
                report,

            "completed_agents":
                _mark_completed(
                    state,
                    "experiment_agent",
                ),
        }

    except Exception as exc:

        errors.append(
            "experiment_agent failed: "
            f"{type(exc).__name__}: {exc}"
        )

        return {
            "completed_agents":
                _mark_completed(
                    state,
                    "experiment_agent",
                ),

            "errors":
                errors,
        }

def parallel_specialists_node(
    state: AgentState,
) -> dict:
    """
    Run multiple selected specialists concurrently.

    Each specialist receives the same user/conversation
    context and performs its own domain analysis.

    Cross-domain synthesis is handled later by the
    Decision Agent.
    """

    selected_agents = list(
        state.get(
            "selected_agents",
            [],
        )
    )

    query = build_contextual_query(
        state
    )

    runner_map = {
        "behavior_agent":
            run_behavior_agent,

        "business_agent":
            run_business_agent,

        "experiment_agent":
            run_experiment_agent,
    }

    report_key_map = {
        "behavior_agent":
            "behavior_report",

        "business_agent":
            "business_report",

        "experiment_agent":
            "experiment_report",
    }

    run_id = state[
        "run_id"
    ]

    reports = {}
    timings = {}
    failures = {}

    def run_one(
        agent_name: str,
    ):

        start = perf_counter()

        try:

            report = runner_map[
                agent_name
            ](
                query
            )

            elapsed = (
                perf_counter()
                -
                start
            )

            return (
                agent_name,
                report,
                elapsed,
                None,
            )

        except Exception as exc:

            elapsed = (
                perf_counter()
                -
                start
            )

            return (
                agent_name,
                None,
                elapsed,
                (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

    with ThreadPoolExecutor(
        max_workers=len(
            selected_agents
        )
    ) as executor:

        futures = {
            executor.submit(
                run_one,
                agent_name,
            ):
                agent_name

            for agent_name
            in selected_agents
        }

        for future in as_completed(
            futures
        ):

            (
                agent_name,
                report,
                elapsed,
                error,
            ) = future.result()

            timings[
                agent_name
            ] = elapsed

            if report is not None:

                reports[
                    report_key_map[
                        agent_name
                    ]
                ] = report

            if error is not None:

                failures[
                    agent_name
                ] = error

    errors = list(
        state.get(
            "errors",
            [],
        )
    )

    for agent_name in selected_agents:

        if agent_name in failures:

            errors.append(
                f"{agent_name} failed: "
                f"{failures[agent_name]}"
            )

    latency_trace = []

    for agent_name in selected_agents:

        if agent_name in timings:

            latency_trace.append(
                {
                    "run_id":
                        run_id,

                    "node":
                        agent_name,

                    "seconds":
                        timings[
                            agent_name
                        ],
                }
            )

    return {
        **reports,

        "completed_agents":
            selected_agents,

        "errors":
            errors,

        "latency_trace":
            latency_trace,
    }
# =========================================================
# Conversation Memory
# =========================================================
def save_conversation_node(
    state: AgentState,
) -> dict:

    decision = state.get(
        "decision_report"
    )

    if decision is None:

        answer = (
            "The workflow finished without "
            "a decision report."
        )

    else:

        answer = decision.conclusion

        if decision.recommendation:

            answer += (
                "\n\nRecommendation: "
                + decision.recommendation
            )

    return {
        "messages": [
            AIMessage(
                content=answer
            )
        ]
    }


# =========================================================
# Build Workflow
# =========================================================
def build_workflow(
    checkpointer=None,
):

    workflow = StateGraph(
        AgentState
    )

    # =========================================================
    # Nodes
    # =========================================================
    workflow.add_node(
        "coordinator_node",
        _timed_node(
            "coordinator",
            coordinator_node,
        ),
    )

    workflow.add_node(
        "behavior_agent_node",
        _timed_node(
            "behavior_agent",
            behavior_agent_node,
        ),
    )

    workflow.add_node(
        "business_agent_node",
        _timed_node(
            "business_agent",
            business_agent_node,
        ),
    )

    workflow.add_node(
        "experiment_agent_node",
        _timed_node(
            "experiment_agent",
            experiment_agent_node,
        ),
    )

    workflow.add_node(
        "parallel_specialists_node",
        _timed_node(
            "parallel_specialists",
            parallel_specialists_node,
        ),
    )

    workflow.add_node(
        "decision_agent_node",
        _timed_node(
            "decision_agent",
            decision_agent_node,
        ),
    )

    workflow.add_node(
        "fast_decision_agent_node",
        _timed_node(
            "fast_decision",
            fast_single_specialist_decision_node,
        ),
    )

    workflow.add_node(
        "critic_agent_node",
        _timed_node(
            "critic_agent",
            critic_agent_node,
        ),
    )

    workflow.add_node(
        "save_conversation_node",
        save_conversation_node,
    )

    # =========================================================
    # Start
    # =========================================================
    workflow.add_edge(
        START,
        "coordinator_node",
    )

    # =========================================================
    # Coordinator Routing
    # =========================================================
    specialist_routes = {
        "behavior_agent_node":
            "behavior_agent_node",

        "business_agent_node":
            "business_agent_node",

        "experiment_agent_node":
            "experiment_agent_node",

        "parallel_specialists_node":
            "parallel_specialists_node",

        "fast_decision_agent_node":
            "fast_decision_agent_node",

        "decision_agent_node":
            "decision_agent_node",
    }

    workflow.add_conditional_edges(
        "coordinator_node",
        route_next_specialist,
        specialist_routes,
    )

    # =========================================================
    # Specialist Routing
    # =========================================================
    for node_name in [
        "behavior_agent_node",
        "business_agent_node",
        "experiment_agent_node",
    ]:

        workflow.add_conditional_edges(
            node_name,
            route_next_specialist,
            specialist_routes,
        )

    workflow.add_edge(
        "parallel_specialists_node",
        "decision_agent_node",
    )
    # =========================================================
    # Decision -> Critic
    # =========================================================
    workflow.add_edge(
        "decision_agent_node",
        "critic_agent_node",
    )

    workflow.add_edge(
        "fast_decision_agent_node",
        "critic_agent_node",
    )

    # =========================================================
    # Critic Routing
    # =========================================================
    workflow.add_conditional_edges(
        "critic_agent_node",
        route_after_critic,
        {
            "revise":
                "decision_agent_node",

            "finish":
                "save_conversation_node",
        },
    )

    workflow.add_edge(
        "save_conversation_node",
        END,
    )

    return workflow.compile(
        checkpointer=checkpointer
    )


# =========================================================
# Checkpointer
# =========================================================
_checkpoint_serializer = (
    JsonPlusSerializer(
        allowed_msgpack_modules=[
            SpecialistReport,
            DecisionReport,
            CriticReport,
            CoordinatorDecision,
        ]
    )
)


_default_checkpointer = (
    InMemorySaver(
        serde=_checkpoint_serializer
    )
)


_default_app = None


def get_workflow():

    global _default_app

    if _default_app is None:

        _default_app = build_workflow(
            checkpointer=(
                _default_checkpointer
            )
        )

    return _default_app


# =========================================================
# Run Workflow
# =========================================================
def run_workflow(
    query: str,
    thread_id: str | None = None,
    app=None,
):

    workflow_app = (
        app
        if app is not None
        else get_workflow()
    )

    effective_thread_id = (
        thread_id
        if thread_id is not None
        else str(uuid4())
    )

    # IMPORTANT:
    # Generate a fresh run_id for every invocation.
    current_run_id = (
        uuid4().hex[:12]
    )

    initial_state = {
        "user_query":
            query,

        "thread_id":
            effective_thread_id,

        "messages": [
            HumanMessage(
                content=query
            )
        ],

        "run_id":
            current_run_id,

        "latency_trace":
            [],

        "completed_agents":
            [],

        "revision_count":
            0,

        "errors":
            [],

        # Clear reports from previous turn.
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
    }

    start = perf_counter()

    try:
        result = workflow_app.invoke(
            initial_state,
            config={
                "configurable": {
                    "thread_id":
                        effective_thread_id,
                }
            },
        )
    except Exception as exc:
        log_error_event(
            "workflow.failed",
            query=query,
            thread_id=effective_thread_id,
            run_id=current_run_id,
            error=exc,
            workflow_latency_seconds=(
                perf_counter()
                -
                start
            ),
        )
        raise

    total_latency = (
        perf_counter()
        -
        start
    )

    result[
        "workflow_latency_seconds"
    ] = total_latency

    log_workflow_summary(
        "workflow.completed",
        query,
        result,
    )

    return result


# =========================================================
# Manual Test
# =========================================================
if __name__ == "__main__":

    query = (
        "Which categories do repeat customers "
        "prefer, and are those categories "
        "performing well in revenue?"
    )

    result = run_workflow(
        query,
        thread_id="manual-demo",
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "COORDINATOR"
    )

    print(
        "=" * 70
    )

    print(
        result[
            "coordinator_decision"
        ].model_dump_json(
            indent=2
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "COMPLETED AGENTS"
    )

    print(
        "=" * 70
    )

    print(
        result.get(
            "completed_agents"
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DECISION"
    )

    print(
        "=" * 70
    )

    print(
        result[
            "decision_report"
        ].model_dump_json(
            indent=2
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CRITIC"
    )

    print(
        "=" * 70
    )

    print(
        result[
            "critique"
        ].model_dump_json(
            indent=2
        )
    )

    print(
        "\nRevision count:",
        result.get(
            "revision_count",
            0,
        )
    )

    print(
        "\nErrors:",
        result.get(
            "errors",
            [],
        )
    )

    print(
        "\nWorkflow latency:",
        round(
            result.get(
                "workflow_latency_seconds",
                0,
            ),
            2,
        ),
        "seconds",
    )

    print(
        "\nLatency trace:"
    )

    for event in result.get(
        "latency_trace",
        [],
    ):

        if (
            event.get("run_id")
            ==
            result.get("run_id")
        ):

            print(
                f"{event['node']:20s}: "
                f"{event['seconds']:.2f}s"
            )
