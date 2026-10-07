import json

from collections import defaultdict
from pathlib import Path
from statistics import mean
from time import perf_counter

from app.graph.workflow import (
    get_workflow,
    run_workflow,
)

from evaluation.metrics import (
    agent_selection_metrics,
    percentile,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

CASE_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "test_cases.json"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# Load Cases
# =========================================================
def load_cases():

    with open(
        CASE_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# =========================================================
# Latency Helpers
# =========================================================
def extract_current_run_latencies(
    result: dict,
) -> dict[str, float]:

    current_run_id = result.get(
        "run_id"
    )

    events = [
        event
        for event in result.get(
            "latency_trace",
            [],
        )
        if event.get(
            "run_id"
        ) == current_run_id
    ]

    node_latencies = {}

    for event in events:

        node = event[
            "node"
        ]

        seconds = float(
            event[
                "seconds"
            ]
        )

        # A node may execute more than once,
        # e.g. Decision -> Critic -> Revision.
        node_latencies[node] = (
            node_latencies.get(
                node,
                0.0,
            )
            +
            seconds
        )

    return node_latencies


# =========================================================
# Result Evaluation
# =========================================================
def evaluate_result(
    result: dict,
    expected_agents: list[str],
    expected_recommendation: str | None,
) -> dict:

    coordinator = result.get(
        "coordinator_decision"
    )

    predicted_agents = (
        coordinator.selected_agents
        if coordinator is not None
        else []
    )

    routing = (
        agent_selection_metrics(
            expected_agents,
            predicted_agents,
        )
    )

    decision = result.get(
        "decision_report"
    )

    critique = result.get(
        "critique"
    )

    errors = result.get(
        "errors",
        [],
    )

    recommendation_match = None

    predicted_recommendation = None

    if (
        expected_recommendation
        is not None
        and decision is not None
    ):

        predicted_recommendation = (
            decision.decision_label
        )

        recommendation_match = (
            predicted_recommendation
            ==
            expected_recommendation
        )

    critic_pass = (
        critique is not None
        and critique.verdict == "pass"
    )

    workflow_completed = (
        decision is not None
        and critique is not None
        and not errors
    )

    automated_success = (
        workflow_completed
        and routing[
            "exact_set_match"
        ]
        and critic_pass
    )

    if recommendation_match is not None:

        automated_success = (
            automated_success
            and recommendation_match
        )

    return {
        "predicted_agents":
            predicted_agents,

        "routing_exact":
            routing[
                "exact_set_match"
            ],

        "agent_precision":
            routing[
                "precision"
            ],

        "agent_recall":
            routing[
                "recall"
            ],

        "agent_f1":
            routing[
                "f1"
            ],

        "workflow_completed":
            workflow_completed,

        "critic_pass":
            critic_pass,

        "revision_count":
            result.get(
                "revision_count",
                0,
            ),

        "specialist_calls":
            len(
                result.get(
                    "completed_agents",
                    [],
                )
            ),

        "expected_recommendation":
            expected_recommendation,

        "predicted_recommendation":
            predicted_recommendation,

        "recommendation_match":
            recommendation_match,

        "automated_success":
            automated_success,

        "errors":
            errors,
    }


# =========================================================
# Single-Turn Cases
# =========================================================
def run_single_turn_cases(
    cases: list,
    app,
):

    outputs = []

    for case in cases:

        case_id = case[
            "id"
        ]

        print(
            f"\nRunning {case_id}..."
        )

        thread_id = (
            f"eval-single-{case_id}"
        )

        start = perf_counter()

        result = {}

        node_latencies = {}

        try:

            result = run_workflow(
                query=case[
                    "query"
                ],
                thread_id=thread_id,
                app=app,
            )

            latency = (
                perf_counter()
                -
                start
            )

            node_latencies = (
                extract_current_run_latencies(
                    result
                )
            )

            metrics = evaluate_result(
                result=result,
                expected_agents=case[
                    "expected_agents"
                ],
                expected_recommendation=(
                    case.get(
                        "expected_recommendation"
                    )
                ),
            )

            fatal_error = None

        except Exception as exc:

            latency = (
                perf_counter()
                -
                start
            )

            metrics = {
                "predicted_agents":
                    [],

                "routing_exact":
                    False,

                "agent_precision":
                    0.0,

                "agent_recall":
                    0.0,

                "agent_f1":
                    0.0,

                "workflow_completed":
                    False,

                "critic_pass":
                    False,

                "revision_count":
                    0,

                "specialist_calls":
                    0,

                "expected_recommendation":
                    case.get(
                        "expected_recommendation"
                    ),

                "predicted_recommendation":
                    None,

                "recommendation_match":
                    (
                        False
                        if case.get(
                            "expected_recommendation"
                        )
                        else None
                    ),

                "automated_success":
                    False,

                "errors":
                    [],
            }

            fatal_error = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

        output = {
            "id":
                case_id,

            "query":
                case[
                    "query"
                ],

            "expected_agents":
                case[
                    "expected_agents"
                ],

            "latency_seconds":
                latency,

            "workflow_reported_latency":
                result.get(
                    "workflow_latency_seconds"
                ),

            "node_latencies":
                node_latencies,

            "fatal_error":
                fatal_error,

            **metrics,
        }

        outputs.append(
            output
        )

        status = (
            "PASS"
            if output[
                "automated_success"
            ]
            else "FAIL"
        )

        print(
            f"{case_id}: "
            f"{status} "
            f"| {latency:.2f}s"
        )

    return outputs


# =========================================================
# Memory Cases
# =========================================================
def run_memory_cases(
    cases: list,
    app,
):

    outputs = []

    for case in cases:

        case_id = case[
            "id"
        ]

        thread_id = (
            f"eval-memory-{case_id}"
        )

        print(
            f"\nMemory case {case_id}"
        )

        turn_outputs = []

        for index, turn in enumerate(
            case[
                "turns"
            ],
            start=1,
        ):

            start = perf_counter()

            result = {}

            node_latencies = {}

            try:

                result = run_workflow(
                    query=turn[
                        "query"
                    ],
                    thread_id=thread_id,
                    app=app,
                )

                latency = (
                    perf_counter()
                    -
                    start
                )

                node_latencies = (
                    extract_current_run_latencies(
                        result
                    )
                )

                metrics = evaluate_result(
                    result=result,
                    expected_agents=turn[
                        "expected_agents"
                    ],
                    expected_recommendation=(
                        turn.get(
                            "expected_recommendation"
                        )
                    ),
                )

                fatal_error = None

            except Exception as exc:

                latency = (
                    perf_counter()
                    -
                    start
                )

                metrics = {
                    "predicted_agents":
                        [],

                    "routing_exact":
                        False,

                    "agent_precision":
                        0.0,

                    "agent_recall":
                        0.0,

                    "agent_f1":
                        0.0,

                    "workflow_completed":
                        False,

                    "critic_pass":
                        False,

                    "revision_count":
                        0,

                    "specialist_calls":
                        0,

                    "expected_recommendation":
                        turn.get(
                            "expected_recommendation"
                        ),

                    "predicted_recommendation":
                        None,

                    "recommendation_match":
                        (
                            False
                            if turn.get(
                                "expected_recommendation"
                            )
                            else None
                        ),

                    "automated_success":
                        False,

                    "errors":
                        [],
                }

                fatal_error = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

            turn_output = {
                "turn":
                    index,

                "query":
                    turn[
                        "query"
                    ],

                "latency_seconds":
                    latency,

                "workflow_reported_latency":
                    result.get(
                        "workflow_latency_seconds"
                    ),

                "node_latencies":
                    node_latencies,

                "fatal_error":
                    fatal_error,

                **metrics,
            }

            turn_outputs.append(
                turn_output
            )

            print(
                f"  Turn {index}: "
                f"{'PASS' if turn_output.get('automated_success') else 'FAIL'} "
                f"| {latency:.2f}s"
            )

        followups = (
            turn_outputs[
                1:
            ]
        )

        memory_success = (
            all(
                turn.get(
                    "automated_success",
                    False,
                )
                for turn in followups
            )
            if followups
            else True
        )

        outputs.append(
            {
                "id":
                    case_id,

                "memory_success":
                    memory_success,

                "turns":
                    turn_outputs,
            }
        )

    return outputs


# =========================================================
# Main
# =========================================================
def main():

    data = load_cases()

    app = get_workflow()

    single_results = (
        run_single_turn_cases(
            data[
                "single_turn_cases"
            ],
            app,
        )
    )

    memory_results = (
        run_memory_cases(
            data[
                "memory_cases"
            ],
            app,
        )
    )

    # =========================================================
    # Core Metrics
    # =========================================================
    latencies = [
        result[
            "latency_seconds"
        ]
        for result in single_results
    ]

    successes = [
        result[
            "automated_success"
        ]
        for result in single_results
    ]

    routing_matches = [
        result[
            "routing_exact"
        ]
        for result in single_results
    ]

    critic_passes = [
        result[
            "critic_pass"
        ]
        for result in single_results
    ]

    revisions = [
        result[
            "revision_count"
        ]
        for result in single_results
    ]

    specialist_calls = [
        result[
            "specialist_calls"
        ]
        for result in single_results
    ]

    fatal_errors = [
        result
        for result in single_results
        if result[
            "fatal_error"
        ] is not None
    ]

    recommendation_cases = [
        result
        for result in single_results
        if result[
            "recommendation_match"
        ] is not None
    ]

    summary = {
        "total_single_turn_cases":
            len(
                single_results
            ),

        "routing_accuracy":
            mean(
                routing_matches
            ),

        "automated_workflow_success_rate":
            mean(
                successes
            ),

        "critic_pass_rate":
            mean(
                critic_passes
            ),

        "critic_revision_rate":
            mean(
                1
                if count > 0
                else 0
                for count in revisions
            ),

        "recommendation_accuracy":
            (
                mean(
                    case[
                        "recommendation_match"
                    ]
                    for case
                    in recommendation_cases
                )
                if recommendation_cases
                else None
            ),

        "memory_followup_accuracy":
            mean(
                case[
                    "memory_success"
                ]
                for case
                in memory_results
            ),

        "error_rate":
            (
                len(
                    fatal_errors
                )
                /
                len(
                    single_results
                )
            ),

        "average_latency_seconds":
            mean(
                latencies
            ),

        "p95_latency_seconds":
            percentile(
                latencies,
                95,
            ),

        "average_specialist_calls":
            mean(
                specialist_calls
            ),
    }

    # =========================================================
    # Node-Level Latency
    # =========================================================
    node_samples = defaultdict(
        list
    )

    for result in single_results:

        for (
            node_name,
            seconds,
        ) in result.get(
            "node_latencies",
            {},
        ).items():

            node_samples[
                node_name
            ].append(
                seconds
            )

    node_latency_summary = {}

    for (
        node_name,
        values,
    ) in node_samples.items():

        node_latency_summary[
            node_name
        ] = {
            "count":
                len(
                    values
                ),

            "average_seconds":
                mean(
                    values
                ),

            "p95_seconds":
                percentile(
                    values,
                    95,
                ),

            "max_seconds":
                max(
                    values
                ),
        }

    # =========================================================
    # Print Workflow Summary
    # =========================================================
    print(
        "\n"
        + "=" * 70
    )

    print(
        "WORKFLOW EVALUATION SUMMARY"
    )

    print(
        "=" * 70
    )

    for key, value in summary.items():

        if isinstance(
            value,
            float,
        ):

            print(
                f"{key}: "
                f"{value:.4f}"
            )

        else:

            print(
                f"{key}: "
                f"{value}"
            )

    # =========================================================
    # Print Node Summary
    # =========================================================
    print(
        "\n"
        + "=" * 70
    )

    print(
        "NODE LATENCY SUMMARY"
    )

    print(
        "=" * 70
    )

    for (
        node_name,
        stats,
    ) in sorted(
        node_latency_summary.items()
    ):

        print(
            f"{node_name:20s} "
            f"| n={stats['count']:2d} "
            f"| avg={stats['average_seconds']:.2f}s "
            f"| p95={stats['p95_seconds']:.2f}s "
            f"| max={stats['max_seconds']:.2f}s"
        )

    # =========================================================
    # Slowest Cases
    # =========================================================
    slowest_cases = sorted(
        single_results,
        key=lambda x: x[
            "latency_seconds"
        ],
        reverse=True,
    )[:5]

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SLOWEST CASES"
    )

    print(
        "=" * 70
    )

    for case in slowest_cases:

        print(
            f"\n{case['id']:5s} "
            f"| {case['latency_seconds']:.2f}s "
            f"| agents="
            f"{case['predicted_agents']}"
        )

        for (
            node,
            seconds,
        ) in case.get(
            "node_latencies",
            {},
        ).items():

            print(
                f"    "
                f"{node:20s}: "
                f"{seconds:.2f}s"
            )

    # =========================================================
    # Save Results
    # =========================================================
    output = {
        "summary":
            summary,

        "node_latency_summary":
            node_latency_summary,

        "single_turn_cases":
            single_results,

        "memory_cases":
            memory_results,
    }

    path = (
        RESULT_DIR
        / "workflow_results.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    print(
        "\nSaved:",
        path,
    )


if __name__ == "__main__":
    main()