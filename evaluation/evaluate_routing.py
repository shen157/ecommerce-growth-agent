import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from app.agents.coordinator import (
    run_coordinator,
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


def load_cases():

    with open(
        CASE_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def main():

    data = load_cases()

    cases = data[
        "single_turn_cases"
    ]

    results = []

    latencies = []

    print(
        "=" * 70
    )

    print(
        "ROUTING EVALUATION"
    )

    print(
        "=" * 70
    )

    for case in cases:

        case_id = case["id"]
        query = case["query"]

        expected_agents = case[
            "expected_agents"
        ]

        start = perf_counter()

        try:

            decision = run_coordinator(
                query
            )

            latency = (
                perf_counter()
                -
                start
            )

            predicted_agents = (
                decision.selected_agents
            )

            metrics = (
                agent_selection_metrics(
                    expected_agents,
                    predicted_agents,
                )
            )

            error = None

        except Exception as exc:

            latency = (
                perf_counter()
                -
                start
            )

            predicted_agents = []

            metrics = (
                agent_selection_metrics(
                    expected_agents,
                    predicted_agents,
                )
            )

            error = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

        latencies.append(
            latency
        )

        result = {
            "id":
                case_id,

            "query":
                query,

            "expected_agents":
                expected_agents,

            "predicted_agents":
                predicted_agents,

            "precision":
                metrics["precision"],

            "recall":
                metrics["recall"],

            "f1":
                metrics["f1"],

            "exact_set_match":
                metrics[
                    "exact_set_match"
                ],

            "exact_order_match":
                metrics[
                    "exact_order_match"
                ],

            "latency_seconds":
                latency,

            "error":
                error,
        }

        results.append(
            result
        )

        status = (
            "PASS"
            if result[
                "exact_set_match"
            ]
            else "FAIL"
        )

        print(
            f"{case_id}: "
            f"{status} "
            f"| expected={expected_agents} "
            f"| predicted={predicted_agents}"
        )

    # =========================================================
    # Aggregate
    # =========================================================
    total = len(results)

    exact_correct = sum(
        item["exact_set_match"]
        for item in results
    )

    order_correct = sum(
        item["exact_order_match"]
        for item in results
    )

    error_count = sum(
        item["error"] is not None
        for item in results
    )

    summary = {
        "total_cases":
            total,

        "routing_accuracy":
            exact_correct / total,

        "routing_order_accuracy":
            order_correct / total,

        "agent_selection_precision":
            mean(
                item["precision"]
                for item in results
            ),

        "agent_selection_recall":
            mean(
                item["recall"]
                for item in results
            ),

        "agent_selection_f1":
            mean(
                item["f1"]
                for item in results
            ),

        "error_rate":
            error_count / total,

        "average_latency_seconds":
            mean(latencies),

        "p95_latency_seconds":
            percentile(
                latencies,
                95,
            ),
    }

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ROUTING SUMMARY"
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

    output = {
        "summary":
            summary,

        "cases":
            results,
    }

    output_path = (
        RESULT_DIR
        / "routing_results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\nSaved:",
        output_path,
    )


if __name__ == "__main__":
    main()