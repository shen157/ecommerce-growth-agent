from __future__ import annotations

from typing import Iterable

import numpy as np


def agent_selection_metrics(
    expected: Iterable[str],
    predicted: Iterable[str],
) -> dict:
    """
    Calculate precision / recall / F1 for agent selection.
    """

    expected_set = set(expected)
    predicted_set = set(predicted)

    true_positive = len(
        expected_set & predicted_set
    )

    precision = (
        true_positive / len(predicted_set)
        if predicted_set
        else 0.0
    )

    recall = (
        true_positive / len(expected_set)
        if expected_set
        else 0.0
    )

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_set_match":
            expected_set == predicted_set,
        "exact_order_match":
            list(expected) == list(predicted),
    }


def percentile(
    values: list[float],
    q: float,
) -> float:
    """
    Calculate percentile safely.
    """

    if not values:
        return 0.0

    return float(
        np.percentile(
            values,
            q,
        )
    )


def normalize_recommendation(
    text: str | None,
) -> str | None:
    """
    Normalize free-text rollout recommendations.
    """

    if not text:
        return None

    value = text.lower()

    negative_patterns = [
        "do not roll out",
        "do_not_rollout",
        "don't roll out",
        "not roll out",
        "should not roll out",
    ]

    for pattern in negative_patterns:
        if pattern in value:
            return "do_not_rollout"

    positive_patterns = [
        "roll out",
        "rollout",
        "launch",
        "deploy",
    ]

    for pattern in positive_patterns:
        if pattern in value:
            return "rollout"

    return None