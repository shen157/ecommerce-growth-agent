from app.tools.experiment_tools import (
    list_experiments,
    evaluate_conversion,
    evaluate_business_impact,
    evaluate_experiment,
)


def test_experiments_exist():

    experiments = list_experiments()

    ids = {
        item["experiment_id"]
        for item in experiments
    }

    assert "EXP001" in ids
    assert "EXP002" in ids
    assert "EXP003" in ids


def test_exp001_conversion_improves():

    result = evaluate_conversion(
        "EXP001"
    )

    assert (
        result[
            "treatment_conversion_rate"
        ]
        >
        result[
            "control_conversion_rate"
        ]
    )

    assert (
        result[
            "statistically_significant"
        ]
        is True
    )


def test_exp001_profit_declines():

    result = evaluate_business_impact(
        "EXP001"
    )

    assert (
        result[
            "profit"
        ][
            "difference"
        ]
        <
        0
    )


def test_exp001_should_not_rollout():

    result = evaluate_experiment(
        "EXP001"
    )

    assert (
        result[
            "recommendation"
        ]
        ==
        "do_not_rollout"
    )


def test_exp002_should_rollout():

    result = evaluate_experiment(
        "EXP002"
    )

    assert (
        result[
            "conversion_analysis"
        ][
            "absolute_lift"
        ]
        >
        0
    )

    assert (
        result[
            "business_analysis"
        ][
            "profit"
        ][
            "difference"
        ]
        >
        0
    )

    assert (
        result[
            "recommendation"
        ]
        ==
        "rollout"
    )


def test_exp003_not_significant():

    result = evaluate_conversion(
        "EXP003"
    )

    assert (
        result[
            "statistically_significant"
        ]
        is False
    )