from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from scipy.stats import norm
from scipy.stats import ttest_ind


# =========================================================
# Paths
# =========================================================
PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

EXPERIMENT_PATH = (
    PROJECT_ROOT
    /
    "data"
    /
    "synthetic"
    /
    "experiments.csv"
)


# =========================================================
# Data Loading
# =========================================================
def load_experiment_data():
    """
    Load semi-synthetic experiment data.
    """

    if not EXPERIMENT_PATH.exists():

        raise FileNotFoundError(
            "experiments.csv not found. "
            "Run scripts/build_experiments.py first."
        )

    df = pd.read_csv(
        EXPERIMENT_PATH
    )

    df["exposure_date"] = pd.to_datetime(
        df["exposure_date"],
        errors="coerce"
    )

    return df


# =========================================================
# Helpers
# =========================================================
def _get_experiment(
    experiment_id: str
):
    """
    Return data belonging to one experiment.
    """

    df = load_experiment_data()

    experiment = df[
        df["experiment_id"]
        ==
        experiment_id
    ].copy()

    if experiment.empty:

        available = sorted(
            df[
                "experiment_id"
            ].unique()
        )

        raise ValueError(
            f"Unknown experiment_id: "
            f"{experiment_id}. "
            f"Available: {available}"
        )

    return experiment


# =========================================================
# List Experiments
# =========================================================
def list_experiments():
    """
    Return all available experiments.
    """

    df = load_experiment_data()

    result = (
        df[
            [
                "experiment_id",
                "experiment_name"
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "experiment_id"
        )
    )

    return result.to_dict(
        orient="records"
    )


# =========================================================
# Experiment Summary
# =========================================================
def summarize_experiment(
    experiment_id: str
):
    """
    Return high-level performance for control
    and treatment groups.
    """

    df = _get_experiment(
        experiment_id
    )

    rows = []

    for variant in [
        "control",
        "treatment"
    ]:

        group = df[
            df["variant"]
            ==
            variant
        ]

        converted = group[
            group["converted"] == 1
        ]

        rows.append(
            {
                "variant":
                    variant,

                "users":
                    int(len(group)),

                "conversions":
                    int(
                        group[
                            "converted"
                        ].sum()
                    ),

                "conversion_rate":
                    float(
                        group[
                            "converted"
                        ].mean()
                    ),

                "total_revenue":
                    float(
                        group[
                            "net_revenue"
                        ].sum()
                    ),

                "revenue_per_user":
                    float(
                        group[
                            "net_revenue"
                        ].mean()
                    ),

                "average_order_value":
                    (
                        float(
                            converted[
                                "net_revenue"
                            ].mean()
                        )
                        if len(converted) > 0
                        else 0.0
                    ),

                "total_profit":
                    float(
                        group[
                            "profit"
                        ].sum()
                    ),

                "profit_per_user":
                    float(
                        group[
                            "profit"
                        ].mean()
                    ),
            }
        )

    return {
        "experiment_id":
            experiment_id,

        "experiment_name":
            df[
                "experiment_name"
            ].iloc[0],

        "variants":
            rows
    }


# =========================================================
# Conversion Test
# =========================================================
def evaluate_conversion(
    experiment_id: str,
    alpha: float = 0.05
):
    """
    Two-proportion z-test for conversion rate.

    H0:
        conversion_control
        =
        conversion_treatment

    H1:
        conversion rates differ.
    """

    df = _get_experiment(
        experiment_id
    )

    control = df[
        df["variant"]
        ==
        "control"
    ]

    treatment = df[
        df["variant"]
        ==
        "treatment"
    ]

    n_control = len(control)
    n_treatment = len(treatment)

    x_control = int(
        control["converted"].sum()
    )

    x_treatment = int(
        treatment["converted"].sum()
    )

    p_control = (
        x_control
        /
        n_control
    )

    p_treatment = (
        x_treatment
        /
        n_treatment
    )

    absolute_lift = (
        p_treatment
        -
        p_control
    )

    if p_control > 0:

        relative_lift = (
            absolute_lift
            /
            p_control
        )

    else:

        relative_lift = None

    # =========================================================
    # Pooled proportion
    # =========================================================
    pooled_p = (
        x_control
        +
        x_treatment
    ) / (
        n_control
        +
        n_treatment
    )

    pooled_se = np.sqrt(
        pooled_p
        *
        (1 - pooled_p)
        *
        (
            (1 / n_control)
            +
            (1 / n_treatment)
        )
    )

    if pooled_se == 0:

        z_score = 0.0
        p_value = 1.0

    else:

        z_score = (
            absolute_lift
            /
            pooled_se
        )

        p_value = (
            2
            *
            (
                1
                -
                norm.cdf(
                    abs(z_score)
                )
            )
        )

    # =========================================================
    # 95% CI for difference
    # =========================================================
    difference_se = np.sqrt(
        (
            p_control
            *
            (1 - p_control)
            /
            n_control
        )
        +
        (
            p_treatment
            *
            (1 - p_treatment)
            /
            n_treatment
        )
    )

    critical_value = norm.ppf(
        1
        -
        alpha / 2
    )

    ci_low = (
        absolute_lift
        -
        critical_value
        *
        difference_se
    )

    ci_high = (
        absolute_lift
        +
        critical_value
        *
        difference_se
    )

    statistically_significant = (
        p_value < alpha
    )

    return {
        "experiment_id":
            experiment_id,

        "metric":
            "conversion_rate",

        "control_users":
            int(n_control),

        "treatment_users":
            int(n_treatment),

        "control_conversions":
            int(x_control),

        "treatment_conversions":
            int(x_treatment),

        "control_conversion_rate":
            round(
                float(p_control),
                6
            ),

        "treatment_conversion_rate":
            round(
                float(p_treatment),
                6
            ),

        "absolute_lift":
            round(
                float(absolute_lift),
                6
            ),

        "relative_lift":
            (
                round(
                    float(relative_lift),
                    6
                )
                if relative_lift
                is not None
                else None
            ),

        "z_score":
            round(
                float(z_score),
                4
            ),

        "p_value":
            round(
                float(p_value),
                6
            ),

        "confidence_interval":
            [
                round(
                    float(ci_low),
                    6
                ),

                round(
                    float(ci_high),
                    6
                )
            ],

        "alpha":
            alpha,

        "statistically_significant":
            bool(
                statistically_significant
            )
    }


# =========================================================
# Business Impact
# =========================================================
def evaluate_business_impact(
    experiment_id: str,
    alpha: float = 0.05
):
    """
    Evaluate revenue and profit impact.

    Profit is evaluated per exposed user,
    not only among converted customers.

    This prevents a high-conversion experiment from
    automatically being considered successful.
    """

    df = _get_experiment(
        experiment_id
    )

    control = df[
        df["variant"]
        ==
        "control"
    ]

    treatment = df[
        df["variant"]
        ==
        "treatment"
    ]

    # =========================================================
    # Revenue
    # =========================================================
    control_revenue = (
        control[
            "net_revenue"
        ]
        .to_numpy()
    )

    treatment_revenue = (
        treatment[
            "net_revenue"
        ]
        .to_numpy()
    )

    revenue_test = ttest_ind(
        treatment_revenue,
        control_revenue,
        equal_var=False,
        nan_policy="omit"
    )

    control_revenue_per_user = float(
        np.mean(
            control_revenue
        )
    )

    treatment_revenue_per_user = float(
        np.mean(
            treatment_revenue
        )
    )

    revenue_difference = (
        treatment_revenue_per_user
        -
        control_revenue_per_user
    )

    # =========================================================
    # Profit
    # =========================================================
    control_profit = (
        control[
            "profit"
        ]
        .to_numpy()
    )

    treatment_profit = (
        treatment[
            "profit"
        ]
        .to_numpy()
    )

    profit_test = ttest_ind(
        treatment_profit,
        control_profit,
        equal_var=False,
        nan_policy="omit"
    )

    control_profit_per_user = float(
        np.mean(
            control_profit
        )
    )

    treatment_profit_per_user = float(
        np.mean(
            treatment_profit
        )
    )

    profit_difference = (
        treatment_profit_per_user
        -
        control_profit_per_user
    )

    # =========================================================
    # AOV among converters
    # =========================================================
    control_converted = control[
        control["converted"] == 1
    ]

    treatment_converted = treatment[
        treatment["converted"] == 1
    ]

    control_aov = float(
        control_converted[
            "net_revenue"
        ].mean()
    )

    treatment_aov = float(
        treatment_converted[
            "net_revenue"
        ].mean()
    )

    # =========================================================
    # Result
    # =========================================================
    return {
        "experiment_id":
            experiment_id,

        "revenue": {
            "control_per_user":
                round(
                    control_revenue_per_user,
                    4
                ),

            "treatment_per_user":
                round(
                    treatment_revenue_per_user,
                    4
                ),

            "difference":
                round(
                    revenue_difference,
                    4
                ),

            "p_value":
                round(
                    float(
                        revenue_test.pvalue
                    ),
                    6
                ),

            "statistically_significant":
                bool(
                    revenue_test.pvalue
                    <
                    alpha
                )
        },

        "profit": {
            "control_per_user":
                round(
                    control_profit_per_user,
                    4
                ),

            "treatment_per_user":
                round(
                    treatment_profit_per_user,
                    4
                ),

            "difference":
                round(
                    profit_difference,
                    4
                ),

            "p_value":
                round(
                    float(
                        profit_test.pvalue
                    ),
                    6
                ),

            "statistically_significant":
                bool(
                    profit_test.pvalue
                    <
                    alpha
                )
        },

        "average_order_value": {
            "control":
                round(
                    control_aov,
                    4
                ),

            "treatment":
                round(
                    treatment_aov,
                    4
                ),

            "difference":
                round(
                    treatment_aov
                    -
                    control_aov,
                    4
                )
        }
    }


# =========================================================
# Full Experiment Evaluation
# =========================================================
def evaluate_experiment(
    experiment_id: str,
    alpha: float = 0.05
):
    """
    Full experiment evaluation.

    Combines:
        1. Conversion significance
        2. Revenue impact
        3. Profit impact
        4. Rollout recommendation
    """

    conversion = (
        evaluate_conversion(
            experiment_id,
            alpha
        )
    )

    business = (
        evaluate_business_impact(
            experiment_id,
            alpha
        )
    )

    conversion_significant = (
        conversion[
            "statistically_significant"
        ]
    )

    conversion_lift = (
        conversion[
            "absolute_lift"
        ]
    )

    profit_difference = (
        business[
            "profit"
        ][
            "difference"
        ]
    )

    profit_significant = (
        business[
            "profit"
        ][
            "statistically_significant"
        ]
    )

    # =========================================================
    # Decision Logic
    # =========================================================
    if (
        conversion_significant
        and conversion_lift > 0
        and profit_difference > 0
    ):

        recommendation = "rollout"

        reason = (
            "Treatment significantly improves "
            "conversion and also increases "
            "profit per exposed user."
        )

    elif (
        conversion_significant
        and conversion_lift > 0
        and profit_difference <= 0
    ):

        recommendation = (
            "do_not_rollout"
        )

        reason = (
            "Treatment improves conversion, "
            "but the improvement does not "
            "translate into higher profit."
        )

    elif not conversion_significant:

        recommendation = (
            "do_not_rollout"
        )

        reason = (
            "The observed conversion difference "
            "is not statistically significant."
        )

    else:

        recommendation = "investigate"

        reason = (
            "Experiment results are mixed and "
            "require additional analysis."
        )

    return {
        "experiment_id":
            experiment_id,

        "conversion_analysis":
            conversion,

        "business_analysis":
            business,

        "recommendation":
            recommendation,

        "reason":
            reason,

        "diagnostics": {
            "conversion_significant":
                bool(
                    conversion_significant
                ),

            "profit_significant":
                bool(
                    profit_significant
                ),

            "conversion_lift":
                conversion_lift,

            "profit_difference":
                profit_difference
        }
    }
