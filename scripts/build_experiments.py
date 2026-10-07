from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# Config
# =========================================================
RANDOM_SEED = 42
USERS_PER_EXPERIMENT = 12000

rng = np.random.default_rng(RANDOM_SEED)


# =========================================================
# Paths
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"

SYNTHETIC_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CUSTOMER_FEATURES_PATH = (
    PROCESSED_DIR / "customer_features.csv"
)


# =========================================================
# Experiment configuration
# =========================================================
EXPERIMENT_CONFIGS = [
    {
        "experiment_id": "EXP001",
        "experiment_name": "Deep Discount Campaign",
        "exposure_date": "2018-06-01",

        "control_conversion_rate": 0.080,
        "treatment_conversion_rate": 0.105,

        "control_discount_rate": 0.02,
        "treatment_discount_rate": 0.20,

        "gross_margin_rate": 0.30,

        "control_marketing_cost": 0.50,
        "treatment_marketing_cost": 1.00,

        "expected_conversion_effect": "positive",
        "expected_profit_effect": "negative",
        "expected_recommendation": "do_not_rollout",
    },

    {
        "experiment_id": "EXP002",
        "experiment_name": "Personalized Recommendation",
        "exposure_date": "2018-07-01",

        "control_conversion_rate": 0.070,
        "treatment_conversion_rate": 0.095,

        "control_discount_rate": 0.02,
        "treatment_discount_rate": 0.05,

        "gross_margin_rate": 0.38,

        "control_marketing_cost": 0.40,
        "treatment_marketing_cost": 0.60,

        "expected_conversion_effect": "positive",
        "expected_profit_effect": "positive",
        "expected_recommendation": "rollout",
    },

    {
        "experiment_id": "EXP003",
        "experiment_name": "Minor UI Checkout Change",
        "exposure_date": "2018-08-01",

        "control_conversion_rate": 0.085,
        "treatment_conversion_rate": 0.089,

        "control_discount_rate": 0.02,
        "treatment_discount_rate": 0.02,

        "gross_margin_rate": 0.34,

        "control_marketing_cost": 0.50,
        "treatment_marketing_cost": 0.50,

        "expected_conversion_effect": "neutral",
        "expected_profit_effect": "neutral",
        "expected_recommendation": "do_not_rollout",
    },
]


# =========================================================
# Load customers
# =========================================================
def load_customer_pool():
    """
    Load real Olist customer features.

    We use real customer AOV distribution as the base for
    semi-synthetic experiment simulation.
    """

    customers = pd.read_csv(
        CUSTOMER_FEATURES_PATH
    )

    required_columns = [
        "customer_unique_id",
        "average_order_value"
    ]

    missing = [
        col
        for col in required_columns
        if col not in customers.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    median_aov = (
        customers["average_order_value"]
        .median()
    )

    customers["average_order_value"] = (
        customers["average_order_value"]
        .fillna(median_aov)
        .clip(
            lower=20,
            upper=500
        )
    )

    # Randomize user order once.
    customers = customers.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    return customers


# =========================================================
# Generate one experiment
# =========================================================
def generate_single_experiment(
    customers,
    config
):
    """
    Generate one controlled A/B experiment.

    Grain:
        One row = one exposed customer.
    """

    experiment_id = config["experiment_id"]

    n = len(customers)

    if n % 2 != 0:
        raise ValueError(
            "Experiment sample size must be even."
        )

    # =========================================================
    # Random assignment
    # =========================================================
    variants = np.array(
        ["control"] * (n // 2)
        +
        ["treatment"] * (n // 2)
    )

    rng.shuffle(variants)

    experiment = customers[
        [
            "customer_unique_id",
            "average_order_value"
        ]
    ].copy()

    experiment["experiment_id"] = (
        experiment_id
    )

    experiment["experiment_name"] = (
        config["experiment_name"]
    )

    experiment["exposure_date"] = pd.to_datetime(
        config["exposure_date"]
    )

    experiment["variant"] = variants

    # =========================================================
    # Conversion probability
    # =========================================================
    experiment["conversion_probability"] = np.where(
        experiment["variant"] == "control",
        config["control_conversion_rate"],
        config["treatment_conversion_rate"]
    )

    experiment["converted"] = (
        rng.random(n)
        <
        experiment["conversion_probability"]
    ).astype(int)

    # =========================================================
    # Discount
    # =========================================================
    experiment["discount_rate"] = np.where(
        experiment["variant"] == "control",
        config["control_discount_rate"],
        config["treatment_discount_rate"]
    )

    # =========================================================
    # Marketing cost
    # Cost is incurred for every exposed user.
    # =========================================================
    experiment["marketing_cost"] = np.where(
        experiment["variant"] == "control",
        config["control_marketing_cost"],
        config["treatment_marketing_cost"]
    )

    # =========================================================
    # Potential order value
    #
    # Real Olist customer AOV is used as baseline.
    # A log-normal noise term introduces realistic variance.
    # =========================================================
    order_multiplier = rng.lognormal(
        mean=0.0,
        sigma=0.25,
        size=n
    )

    potential_order_value = (
        experiment["average_order_value"]
        *
        order_multiplier
    )

    potential_order_value = (
        potential_order_value
        .clip(
            lower=10,
            upper=800
        )
    )

    # Only converted users generate an order.
    experiment["gross_order_value"] = np.where(
        experiment["converted"] == 1,
        potential_order_value,
        0.0
    )

    # =========================================================
    # Product cost
    #
    # Product cost is calculated from the value BEFORE
    # discount.
    # =========================================================
    margin_rate = config[
        "gross_margin_rate"
    ]

    experiment["product_cost"] = (
        experiment["gross_order_value"]
        *
        (1 - margin_rate)
    )

    # =========================================================
    # Revenue after discount
    # =========================================================
    experiment["net_revenue"] = (
        experiment["gross_order_value"]
        *
        (
            1
            -
            experiment["discount_rate"]
        )
    )

    # =========================================================
    # Gross profit
    # =========================================================
    experiment["gross_profit"] = (
        experiment["net_revenue"]
        -
        experiment["product_cost"]
    )

    # =========================================================
    # Final profit
    #
    # Marketing cost applies even if customer did not buy.
    # =========================================================
    experiment["profit"] = (
        experiment["gross_profit"]
        -
        experiment["marketing_cost"]
    )

    # =========================================================
    # Round
    # =========================================================
    numeric_columns = [
        "average_order_value",
        "conversion_probability",
        "discount_rate",
        "marketing_cost",
        "gross_order_value",
        "product_cost",
        "net_revenue",
        "gross_profit",
        "profit"
    ]

    experiment[numeric_columns] = (
        experiment[numeric_columns]
        .round(4)
    )

    return experiment


# =========================================================
# Build Ground Truth Table
# =========================================================
def build_ground_truth():
    """
    Store known experiment design outcomes.

    IMPORTANT:
    This file is for evaluation only.
    Agents should NOT use this table when making decisions.
    """

    rows = []

    for config in EXPERIMENT_CONFIGS:

        rows.append(
            {
                "experiment_id":
                    config["experiment_id"],

                "experiment_name":
                    config["experiment_name"],

                "expected_conversion_effect":
                    config[
                        "expected_conversion_effect"
                    ],

                "expected_profit_effect":
                    config[
                        "expected_profit_effect"
                    ],

                "expected_recommendation":
                    config[
                        "expected_recommendation"
                    ],
            }
        )

    return pd.DataFrame(rows)


# =========================================================
# Validation
# =========================================================
def validate_experiments(
    experiments
):
    """
    Basic experiment integrity checks.
    """

    assert experiments[
        "experiment_id"
    ].notna().all()

    assert experiments[
        "customer_unique_id"
    ].notna().all()

    assert set(
        experiments["variant"].unique()
    ) == {
        "control",
        "treatment"
    }

    assert experiments[
        "converted"
    ].isin([0, 1]).all()

    # Ensure each experiment is balanced.
    for experiment_id, group in (
        experiments.groupby(
            "experiment_id"
        )
    ):

        counts = (
            group["variant"]
            .value_counts()
        )

        assert (
            counts["control"]
            ==
            counts["treatment"]
        ), (
            f"{experiment_id}: "
            "control/treatment imbalance"
        )

    print(
        "\nExperiment validation: PASS"
    )


# =========================================================
# Main
# =========================================================
def main():

    print("=" * 70)
    print(
        "Generating Semi-Synthetic "
        "Experiment Layer"
    )
    print("=" * 70)

    customers = load_customer_pool()

    required_users = (
        USERS_PER_EXPERIMENT
        *
        len(EXPERIMENT_CONFIGS)
    )

    if len(customers) < required_users:

        raise ValueError(
            f"Need {required_users} customers, "
            f"but only {len(customers)} available."
        )

    all_experiments = []

    start = 0

    for config in EXPERIMENT_CONFIGS:

        end = (
            start
            +
            USERS_PER_EXPERIMENT
        )

        experiment_customers = (
            customers.iloc[start:end]
            .copy()
        )

        experiment = (
            generate_single_experiment(
                experiment_customers,
                config
            )
        )

        all_experiments.append(
            experiment
        )

        print(
            f"Generated "
            f"{config['experiment_id']} "
            f"| users={len(experiment)}"
        )

        start = end

    experiments = pd.concat(
        all_experiments,
        ignore_index=True
    )

    validate_experiments(
        experiments
    )

    # =========================================================
    # Save experiment data
    # =========================================================
    experiment_path = (
        SYNTHETIC_DIR
        /
        "experiments.csv"
    )

    experiments.to_csv(
        experiment_path,
        index=False
    )

    # =========================================================
    # Save Ground Truth
    # =========================================================
    ground_truth = (
        build_ground_truth()
    )

    ground_truth_path = (
        SYNTHETIC_DIR
        /
        "experiment_ground_truth.csv"
    )

    ground_truth.to_csv(
        ground_truth_path,
        index=False
    )

    print("\nSaved:")
    print(
        f"{experiment_path.name}: "
        f"{experiments.shape}"
    )

    print(
        f"{ground_truth_path.name}: "
        f"{ground_truth.shape}"
    )

    # =========================================================
    # Quick sanity check
    # =========================================================
    print(
        "\nObserved Conversion Rates:"
    )

    summary = (
        experiments
        .groupby(
            [
                "experiment_id",
                "variant"
            ]
        )
        .agg(
            users=(
                "customer_unique_id",
                "count"
            ),

            conversion_rate=(
                "converted",
                "mean"
            ),

            revenue_per_user=(
                "net_revenue",
                "mean"
            ),

            profit_per_user=(
                "profit",
                "mean"
            )
        )
        .round(4)
    )

    print(summary)


if __name__ == "__main__":
    main()