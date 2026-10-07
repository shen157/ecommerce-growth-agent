from pathlib import Path

import pandas as pd
import numpy as np
import pandas as pd

# =========================================================
# Paths
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ORDER_SUMMARY_PATH = PROCESSED_DIR / "order_summary.csv"
ORDER_FACTS_PATH = PROCESSED_DIR / "order_facts.csv"


# =========================================================
# Load Data
# =========================================================
def load_data():
    """
    Load processed order-level and item-level analytical tables.
    """

    order_summary = pd.read_csv(
        ORDER_SUMMARY_PATH
    )

    order_facts = pd.read_csv(
        ORDER_FACTS_PATH
    )

    # Convert timestamps
    order_summary["order_purchase_timestamp"] = pd.to_datetime(
        order_summary["order_purchase_timestamp"],
        errors="coerce"
    )

    order_facts["order_purchase_timestamp"] = pd.to_datetime(
        order_facts["order_purchase_timestamp"],
        errors="coerce"
    )

    return order_summary, order_facts


# =========================================================
# Customer Features
# =========================================================
def build_customer_features(order_summary):
    """
    Build one-row-per-customer feature table.

    Grain:
        One row = one unique customer.
    """

    # Use delivered orders for actual purchase behavior
    delivered = order_summary[
        order_summary["order_status"] == "delivered"
    ].copy()

    # Recalculate late-delivery flag to ensure boolean consistency
    delivered["late_delivery_flag"] = (
        delivered["delivery_delay_days"] > 0
    ).astype(int)

    customer_features = (
        delivered
        .groupby(
            "customer_unique_id",
            as_index=False
        )
        .agg(
            order_count=(
                "order_id",
                "nunique"
            ),

            total_spend=(
                "payment_value",
                "sum"
            ),

            average_order_value=(
                "payment_value",
                "mean"
            ),

            total_merchandise_value=(
                "merchandise_value",
                "sum"
            ),

            total_freight_value=(
                "freight_value",
                "sum"
            ),

            total_items=(
                "item_count",
                "sum"
            ),

            average_items_per_order=(
                "item_count",
                "mean"
            ),

            average_review_score=(
                "review_score",
                "mean"
            ),

            average_delivery_days=(
                "delivery_time_days",
                "mean"
            ),

            late_delivery_rate=(
                "late_delivery_flag",
                "mean"
            ),

            first_purchase=(
                "order_purchase_timestamp",
                "min"
            ),

            last_purchase=(
                "order_purchase_timestamp",
                "max"
            ),

            customer_state=(
                "customer_state",
                "first"
            )
        )
    )

    # Customer lifetime
    customer_features["customer_lifetime_days"] = (
        customer_features["last_purchase"]
        -
        customer_features["first_purchase"]
    ).dt.days

    # Repeat purchase flag
    customer_features["repeat_customer"] = (
        customer_features["order_count"] > 1
    ).astype(int)

    # Average item value
    customer_features["average_item_value"] = (
        customer_features["total_merchandise_value"]
        /
        customer_features["total_items"]
    )

    # Freight as percentage of total spending
    valid_spend = (
        customer_features["total_spend"]
        .replace(0, np.nan)
    )

    customer_features["freight_ratio"] = (
        customer_features["total_freight_value"]
        /
        valid_spend
    )

    customer_features["freight_ratio"] = (
        customer_features["freight_ratio"]
        .replace(
            [np.inf, -np.inf],
            np.nan
       )
    )

    # Round numerical columns
    numeric_columns = [
        "total_spend",
        "average_order_value",
        "total_merchandise_value",
        "total_freight_value",
        "average_items_per_order",
        "average_review_score",
        "average_delivery_days",
        "late_delivery_rate",
        "average_item_value",
        "freight_ratio"
    ]

    customer_features[numeric_columns] = (
        customer_features[numeric_columns]
        .round(4)
    )

    return customer_features


# =========================================================
# Daily Business Metrics
# =========================================================
def build_daily_business_metrics(order_summary):
    """
    Build daily business KPI table.

    Grain:
        One row = one calendar day.
    """

    delivered = order_summary[
        order_summary["order_status"] == "delivered"
    ].copy()

    delivered["purchase_date"] = (
        delivered["order_purchase_timestamp"]
        .dt.date
    )

    daily_metrics = (
        delivered
        .groupby(
            "purchase_date",
            as_index=False
        )
        .agg(
            orders=(
                "order_id",
                "nunique"
            ),

            customers=(
                "customer_unique_id",
                "nunique"
            ),

            merchandise_revenue=(
                "merchandise_value",
                "sum"
            ),

            freight_revenue=(
                "freight_value",
                "sum"
            ),

            payment_value=(
                "payment_value",
                "sum"
            ),

            items_sold=(
                "item_count",
                "sum"
            ),

            average_review_score=(
                "review_score",
                "mean"
            ),

            average_delivery_days=(
                "delivery_time_days",
                "mean"
            )
        )
    )

    # AOV
    daily_metrics["average_order_value"] = (
        daily_metrics["payment_value"]
        /
        daily_metrics["orders"]
    )

    # Items per order
    daily_metrics["items_per_order"] = (
        daily_metrics["items_sold"]
        /
        daily_metrics["orders"]
    )

    # Revenue per customer
    daily_metrics["revenue_per_customer"] = (
        daily_metrics["payment_value"]
        /
        daily_metrics["customers"]
    )

    numeric_columns = [
        "merchandise_revenue",
        "freight_revenue",
        "payment_value",
        "average_review_score",
        "average_delivery_days",
        "average_order_value",
        "items_per_order",
        "revenue_per_customer"
    ]

    daily_metrics[numeric_columns] = (
        daily_metrics[numeric_columns]
        .round(4)
    )

    return daily_metrics


# =========================================================
# Category Monthly Metrics
# =========================================================
def build_category_monthly_metrics(order_facts):
    """
    Build monthly category-level performance table.

    Grain:
        One row = one category in one month.
    """

    delivered = order_facts[
        order_facts["order_status"] == "delivered"
    ].copy()

    delivered["purchase_month"] = (
        delivered["order_purchase_timestamp"]
        .dt.to_period("M")
        .astype(str)
    )

    category_metrics = (
        delivered
        .groupby(
            [
                "purchase_month",
                "category"
            ],
            as_index=False
        )
        .agg(
            orders=(
                "order_id",
                "nunique"
            ),

            customers=(
                "customer_unique_id",
                "nunique"
            ),

            items_sold=(
                "product_id",
                "count"
            ),

            unique_products=(
                "product_id",
                "nunique"
            ),

            merchandise_revenue=(
                "price",
                "sum"
            ),

            freight_value=(
                "freight_value",
                "sum"
            ),

            average_item_price=(
                "price",
                "mean"
            ),

            average_review_score=(
                "review_score",
                "mean"
            )
        )
    )

    # Average revenue per order
    category_metrics["revenue_per_order"] = (
        category_metrics["merchandise_revenue"]
        /
        category_metrics["orders"]
    )

    # Average items per order
    category_metrics["items_per_order"] = (
        category_metrics["items_sold"]
        /
        category_metrics["orders"]
    )

    # Month-over-month revenue growth
    category_metrics = (
        category_metrics
        .sort_values(
            [
                "category",
                "purchase_month"
            ]
        )
    )

    category_metrics["revenue_mom_growth"] = (
        category_metrics
        .groupby("category")[
            "merchandise_revenue"
        ]
        .pct_change()
    )

    numeric_columns = [
        "merchandise_revenue",
        "freight_value",
        "average_item_price",
        "average_review_score",
        "revenue_per_order",
        "items_per_order",
        "revenue_mom_growth"
    ]

    category_metrics[numeric_columns] = (
        category_metrics[numeric_columns]
        .round(4)
    )

    return category_metrics


# =========================================================
# Validation
# =========================================================
def validate_tables(
    customer_features,
    daily_metrics,
    category_metrics
):
    """
    Validate the grain and uniqueness of generated tables.
    """

    assert (
        customer_features[
            "customer_unique_id"
        ].is_unique
    ), "Duplicate customer_unique_id detected."

    assert (
        daily_metrics[
            "purchase_date"
        ].is_unique
    ), "Duplicate purchase_date detected."

    assert not category_metrics.duplicated(
        subset=[
            "purchase_month",
            "category"
        ]
    ).any(), (
        "Duplicate category-month rows detected."
    )

    print("\nValidation: PASS")


# =========================================================
# Save
# =========================================================
def save_table(df, filename):

    output_path = PROCESSED_DIR / filename

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved: {filename} | shape={df.shape}"
    )


# =========================================================
# Main
# =========================================================
def main():

    print("=" * 70)
    print("Building Feature Tables")
    print("=" * 70)

    order_summary, order_facts = load_data()

    customer_features = build_customer_features(
        order_summary
    )

    daily_metrics = build_daily_business_metrics(
        order_summary
    )

    category_metrics = build_category_monthly_metrics(
        order_facts
    )

    validate_tables(
        customer_features,
        daily_metrics,
        category_metrics
    )

    print("\nSaving tables...")

    save_table(
        customer_features,
        "customer_features.csv"
    )

    save_table(
        daily_metrics,
        "daily_business_metrics.csv"
    )

    save_table(
        category_metrics,
        "category_monthly_metrics.csv"
    )

    # =========================================================
    # Quick business checks
    # =========================================================
    print("\n" + "=" * 70)
    print("Feature Table Summary")
    print("=" * 70)

    print(
        "\nCustomers:",
        len(customer_features)
    )

    print(
        "Repeat customers:",
        customer_features[
            "repeat_customer"
        ].sum()
    )

    repeat_rate = (
        customer_features[
            "repeat_customer"
        ].mean()
    )

    print(
        f"Repeat customer rate: "
        f"{repeat_rate:.2%}"
    )

    print(
        "\nDaily metrics rows:",
        len(daily_metrics)
    )

    print(
        "Category-month rows:",
        len(category_metrics)
    )

    print("\nCustomer Features Preview:")
    print(
        customer_features.head()
    )

    print("\nDaily Metrics Preview:")
    print(
        daily_metrics.head()
    )

    print("\nCategory Monthly Preview:")
    print(
        category_metrics.head()
    )


if __name__ == "__main__":
    main()