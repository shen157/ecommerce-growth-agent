from pathlib import Path

import pandas as pd


# =========================================================
# Paths
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "olist"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Load data
# =========================================================
def load_data(filename):
    path = RAW_DIR / filename
    return pd.read_csv(path)


# =========================================================
# Main
# =========================================================
def main():

    print("=" * 70)
    print("Building Analytics Tables")
    print("=" * 70)

    customers = load_data(
        "olist_customers_dataset.csv"
    )

    orders = load_data(
        "olist_orders_dataset.csv"
    )

    order_items = load_data(
        "olist_order_items_dataset.csv"
    )

    products = load_data(
        "olist_products_dataset.csv"
    )

    payments = load_data(
        "olist_order_payments_dataset.csv"
    )

    reviews = load_data(
        "olist_order_reviews_dataset.csv"
    )

    translations = load_data(
        "product_category_name_translation.csv"
    )

    # =========================================================
    # Convert timestamps
    # =========================================================
    date_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    for col in date_columns:
        orders[col] = pd.to_datetime(
            orders[col],
            errors="coerce"
        )

    # =========================================================
    # Product categories
    # =========================================================
    products = products.merge(
        translations,
        on="product_category_name",
        how="left"
    )

    products["category"] = (
        products["product_category_name_english"]
        .fillna(products["product_category_name"])
        .fillna("unknown")
    )

    # =========================================================
    # Reviews: order level
    # =========================================================
    reviews_agg = (
        reviews
        .groupby("order_id", as_index=False)
        .agg(
            review_score=("review_score", "mean"),
            review_count=("review_id", "count")
        )
    )

    # =========================================================
    # Payments: order level
    # =========================================================
    payments_agg = (
        payments
        .groupby("order_id", as_index=False)
        .agg(
            payment_value=("payment_value", "sum"),
            payment_installments=(
                "payment_installments",
                "max"
            ),
            payment_methods=(
                "payment_type",
                lambda x: ",".join(
                    sorted(
                        set(
                            x.dropna().astype(str)
                        )
                    )
                )
            )
        )
    )

    # =========================================================
    # Items: order level
    # =========================================================
    items_agg = (
        order_items
        .groupby("order_id", as_index=False)
        .agg(
            item_count=("order_item_id", "count"),
            unique_products=("product_id", "nunique"),
            merchandise_value=("price", "sum"),
            freight_value=("freight_value", "sum")
        )
    )

    # =========================================================
    # TABLE 1: ORDER FACTS
    # Grain = one order item
    # =========================================================
    product_columns = [
        "product_id",
        "category",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm"
    ]

    order_facts = (
        order_items
        .merge(
            products[product_columns],
            on="product_id",
            how="left"
        )
        .merge(
            orders[
                [
                    "order_id",
                    "customer_id",
                    "order_status",
                    "order_purchase_timestamp"
                ]
            ],
            on="order_id",
            how="left"
        )
        .merge(
            customers[
                [
                    "customer_id",
                    "customer_unique_id",
                    "customer_city",
                    "customer_state"
                ]
            ],
            on="customer_id",
            how="left"
        )
        .merge(
            reviews_agg[
                [
                    "order_id",
                    "review_score"
                ]
            ],
            on="order_id",
            how="left"
        )
    )

    order_facts["purchase_date"] = (
        order_facts["order_purchase_timestamp"]
        .dt.date
    )

    order_facts["purchase_year"] = (
        order_facts["order_purchase_timestamp"]
        .dt.year
    )

    order_facts["purchase_month"] = (
        order_facts["order_purchase_timestamp"]
        .dt.to_period("M")
        .astype(str)
    )

    # =========================================================
    # TABLE 2: ORDER SUMMARY
    # Grain = one order
    # =========================================================
    order_summary = (
        orders
        .merge(
            customers,
            on="customer_id",
            how="left"
        )
        .merge(
            items_agg,
            on="order_id",
            how="left"
        )
        .merge(
            payments_agg,
            on="order_id",
            how="left"
        )
        .merge(
            reviews_agg,
            on="order_id",
            how="left"
        )
    )

    # =========================================================
    # Derived features
    # =========================================================
    order_summary["purchase_date"] = (
        order_summary["order_purchase_timestamp"]
        .dt.date
    )

    order_summary["purchase_month"] = (
        order_summary["order_purchase_timestamp"]
        .dt.to_period("M")
        .astype(str)
    )

    order_summary["order_total_value"] = (
        order_summary["merchandise_value"].fillna(0)
        +
        order_summary["freight_value"].fillna(0)
    )

    # Purchase -> delivery
    order_summary["delivery_time_days"] = (
        (
            order_summary[
                "order_delivered_customer_date"
            ]
            -
            order_summary[
                "order_purchase_timestamp"
            ]
        )
        .dt.total_seconds()
        / 86400
    )

    # Actual delivery - estimated delivery
    order_summary["delivery_delay_days"] = (
        (
            order_summary[
                "order_delivered_customer_date"
            ]
            -
            order_summary[
                "order_estimated_delivery_date"
            ]
        )
        .dt.total_seconds()
        / 86400
    )

    order_summary["late_delivery"] = (
        order_summary["delivery_delay_days"] > 0
    )

    # Useful quality check
    order_summary["payment_gap"] = (
        order_summary["payment_value"]
        -
        order_summary["order_total_value"]
    )

    # =========================================================
    # Save
    # =========================================================
    order_facts_path = (
        PROCESSED_DIR / "order_facts.csv"
    )

    order_summary_path = (
        PROCESSED_DIR / "order_summary.csv"
    )

    order_facts.to_csv(
        order_facts_path,
        index=False
    )

    order_summary.to_csv(
        order_summary_path,
        index=False
    )

    # =========================================================
    # Summary
    # =========================================================
    print("\nSaved:")
    print(
        f"order_facts.csv: {order_facts.shape}"
    )

    print(
        f"order_summary.csv: {order_summary.shape}"
    )

    print("\nOrder Facts Preview:")
    print(
        order_facts[
            [
                "order_id",
                "customer_unique_id",
                "product_id",
                "category",
                "price",
                "freight_value",
                "review_score"
            ]
        ].head()
    )

    print("\nOrder Summary Preview:")
    print(
        order_summary[
            [
                "order_id",
                "customer_unique_id",
                "order_status",
                "item_count",
                "merchandise_value",
                "freight_value",
                "payment_value",
                "review_score"
            ]
        ].head()
    )


if __name__ == "__main__":
    main()