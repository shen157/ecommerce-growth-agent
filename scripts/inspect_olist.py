from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "olist"

FILES = {
    "customers": "olist_customers_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

def inspect_dataset(name, filename):
    path = DATA_DIR / filename
    df = pd.read_csv(path)

    print("\n" + "=" * 70)
    print(f"Dataset: {name}")
    print("=" * 70)

    print(f"Shape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    missing = df.isna().sum()
    missing = missing[missing > 0]

    print("\nMissing values:")
    if len(missing) == 0:
        print("No missing values")
    else:
        print(missing.sort_values(ascending=False))

    print("\nFirst 3 rows:")
    print(df.head(3))

    return df


def main():

    datasets = {}

    for name, filename in FILES.items():
        datasets[name] = inspect_dataset(
            name,
            filename
        )

    customers = datasets["customers"]
    orders = datasets["orders"]
    order_items = datasets["order_items"]
    products = datasets["products"]

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"]
    )

    print("\n" + "=" * 70)
    print("Core Dataset Summary")
    print("=" * 70)

    print(
        "Unique customers:",
        customers["customer_unique_id"].nunique()
    )

    print(
        "Unique orders:",
        orders["order_id"].nunique()
    )

    print(
        "Unique products:",
        products["product_id"].nunique()
    )

    print(
        "Order item rows:",
        len(order_items)
    )

    print(
        "Order date range:",
        orders["order_purchase_timestamp"].min(),
        "->",
        orders["order_purchase_timestamp"].max()
    )


if __name__ == "__main__":
    main()