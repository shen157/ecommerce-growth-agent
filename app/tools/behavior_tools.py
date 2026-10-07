"""Deterministic customer-behaviour analytics tools.

The public functions in this module return JSON-serialisable dictionaries so
they can be exposed to an LLM agent without leaking pandas objects.  Each
function can either load the project's processed data or receive a DataFrame,
which keeps the tools easy to unit test and reuse in notebooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _load_processed_table(filename: str) -> pd.DataFrame:
    path = PROCESSED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Processed table not found: {path}. Run the data build scripts first."
        )
    return pd.read_csv(path)


def _require_columns(data: pd.DataFrame, columns: set[str], tool_name: str) -> None:
    missing = sorted(columns.difference(data.columns))
    if missing:
        raise ValueError(f"{tool_name} requires columns: {', '.join(missing)}")


def _number(value: Any, digits: int = 4) -> int | float | None:
    """Convert numpy/pandas numbers to stable JSON-compatible Python values."""

    if pd.isna(value):
        return None
    numeric = float(value)
    if numeric.is_integer():
        return int(numeric)
    return round(numeric, digits)


def get_repeat_customer_profile(
    customer_features: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Compare repeat customers with one-time customers.

    A repeat customer is a customer with more than one delivered order.  The
    returned segment metrics are customer-level averages, preventing large
    customers from silently dominating the comparison.
    """

    data = (
        _load_processed_table("customer_features.csv")
        if customer_features is None
        else customer_features.copy()
    )
    required = {
        "customer_unique_id",
        "order_count",
        "total_spend",
        "average_order_value",
        "total_items",
        "average_review_score",
        "average_delivery_days",
        "late_delivery_rate",
        "freight_ratio",
    }
    _require_columns(data, required, "get_repeat_customer_profile")
    if data.empty:
        return {
            "analysis": "repeat_customer_profile",
            "customer_count": 0,
            "repeat_customer_count": 0,
            "repeat_customer_rate": None,
            "segments": [],
        }

    # Derive from order_count instead of trusting a possibly stale flag column.
    data["segment"] = data["order_count"].gt(1).map(
        {True: "repeat", False: "one_time"}
    )
    segment_order = ["repeat", "one_time"]
    segments: list[dict[str, Any]] = []
    for segment in segment_order:
        subset = data.loc[data["segment"].eq(segment)]
        if subset.empty:
            continue
        segments.append(
            {
                "segment": segment,
                "customers": int(subset["customer_unique_id"].nunique()),
                "average_orders": _number(subset["order_count"].mean()),
                "average_total_spend": _number(subset["total_spend"].mean()),
                "average_order_value": _number(
                    subset["average_order_value"].mean()
                ),
                "average_items": _number(subset["total_items"].mean()),
                "average_review_score": _number(
                    subset["average_review_score"].mean()
                ),
                "average_delivery_days": _number(
                    subset["average_delivery_days"].mean()
                ),
                "average_late_delivery_rate": _number(
                    subset["late_delivery_rate"].mean()
                ),
                "average_freight_ratio": _number(subset["freight_ratio"].mean()),
            }
        )

    customer_count = int(data["customer_unique_id"].nunique())
    repeat_count = int(
        data.loc[data["order_count"].gt(1), "customer_unique_id"].nunique()
    )
    return {
        "analysis": "repeat_customer_profile",
        "customer_count": customer_count,
        "repeat_customer_count": repeat_count,
        "repeat_customer_rate": _number(repeat_count / customer_count),
        "segments": segments,
    }


def analyze_delivery_review_relationship(
    order_summary: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Measure how delivery speed and lateness relate to review scores."""

    data = (
        _load_processed_table("order_summary.csv")
        if order_summary is None
        else order_summary.copy()
    )
    required = {
        "order_id",
        "review_score",
        "delivery_time_days",
        "delivery_delay_days",
    }
    _require_columns(data, required, "analyze_delivery_review_relationship")
    if "order_status" in data.columns:
        data = data.loc[data["order_status"].eq("delivered")].copy()

    valid = data.dropna(
        subset=["review_score", "delivery_time_days", "delivery_delay_days"]
    ).copy()
    if valid.empty:
        return {
            "analysis": "delivery_review_relationship",
            "orders_analyzed": 0,
            "delivery_days_review_correlation": None,
            "delay_days_review_correlation": None,
            "on_time": None,
            "late": None,
            "delay_buckets": [],
        }

    valid["late"] = valid["delivery_delay_days"].gt(0)
    valid["delay_bucket"] = pd.cut(
        valid["delivery_delay_days"],
        bins=[float("-inf"), 0, 3, 7, float("inf")],
        labels=["early_or_on_time", "late_1_3_days", "late_4_7_days", "late_over_7_days"],
        include_lowest=True,
    )

    def correlation(x: pd.Series, y: pd.Series) -> float | None:
        if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
            return None
        return _number(x.corr(y))

    def group_summary(subset: pd.DataFrame) -> dict[str, Any] | None:
        if subset.empty:
            return None
        return {
            "orders": int(subset["order_id"].nunique()),
            "average_review_score": _number(subset["review_score"].mean()),
            "low_review_rate": _number(subset["review_score"].le(2).mean()),
            "average_delivery_days": _number(
                subset["delivery_time_days"].mean()
            ),
            "average_delay_days": _number(subset["delivery_delay_days"].mean()),
        }

    buckets: list[dict[str, Any]] = []
    for label in valid["delay_bucket"].cat.categories:
        subset = valid.loc[valid["delay_bucket"].eq(label)]
        summary = group_summary(subset)
        if summary is not None:
            buckets.append({"bucket": str(label), **summary})

    return {
        "analysis": "delivery_review_relationship",
        "orders_analyzed": int(valid["order_id"].nunique()),
        "delivery_days_review_correlation": correlation(
            valid["delivery_time_days"], valid["review_score"]
        ),
        "delay_days_review_correlation": correlation(
            valid["delivery_delay_days"], valid["review_score"]
        ),
        "on_time": group_summary(valid.loc[~valid["late"]]),
        "late": group_summary(valid.loc[valid["late"]]),
        "delay_buckets": buckets,
    }


def get_category_preferences(
    segment: Literal["all", "repeat", "one_time"] = "all",
    top_n: int = 10,
    order_facts: pd.DataFrame | None = None,
    customer_features: pd.DataFrame | None = None,
    include_unknown: bool = False,
) -> dict[str, Any]:
    """Return the most preferred categories for a customer segment.

    Preference is ranked by distinct delivered orders, with merchandise
    revenue as the deterministic tie-breaker.
    """

    if segment not in {"all", "repeat", "one_time"}:
        raise ValueError("segment must be one of: all, repeat, one_time")
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError("top_n must be a positive integer")

    facts = (
        _load_processed_table("order_facts.csv")
        if order_facts is None
        else order_facts.copy()
    )
    required = {
        "order_id",
        "customer_unique_id",
        "product_id",
        "category",
        "price",
    }
    _require_columns(facts, required, "get_category_preferences")
    if "order_status" in facts.columns:
        facts = facts.loc[facts["order_status"].eq("delivered")].copy()

    if segment != "all":
        customers = (
            _load_processed_table("customer_features.csv")
            if customer_features is None
            else customer_features.copy()
        )
        _require_columns(
            customers,
            {"customer_unique_id", "order_count"},
            "get_category_preferences",
        )
        target_repeat = segment == "repeat"
        customer_ids = customers.loc[
            customers["order_count"].gt(1).eq(target_repeat), "customer_unique_id"
        ]
        facts = facts.loc[facts["customer_unique_id"].isin(customer_ids)].copy()

    facts["category"] = facts["category"].fillna("unknown").astype(str)
    if not include_unknown:
        facts = facts.loc[facts["category"].ne("unknown")]

    if facts.empty:
        return {
            "analysis": "category_preferences",
            "segment": segment,
            "orders_analyzed": 0,
            "customers_analyzed": 0,
            "categories": [],
        }

    total_orders = int(facts["order_id"].nunique())
    grouped = (
        facts.groupby("category", as_index=False)
        .agg(
            orders=("order_id", "nunique"),
            customers=("customer_unique_id", "nunique"),
            items=("product_id", "count"),
            merchandise_revenue=("price", "sum"),
            average_item_price=("price", "mean"),
        )
        .sort_values(
            ["orders", "merchandise_revenue", "category"],
            ascending=[False, False, True],
        )
        .head(top_n)
    )

    categories = [
        {
            "rank": rank,
            "category": row.category,
            "orders": int(row.orders),
            "customers": int(row.customers),
            "items": int(row.items),
            "merchandise_revenue": _number(row.merchandise_revenue),
            "average_item_price": _number(row.average_item_price),
            "order_share": _number(row.orders / total_orders),
        }
        for rank, row in enumerate(grouped.itertuples(index=False), start=1)
    ]
    return {
        "analysis": "category_preferences",
        "segment": segment,
        "orders_analyzed": total_orders,
        "customers_analyzed": int(facts["customer_unique_id"].nunique()),
        "categories": categories,
    }


__all__ = [
    "get_repeat_customer_profile",
    "analyze_delivery_review_relationship",
    "get_category_preferences",
]
