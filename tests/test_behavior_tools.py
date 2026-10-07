import json

import pandas as pd
import pytest

from app.tools.behavior_tools import (
    analyze_delivery_review_relationship,
    get_category_preferences,
    get_repeat_customer_profile,
)


@pytest.fixture
def customer_features():
    return pd.DataFrame(
        {
            "customer_unique_id": ["c1", "c2", "c3"],
            "order_count": [3, 1, 1],
            "total_spend": [300.0, 80.0, 120.0],
            "average_order_value": [100.0, 80.0, 120.0],
            "total_items": [5, 1, 2],
            "average_review_score": [4.5, 3.0, 5.0],
            "average_delivery_days": [5.0, 10.0, 4.0],
            "late_delivery_rate": [0.0, 1.0, 0.0],
            "freight_ratio": [0.1, 0.2, 0.15],
        }
    )


def test_repeat_customer_profile_is_customer_level_and_serializable(customer_features):
    result = get_repeat_customer_profile(customer_features)

    assert result["customer_count"] == 3
    assert result["repeat_customer_count"] == 1
    assert result["repeat_customer_rate"] == pytest.approx(1 / 3, abs=1e-4)
    repeat = next(row for row in result["segments"] if row["segment"] == "repeat")
    one_time = next(
        row for row in result["segments"] if row["segment"] == "one_time"
    )
    assert repeat["average_total_spend"] == 300
    assert one_time["average_order_value"] == 100
    json.dumps(result)


def test_delivery_review_relationship_separates_late_orders():
    orders = pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3", "o4", "cancelled"],
            "order_status": ["delivered"] * 4 + ["canceled"],
            "review_score": [5, 4, 2, 1, 1],
            "delivery_time_days": [2.0, 4.0, 10.0, 15.0, 30.0],
            "delivery_delay_days": [-2.0, 0.0, 2.0, 10.0, 20.0],
        }
    )

    result = analyze_delivery_review_relationship(orders)

    assert result["orders_analyzed"] == 4
    assert result["on_time"]["orders"] == 2
    assert result["late"]["orders"] == 2
    assert result["on_time"]["average_review_score"] == 4.5
    assert result["late"]["average_review_score"] == 1.5
    assert result["delay_days_review_correlation"] < 0


def test_category_preferences_filters_repeat_customers(customer_features):
    facts = pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3", "o4"],
            "order_status": ["delivered", "delivered", "delivered", "canceled"],
            "customer_unique_id": ["c1", "c1", "c2", "c1"],
            "product_id": ["p1", "p2", "p3", "p4"],
            "category": ["books", "books", "toys", "toys"],
            "price": [10.0, 15.0, 100.0, 200.0],
        }
    )

    result = get_category_preferences(
        segment="repeat",
        top_n=2,
        order_facts=facts,
        customer_features=customer_features,
    )

    assert result["orders_analyzed"] == 2
    assert result["customers_analyzed"] == 1
    assert result["categories"][0]["category"] == "books"
    assert result["categories"][0]["order_share"] == 1


@pytest.mark.parametrize("top_n", [0, -1, 1.5, True])
def test_category_preferences_rejects_invalid_top_n(top_n):
    facts = pd.DataFrame(
        columns=["order_id", "customer_unique_id", "product_id", "category", "price"]
    )
    with pytest.raises(ValueError, match="top_n"):
        get_category_preferences(top_n=top_n, order_facts=facts)
