import json

import pandas as pd
import pytest

from app.tools.business_tools import (
    analyze_revenue_trend,
    calculate_aov,
    get_bottom_categories,
    get_mom_declining_categories,
    get_top_categories,
)


@pytest.fixture
def daily_metrics():
    return pd.DataFrame(
        {
            "purchase_date": ["2024-01-01", "2024-01-02", "2024-02-01"],
            "orders": [2, 3, 5],
            "payment_value": [100.0, 300.0, 1000.0],
        }
    )


@pytest.fixture
def category_metrics():
    return pd.DataFrame(
        {
            "purchase_month": [
                "2024-01",
                "2024-01",
                "2024-01",
                "2024-02",
                "2024-02",
            ],
            "category": ["books", "toys", "garden", "books", "toys"],
            "orders": [10, 5, 2, 8, 10],
            "customers": [9, 5, 2, 8, 9],
            "items_sold": [12, 6, 2, 9, 11],
            "merchandise_revenue": [1000.0, 500.0, 100.0, 800.0, 1000.0],
        }
    )


def test_monthly_revenue_trend_uses_weighted_aov(daily_metrics):
    result = analyze_revenue_trend("monthly", daily_metrics=daily_metrics)

    assert result["total_revenue"] == 1400
    assert result["total_orders"] == 10
    assert result["overall_aov"] == 140
    assert result["periods"][0] == {
        "period": "2024-01",
        "revenue": 400,
        "orders": 5,
        "aov": 80,
        "revenue_growth": None,
    }
    assert result["periods"][1]["revenue_growth"] == 1.5
    json.dumps(result)


def test_revenue_trend_respects_date_range(daily_metrics):
    result = analyze_revenue_trend(
        "daily",
        start_date="2024-01-02",
        end_date="2024-01-02",
        daily_metrics=daily_metrics,
    )
    assert result["total_revenue"] == 300
    assert result["periods"][0]["period"] == "2024-01-02"


def test_weekly_revenue_trend_labels_week_by_monday(daily_metrics):
    result = analyze_revenue_trend("weekly", daily_metrics=daily_metrics)

    assert result["periods"][0]["period"] == "2024-01-01"


def test_top_and_bottom_category_rankings(category_metrics):
    top = get_top_categories(top_n=2, category_metrics=category_metrics)
    bottom = get_bottom_categories(top_n=1, category_metrics=category_metrics)

    assert [row["category"] for row in top["categories"]] == ["books", "toys"]
    assert top["categories"][0]["merchandise_revenue"] == 1800
    assert bottom["categories"][0]["category"] == "garden"


def test_mom_declines_use_adjacent_months_and_include_disappeared_categories(
    category_metrics,
):
    result = get_mom_declining_categories(
        target_month="2024-02", top_n=5, category_metrics=category_metrics
    )

    assert result["previous_month"] == "2024-01"
    garden = next(row for row in result["categories"] if row["category"] == "garden")
    books = next(row for row in result["categories"] if row["category"] == "books")
    assert garden["decline_pct"] == 100
    assert books["decline_pct"] == 20
    assert all(row["category"] != "toys" for row in result["categories"])


def test_mom_declines_excludes_flat_categories_by_default(category_metrics):
    extra = pd.DataFrame(
        {
            "purchase_month": ["2024-01", "2024-02"],
            "category": ["flat", "flat"],
            "orders": [1, 1],
            "customers": [1, 1],
            "items_sold": [1, 1],
            "merchandise_revenue": [50.0, 50.0],
        }
    )
    data = pd.concat([category_metrics, extra], ignore_index=True)

    result = get_mom_declining_categories(
        target_month="2024-02", top_n=10, category_metrics=data
    )

    assert all(row["category"] != "flat" for row in result["categories"])


def test_aov_and_equal_length_previous_period_comparison(daily_metrics):
    result = calculate_aov(
        start_date="2024-01-02",
        end_date="2024-01-02",
        compare_previous=True,
        daily_metrics=daily_metrics,
    )

    assert result["aov"] == 100
    assert result["previous_period"]["aov"] == 50
    assert result["aov_change"] == 1


def test_aov_comparison_requires_both_dates(daily_metrics):
    with pytest.raises(ValueError, match="required"):
        calculate_aov(
            start_date="2024-01-01",
            compare_previous=True,
            daily_metrics=daily_metrics,
        )


def test_invalid_date_range_is_rejected(daily_metrics):
    with pytest.raises(ValueError, match="start_date"):
        analyze_revenue_trend(
            start_date="2024-02-01",
            end_date="2024-01-01",
            daily_metrics=daily_metrics,
        )
