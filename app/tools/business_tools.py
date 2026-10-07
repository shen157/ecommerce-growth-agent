"""Deterministic business KPI tools for the ecommerce agents."""

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
    if pd.isna(value):
        return None
    numeric = float(value)
    if numeric.is_integer():
        return int(numeric)
    return round(numeric, digits)


def _validate_date_range(
    start_date: str | None, end_date: str | None
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    try:
        start = pd.Timestamp(start_date).normalize() if start_date else None
        end = pd.Timestamp(end_date).normalize() if end_date else None
    except (TypeError, ValueError) as exc:
        raise ValueError("Dates must be valid YYYY-MM-DD values") from exc
    if start is not None and end is not None and start > end:
        raise ValueError("start_date must be on or before end_date")
    return start, end


def _filter_daily_metrics(
    data: pd.DataFrame, start_date: str | None, end_date: str | None
) -> tuple[pd.DataFrame, pd.Timestamp | None, pd.Timestamp | None]:
    _require_columns(
        data,
        {"purchase_date", "orders", "payment_value"},
        "daily business metrics",
    )
    start, end = _validate_date_range(start_date, end_date)
    result = data.copy()
    result["purchase_date"] = pd.to_datetime(result["purchase_date"], errors="coerce")
    result = result.dropna(subset=["purchase_date"])
    if start is not None:
        result = result.loc[result["purchase_date"].ge(start)]
    if end is not None:
        result = result.loc[result["purchase_date"].le(end)]
    return result, start, end


def analyze_revenue_trend(
    frequency: Literal["daily", "weekly", "monthly"] = "monthly",
    start_date: str | None = None,
    end_date: str | None = None,
    daily_metrics: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Aggregate payment revenue and order volume into a time series."""

    if frequency not in {"daily", "weekly", "monthly"}:
        raise ValueError("frequency must be one of: daily, weekly, monthly")
    source = (
        _load_processed_table("daily_business_metrics.csv")
        if daily_metrics is None
        else daily_metrics.copy()
    )
    data, _, _ = _filter_daily_metrics(source, start_date, end_date)
    if data.empty:
        return {
            "analysis": "revenue_trend",
            "frequency": frequency,
            "total_revenue": 0,
            "total_orders": 0,
            "overall_aov": None,
            "periods": [],
        }

    if frequency == "daily":
        data["period"] = data["purchase_date"].dt.strftime("%Y-%m-%d")
    elif frequency == "weekly":
        data["period"] = (
            data["purchase_date"].dt.to_period("W-SUN").apply(lambda p: p.start_time)
        ).dt.strftime("%Y-%m-%d")
    else:
        data["period"] = data["purchase_date"].dt.to_period("M").astype(str)

    grouped = (
        data.groupby("period", as_index=False)
        .agg(revenue=("payment_value", "sum"), orders=("orders", "sum"))
        .sort_values("period")
    )
    grouped["aov"] = grouped["revenue"].div(grouped["orders"].replace(0, pd.NA))
    grouped["revenue_growth"] = grouped["revenue"].pct_change(fill_method=None)

    periods = [
        {
            "period": str(row.period),
            "revenue": _number(row.revenue),
            "orders": int(row.orders),
            "aov": _number(row.aov),
            "revenue_growth": _number(row.revenue_growth),
        }
        for row in grouped.itertuples(index=False)
    ]
    total_revenue = grouped["revenue"].sum()
    total_orders = grouped["orders"].sum()
    return {
        "analysis": "revenue_trend",
        "frequency": frequency,
        "total_revenue": _number(total_revenue),
        "total_orders": int(total_orders),
        "overall_aov": _number(
            total_revenue / total_orders if total_orders else None
        ),
        "periods": periods,
    }


def _rank_categories(
    direction: Literal["top", "bottom"],
    top_n: int,
    start_month: str | None,
    end_month: str | None,
    category_metrics: pd.DataFrame | None,
) -> dict[str, Any]:
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError("top_n must be a positive integer")
    data = (
        _load_processed_table("category_monthly_metrics.csv")
        if category_metrics is None
        else category_metrics.copy()
    )
    required = {
        "purchase_month",
        "category",
        "orders",
        "customers",
        "items_sold",
        "merchandise_revenue",
    }
    _require_columns(data, required, f"get_{direction}_categories")
    data["purchase_month"] = data["purchase_month"].astype(str)
    if start_month is not None:
        try:
            normalized_start = str(pd.Period(start_month, freq="M"))
        except ValueError as exc:
            raise ValueError("start_month must be a valid YYYY-MM value") from exc
        data = data.loc[data["purchase_month"].ge(normalized_start)]
    if end_month is not None:
        try:
            normalized_end = str(pd.Period(end_month, freq="M"))
        except ValueError as exc:
            raise ValueError("end_month must be a valid YYYY-MM value") from exc
        data = data.loc[data["purchase_month"].le(normalized_end)]
    if start_month and end_month and normalized_start > normalized_end:
        raise ValueError("start_month must be on or before end_month")

    grouped = (
        data.groupby("category", as_index=False)
        .agg(
            merchandise_revenue=("merchandise_revenue", "sum"),
            orders=("orders", "sum"),
            customer_count_sum=("customers", "sum"),
            items_sold=("items_sold", "sum"),
        )
    )
    ascending = direction == "bottom"
    grouped = grouped.sort_values(
        ["merchandise_revenue", "orders", "category"],
        ascending=[ascending, ascending, True],
    ).head(top_n)
    total_revenue = data["merchandise_revenue"].sum()
    categories = [
        {
            "rank": rank,
            "category": row.category,
            "merchandise_revenue": _number(row.merchandise_revenue),
            "revenue_share": _number(
                row.merchandise_revenue / total_revenue if total_revenue else None
            ),
            "orders": int(row.orders),
            "customer_count_sum": int(row.customer_count_sum),
            "items_sold": int(row.items_sold),
            "revenue_per_order": _number(
                row.merchandise_revenue / row.orders if row.orders else None
            ),
        }
        for rank, row in enumerate(grouped.itertuples(index=False), start=1)
    ]
    return {
        "analysis": f"{direction}_categories",
        "ranking_metric": "merchandise_revenue",
        "start_month": start_month,
        "end_month": end_month,
        "categories": categories,
    }


def get_top_categories(
    top_n: int = 10,
    start_month: str | None = None,
    end_month: str | None = None,
    category_metrics: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Rank categories by highest merchandise revenue."""

    return _rank_categories(
        "top", top_n, start_month, end_month, category_metrics
    )


def get_bottom_categories(
    top_n: int = 10,
    start_month: str | None = None,
    end_month: str | None = None,
    category_metrics: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Rank categories by lowest merchandise revenue."""

    return _rank_categories(
        "bottom", top_n, start_month, end_month, category_metrics
    )


def get_mom_declining_categories(
    target_month: str | None = None,
    top_n: int = 10,
    min_decline_pct: float = 0.0,
    category_metrics: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Find category revenue declines between two adjacent calendar months.

    A category present in the previous month but absent in the target month is
    treated as a 100% decline. Categories with no previous-month revenue are
    excluded because their growth rate is undefined.
    """

    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError("top_n must be a positive integer")
    if min_decline_pct < 0:
        raise ValueError("min_decline_pct must be non-negative")
    data = (
        _load_processed_table("category_monthly_metrics.csv")
        if category_metrics is None
        else category_metrics.copy()
    )
    _require_columns(
        data,
        {"purchase_month", "category", "merchandise_revenue"},
        "get_mom_declining_categories",
    )
    if data.empty:
        return {
            "analysis": "mom_declining_categories",
            "target_month": target_month,
            "previous_month": None,
            "categories": [],
        }
    try:
        periods = pd.PeriodIndex(data["purchase_month"].astype(str), freq="M")
        target = pd.Period(target_month, freq="M") if target_month else periods.max()
    except ValueError as exc:
        raise ValueError("target_month must be a valid YYYY-MM value") from exc
    previous = target - 1
    monthly = (
        data.assign(_period=periods)
        .groupby(["_period", "category"], as_index=False)["merchandise_revenue"]
        .sum()
    )
    previous_values = monthly.loc[monthly["_period"].eq(previous)].set_index(
        "category"
    )["merchandise_revenue"]
    current_values = monthly.loc[monthly["_period"].eq(target)].set_index(
        "category"
    )["merchandise_revenue"]

    comparison = pd.DataFrame({"previous_revenue": previous_values})
    comparison = comparison.loc[comparison["previous_revenue"].gt(0)]
    comparison["current_revenue"] = current_values.reindex(comparison.index).fillna(0)
    comparison["mom_growth"] = (
        comparison["current_revenue"] / comparison["previous_revenue"] - 1
    )
    comparison["revenue_change"] = (
        comparison["current_revenue"] - comparison["previous_revenue"]
    )
    threshold = min_decline_pct / 100
    if threshold == 0:
        comparison = comparison.loc[comparison["mom_growth"].lt(0)]
    else:
        comparison = comparison.loc[comparison["mom_growth"].le(-threshold)]
    # Most negative percentage decline first; lost revenue breaks ties.
    comparison = comparison.sort_values(
        ["mom_growth", "revenue_change"], ascending=[True, True]
    ).head(top_n)

    categories = [
        {
            "rank": rank,
            "category": str(category),
            "previous_revenue": _number(row.previous_revenue),
            "current_revenue": _number(row.current_revenue),
            "revenue_change": _number(row.revenue_change),
            "mom_growth": _number(row.mom_growth),
            "decline_pct": _number(-row.mom_growth * 100),
        }
        for rank, (category, row) in enumerate(comparison.iterrows(), start=1)
    ]
    return {
        "analysis": "mom_declining_categories",
        "target_month": str(target),
        "previous_month": str(previous),
        "categories": categories,
    }


def calculate_aov(
    start_date: str | None = None,
    end_date: str | None = None,
    compare_previous: bool = False,
    daily_metrics: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Calculate weighted average order value (payment revenue / orders)."""

    source = (
        _load_processed_table("daily_business_metrics.csv")
        if daily_metrics is None
        else daily_metrics.copy()
    )
    current, start, end = _filter_daily_metrics(source, start_date, end_date)
    revenue = current["payment_value"].sum()
    orders = current["orders"].sum()
    aov = revenue / orders if orders else None
    result: dict[str, Any] = {
        "analysis": "average_order_value",
        "start_date": start.strftime("%Y-%m-%d") if start is not None else None,
        "end_date": end.strftime("%Y-%m-%d") if end is not None else None,
        "orders": int(orders),
        "revenue": _number(revenue),
        "aov": _number(aov),
        "previous_period": None,
        "aov_change": None,
    }

    if compare_previous:
        if start is None or end is None:
            raise ValueError(
                "start_date and end_date are required when compare_previous=True"
            )
        period_days = (end - start).days + 1
        previous_end = start - pd.Timedelta(days=1)
        previous_start = previous_end - pd.Timedelta(days=period_days - 1)
        previous, _, _ = _filter_daily_metrics(
            source,
            previous_start.strftime("%Y-%m-%d"),
            previous_end.strftime("%Y-%m-%d"),
        )
        previous_revenue = previous["payment_value"].sum()
        previous_orders = previous["orders"].sum()
        previous_aov = (
            previous_revenue / previous_orders if previous_orders else None
        )
        change = (
            aov / previous_aov - 1
            if aov is not None and previous_aov not in {None, 0}
            else None
        )
        result["previous_period"] = {
            "start_date": previous_start.strftime("%Y-%m-%d"),
            "end_date": previous_end.strftime("%Y-%m-%d"),
            "orders": int(previous_orders),
            "revenue": _number(previous_revenue),
            "aov": _number(previous_aov),
        }
        result["aov_change"] = _number(change)
    return result


__all__ = [
    "analyze_revenue_trend",
    "get_top_categories",
    "get_bottom_categories",
    "get_mom_declining_categories",
    "calculate_aov",
]
