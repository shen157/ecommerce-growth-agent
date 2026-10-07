from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool

from app.config import get_llm

from app.schemas.models import (
    SpecialistReport,
)

from app.tools.business_tools import (
    analyze_revenue_trend,
    get_top_categories,
    get_bottom_categories,
    get_mom_declining_categories,
    calculate_aov,
)

from app.agents.report_utils import (
    ensure_specialist_report,
)

# =========================================================
# Tools
# =========================================================
@tool
def revenue_trend(
    frequency: str = "monthly",
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """
    Analyze revenue and order trends over time.

    Use this as the single tool for overall business trajectory, revenue over
    time, peaks, declines, or a time-window trend. Do not add category or AOV
    tools unless the question explicitly asks for those separate analyses.

    frequency must be one of: daily, weekly, monthly.
    """

    if frequency not in {
        "daily",
        "weekly",
        "monthly",
    }:
        raise ValueError(
            "Invalid frequency."
        )

    return analyze_revenue_trend(
        frequency=frequency,
        start_date=start_date,
        end_date=end_date,
    )


@tool
def top_categories(
    top_n: int = 10,
    start_month: str | None = None,
    end_month: str | None = None,
) -> dict:
    """
    Return the highest-revenue categories for an optional month range.

    Use this as the single tool for best, leading, highest-revenue, or
    strongest category questions. Do not also call bottom_categories.
    """

    return get_top_categories(
        top_n=top_n,
        start_month=start_month,
        end_month=end_month,
    )


@tool
def bottom_categories(
    top_n: int = 10,
    start_month: str | None = None,
    end_month: str | None = None,
) -> dict:
    """
    Return the lowest-revenue categories for an optional month range.

    Use this as the single tool for weakest, lowest-revenue, or bottom
    category questions. Do not use it for month-over-month decline, which
    requires declining_categories.
    """

    return get_bottom_categories(
        top_n=top_n,
        start_month=start_month,
        end_month=end_month,
    )


@tool
def declining_categories(
    target_month: str | None = None,
    top_n: int = 10,
    min_decline_pct: float = 0.0,
) -> dict:
    """
    Find the largest month-over-month category revenue declines.

    Use this as the single tool for category decline, deterioration, or
    month-over-month category performance questions. It already performs the
    required period comparison, so do not call top_categories or
    bottom_categories to confirm its result.
    """

    return get_mom_declining_categories(
        target_month=target_month,
        top_n=top_n,
        min_decline_pct=min_decline_pct,
    )


@tool
def average_order_value(
    start_date: str | None = None,
    end_date: str | None = None,
    compare_previous: bool = False,
) -> dict:
    """
    Calculate AOV and optionally compare it with the prior equal-length period.

    Use this as the single tool only for AOV or order-value questions.
    compare_previous=True requires explicit start_date and end_date values;
    never infer or invent them.
    """

    try:

        return calculate_aov(
            start_date=start_date,
            end_date=end_date,
            compare_previous=compare_previous,
        )

    except ValueError as exc:

        return {
            "tool_error":
                str(exc),

            "recoverable":
                True,

            "instruction": (
                "Correct the tool arguments or choose "
                "a more appropriate business tool. "
                "Do not invent missing dates."
            ),
        }


# =========================================================
# Prompt
# =========================================================
SYSTEM_PROMPT = """
You are the Business Analyst Agent in a multi-agent
e-commerce growth decision system.

Your responsibility is business performance analysis:

- revenue, orders, and customers
- AOV
- category performance
- month-over-month growth and business trends

Evidence rules:

1. Use analytical tools for every quantitative claim. Never invent KPI
   values.
2. Report only numerical values explicitly returned by tools. Do not perform
   new arithmetic, including sums, averages, ratios, percentages,
   differences, or other derived metrics.
3. Clearly state comparison periods.
4. Avoid unsupported causal conclusions.
5. Do not analyze randomized experiments.
6. Provide concise, decision-relevant findings and numerical evidence.

Tool-selection and stopping rules:

1. Call exactly one most relevant analytical tool by default:
   - overall revenue/order trajectory or business trend -> revenue_trend
   - strongest or highest-revenue categories -> top_categories
   - weakest or lowest-revenue categories -> bottom_categories
   - month-over-month category declines -> declining_categories
   - AOV or order value -> average_order_value
2. Call a second tool only when the request explicitly combines two distinct
   analyses that cannot be answered by the first tool. Never use more than
   two analytical tool calls for one request.
3. Never call the same tool more than once in a run. Choose its arguments
   correctly on the first call.
4. Do not call tools merely to explore, confirm another tool's result, or add
   unrelated business context.
5. Once the selected tool results are sufficient, stop calling tools and
   immediately produce the structured SpecialistReport.

Argument and recovery rules:

- Use compare_previous=True only when BOTH start_date and end_date are
  explicitly available from the user's request or trusted context.
- If the user does not specify a date range, do not invent dates and do not
  request a previous-period comparison.
- For broad category financial-performance questions, use the single ranking
  or decline tool that directly matches the requested direction.
- If a tool returns a recoverable validation error, make at most one corrected
  call using valid, non-invented arguments. Do not explore other tools unless
  they directly answer the original question.

Your final report must be grounded in tool outputs.
"""


# =========================================================
# Agent
# =========================================================
def create_business_agent():

    llm = get_llm()

    return create_agent(
        model=llm,

        tools=[
            revenue_trend,
            top_categories,
            bottom_categories,
            declining_categories,
            average_order_value,
        ],

        system_prompt=SYSTEM_PROMPT,

        response_format=ToolStrategy(
            SpecialistReport
        ),
    )


def run_business_agent(
    query: str,
) -> SpecialistReport:

    agent = create_business_agent()

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query,
                }
            ]
        }
    )

    return ensure_specialist_report(
        result=result,
        agent_name="business_agent",
    )


if __name__ == "__main__":

    report = run_business_agent(
        "Which categories experienced "
        "the largest month-over-month "
        "revenue decline in 2018-06?"
    )

    print(
        report.model_dump_json(
            indent=2
        )
    )
