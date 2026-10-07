from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool

from app.config import get_llm

from app.schemas.models import (
    SpecialistReport,
)

from app.tools.behavior_tools import (
    get_repeat_customer_profile,
    analyze_delivery_review_relationship,
    get_category_preferences,
)

from app.agents.report_utils import (
    ensure_specialist_report,
)

# =========================================================
# Tools
# =========================================================
@tool
def repeat_customer_profile() -> dict:
    """
    Compare repeat customers with one-time customers.

    Use this as the single tool for questions about repeat purchasing,
    loyalty, customer value, AOV, order frequency, or general differences
    between repeat and one-time customers. Do not also call
    category_preferences unless the question explicitly asks which product
    categories a segment prefers.
    """

    return get_repeat_customer_profile()


@tool
def delivery_review_relationship() -> dict:
    """
    Analyze the relationship between delivery performance and review scores.

    Use this as the single tool for questions about late delivery, delivery
    performance, review scores, or the relationship between delivery and
    reviews. It is correlational evidence, not causal evidence.
    """

    return (
        analyze_delivery_review_relationship()
    )


@tool
def category_preferences(
    segment: str = "all",
    top_n: int = 10,
    include_unknown: bool = False,
) -> dict:
    """
    Analyze product-category preferences for one customer segment.

    Use this as the single tool when the question asks which categories are
    preferred by all, repeat, or one-time customers. Choose the segment from
    the request; do not call the tool again for another segment unless the
    user explicitly requests a cross-segment category comparison.

    segment must be one of: all, repeat, one_time.
    """

    if segment not in {
        "all",
        "repeat",
        "one_time",
    }:
        raise ValueError(
            "segment must be one of: "
            "all, repeat, one_time"
        )

    return get_category_preferences(
        segment=segment,
        top_n=top_n,
        include_unknown=include_unknown,
    )


# =========================================================
# Prompt
# =========================================================
SYSTEM_PROMPT = """
You are the Behavior Analyst Agent in a multi-agent
e-commerce growth decision system.

Your responsibility is customer behavior analysis:

- repeat purchasing and customer loyalty
- customer segments and category preferences
- delivery experience and customer reviews

Evidence rules:

1. Use tools for every quantitative claim. Never invent statistics.
2. Report only numerical values explicitly returned by tools. Do not perform
   new arithmetic, including sums, averages, ratios, percentages,
   differences, or other derived metrics.
3. Do not perform A/B experiment analysis.
4. Do not make unsupported causal claims; distinguish correlation from
   causation.
5. Include the important numerical evidence and clearly state limitations.
6. Keep conclusions concise and business-oriented.

Tool-selection and stopping rules:

1. Call exactly one most relevant analytical tool by default:
   - repeat/one-time, loyalty, value, frequency -> repeat_customer_profile
   - delivery or reviews -> delivery_review_relationship
   - category preference for a segment -> category_preferences
2. Call a second tool only when the request explicitly combines two distinct
   analyses that cannot be answered by the first tool. Never use more than
   two analytical tool calls for one request.
3. Never call the same tool more than once in a run. Choose its arguments
   correctly on the first call.
4. Do not call tools merely to explore, confirm, or add unrelated context.
5. Once the selected tool results are sufficient, stop calling tools and
   immediately produce the structured SpecialistReport.

Your final response must be grounded in tool results.
"""


# =========================================================
# Agent
# =========================================================
def create_behavior_agent():

    llm = get_llm()

    return create_agent(
        model=llm,

        tools=[
            repeat_customer_profile,
            delivery_review_relationship,
            category_preferences,
        ],

        system_prompt=SYSTEM_PROMPT,

        response_format=ToolStrategy(
            SpecialistReport
        ),
    )


def run_behavior_agent(
    query: str,
) -> SpecialistReport:

    agent = create_behavior_agent()

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
        agent_name="behavior_agent",
    )


if __name__ == "__main__":

    report = run_behavior_agent(
        "How are repeat customers different "
        "from one-time customers?"
    )

    print(
        report.model_dump_json(
            indent=2
        )
    )
