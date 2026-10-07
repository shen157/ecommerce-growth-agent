from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool

from app.config import get_llm

from app.schemas.models import (
    SpecialistReport,
)

from app.tools.experiment_tools import (
    list_experiments,
    summarize_experiment,
    evaluate_conversion,
    evaluate_business_impact,
    evaluate_experiment,
)

from app.agents.report_utils import (
    ensure_specialist_report,
)

# =========================================================
# Tools
# =========================================================
@tool
def available_experiments() -> list:
    """
    List available A/B experiment IDs.

    Use only when the request does not identify an experiment and discovering
    valid IDs is necessary. Do not call it when an experiment ID is already
    present.
    """

    return list_experiments()


@tool
def experiment_summary(
    experiment_id: str,
) -> dict:
    """
    Return descriptive summary statistics for one experiment.

    Use this as the single tool only for a descriptive overview or sample
    summary. Do not use it for significance or rollout decisions.
    """

    return summarize_experiment(
        experiment_id
    )


@tool
def conversion_significance(
    experiment_id: str,
    alpha: float = 0.05,
) -> dict:
    """
    Test the conversion-rate difference between control and treatment.

    Use this as the single tool for conversion significance, lift, p-value,
    or conversion confidence-interval questions that do not ask about rollout
    or business impact.
    """

    return evaluate_conversion(
        experiment_id=experiment_id,
        alpha=alpha,
    )


@tool
def business_impact(
    experiment_id: str,
    alpha: float = 0.05,
) -> dict:
    """
    Evaluate experiment revenue and profit impact.

    Use this as the single tool for revenue, profit, or business-impact
    questions that do not ask for an overall rollout decision.
    """

    return evaluate_business_impact(
        experiment_id=experiment_id,
        alpha=alpha,
    )


@tool
def full_experiment_evaluation(
    experiment_id: str,
    alpha: float = 0.05,
) -> dict:
    """
    Return a complete experiment evaluation for a rollout decision.

    Use this as the single preferred tool for rollout, ship/no-ship, launch,
    go/no-go, or general experiment evaluation questions. It already includes
    conversion and business-impact evidence, so do not also call
    experiment_summary, conversion_significance, or business_impact.
    """

    return evaluate_experiment(
        experiment_id=experiment_id,
        alpha=alpha,
    )


# =========================================================
# Prompt
# =========================================================
SYSTEM_PROMPT = """
You are the Experiment Analyst Agent in a multi-agent
e-commerce growth decision system.

You analyze A/B tests, including conversion, statistical significance,
revenue, profit, business significance, and rollout decisions.

Evidence rules:

1. Use experiment tools for every quantitative claim. Never invent KPI
   values, lift, p-values, confidence intervals, or significance.
2. Report only numerical values explicitly returned by tools. Do not perform
   new arithmetic, including sums, averages, ratios, percentages,
   differences, or other derived metrics.
3. Statistical significance alone does not justify rollout; distinguish it
   from business significance and consider profit for rollout decisions.
4. Higher conversion does not automatically mean rollout.
5. Never access experiment ground-truth files. Base recommendations only on
   observable experiment data.
6. State uncertainty when evidence is inconclusive.

Tool-selection and stopping rules:

1. Call exactly one analytical tool when the experiment ID is known:
   - rollout, launch, go/no-go, or complete evaluation
     -> full_experiment_evaluation
   - conversion significance, lift, p-value, or conversion CI only
     -> conversion_significance
   - revenue, profit, or business impact only -> business_impact
   - descriptive overview or sample summary only -> experiment_summary
2. full_experiment_evaluation is comprehensive. Never combine it with
   experiment_summary, conversion_significance, or business_impact.
3. Call available_experiments only when no experiment ID is provided and ID
   discovery is necessary. After discovery, call only the one analytical tool
   that matches the request.
4. Never call the same tool more than once in a run. Do not call tools merely
   to explore, confirm, or add unrelated context.
5. Once the selected tool result is sufficient, stop calling tools and
   immediately produce the structured SpecialistReport.

Your final report must be grounded in tool outputs.
"""


# =========================================================
# Agent
# =========================================================
def create_experiment_agent():

    llm = get_llm()

    return create_agent(
        model=llm,

        tools=[
            full_experiment_evaluation,
            conversion_significance,
            business_impact,
            experiment_summary,
            available_experiments,
        ],

        system_prompt=SYSTEM_PROMPT,

        response_format=ToolStrategy(
            SpecialistReport
        ),
    )


def run_experiment_agent(
    query: str,
) -> SpecialistReport:

    agent = create_experiment_agent()

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
        agent_name="experiment_agent",
    )


if __name__ == "__main__":

    report = run_experiment_agent(
        "Should we roll out EXP001?"
    )

    print(
        report.model_dump_json(
            indent=2
        )
    )
