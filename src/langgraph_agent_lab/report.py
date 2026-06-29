"""Report generation helper."""

from __future__ import annotations

from pathlib import Path

from .metrics import MetricsReport


def render_report(metrics: MetricsReport) -> str:
    """Render a complete lab report from metrics data.

    Generate a report that includes:
    1. Metrics summary table (total scenarios, success rate, retries, interrupts)
    2. Per-scenario results table
    3. Architecture explanation (your graph design, state schema, reducers)
    4. Failure analysis (at least two failure modes you considered)
    5. Improvement plan

    Use reports/lab_report_template.md as your guide.

    Return: formatted markdown string
    """
    row_template = (
        "| {scenario} | {expected} | {actual} | {success} | "
        "{retries} | {interrupts} | {errors} |"
    )
    scenario_rows = "\n".join(
        row_template.format(
            scenario=item.scenario_id,
            expected=item.expected_route,
            actual=item.actual_route or "",
            success="yes" if item.success else "no",
            retries=item.retry_count,
            interrupts=item.interrupt_count,
            errors="<br>".join(item.errors) if item.errors else "",
        )
        for item in metrics.scenario_metrics
    )
    if not scenario_rows:
        scenario_rows = "| n/a | n/a | n/a | no | 0 | 0 | no scenarios were recorded |"

    architecture = (
        "The workflow is a typed LangGraph `StateGraph` for support-ticket orchestration. "
        "It starts with `intake`, uses `classify` to choose a supported route, and then "
        "fans into direct answering, tool execution, clarification, risky-action approval, "
        "retry, or dead-letter handling. Every route converges through `finalize` before "
        "`END`, so all paths leave an audit event trail."
    )
    routing_summary = (
        "The graph uses conditional edges after classification, tool evaluation, retry "
        "accounting, and approval. The retry loop is bounded by `attempt < max_attempts`; "
        "once the limit is reached the workflow emits a dead-letter response."
    )
    retry_failure = (
        "1. Retry or tool failure: transient tool failures are represented as `ERROR` "
        "tool results. `evaluate` marks them as `needs_retry`, `retry` increments "
        "`attempt`, and `route_after_retry` either returns to `tool` or moves to "
        "`dead_letter` when the limit is reached."
    )
    approval_failure = (
        "2. Risky action without approval: refund, delete, cancellation, and email-sending "
        "requests route to `risky_action` first. The workflow records a proposed action "
        "and must pass through `approval` before any tool execution. If approval is "
        "rejected, the graph asks for a safer alternative through `clarify`."
    )
    persistence = (
        "Each scenario uses a distinct `thread_id` in the LangGraph config. The checkpointer "
        "is injected at compile time, so the same graph works with memory, SQLite, or "
        "Postgres checkpoint backends. `resume_success` is set from state-history "
        "availability when scenario execution can inspect checkpoint history."
    )
    extensions = (
        "The implementation includes a SQLite checkpointer path, a Fireworks/OpenAI-compatible "
        "LLM provider, mock HITL approval by default, optional real interrupts via "
        "`LANGGRAPH_INTERRUPT=true`, and optional LLM-as-judge evaluation via "
        "`LLM_EVALUATE=true`."
    )
    improvement = (
        "With another day, the first production hardening step would be replacing mock "
        "support tools with typed tool adapters, adding trace IDs for external calls, "
        "and expanding eval scenarios for ambiguous multi-intent tickets."
    )

    return f"""# Day 08 Lab Report

## 1. Team / student

- Name:
- Repo/commit:
- Date:

## 2. Architecture

{architecture}

{routing_summary}

## 3. State schema

| Field | Reducer | Why |
|---|---|---|
| `thread_id` | overwrite | LangGraph checkpoint namespace per scenario |
| `scenario_id` | overwrite | metrics and report identity |
| `query` | overwrite | normalized ticket text |
| `route` | overwrite | current LLM-selected route |
| `risk_level` | overwrite | approval and audit signal |
| `attempt` | overwrite | bounded retry counter |
| `max_attempts` | overwrite | per-scenario retry limit |
| `evaluation_result` | overwrite | conditional gate after tool evaluation |
| `pending_question` | overwrite | clarification path output |
| `proposed_action` | overwrite | risky action awaiting approval |
| `approval` | overwrite | HITL or mock decision |
| `final_answer` | overwrite | customer-facing completion response |
| `messages` | append | compact node trace |
| `tool_results` | append | tool history across retries |
| `errors` | append | retry and dead-letter diagnostics |
| `events` | append | structured audit log for metrics |

## 4. Metrics summary

| Metric | Value |
|---|---:|
| Total scenarios | {metrics.total_scenarios} |
| Success rate | {metrics.success_rate:.2%} |
| Average nodes visited | {metrics.avg_nodes_visited:.2f} |
| Total retries | {metrics.total_retries} |
| Total interrupts / approvals | {metrics.total_interrupts} |
| Resume or history evidence | {"yes" if metrics.resume_success else "no"} |

## 5. Scenario results

| Scenario | Expected route | Actual route | Success | Retries | Interrupts | Errors |
|---|---|---|---:|---:|---:|---|
{scenario_rows}

## 6. Failure analysis

{retry_failure}
{approval_failure}

## 7. Persistence / recovery evidence

{persistence}

## 8. Extension work

{extensions}

## 9. Improvement plan

{improvement}
"""


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    """Write the rendered report to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")
