"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, ApprovalDecision, Route, make_event


class RouteClassification(BaseModel):
    """Structured output for LLM intent classification."""

    route: Literal["simple", "tool", "missing_info", "risky", "error"]
    risk_level: Literal["low", "high"] = "low"
    rationale: str = Field(default="", max_length=300)


class ToolEvaluation(BaseModel):
    """Optional LLM-as-judge output for tool quality checks."""

    evaluation_result: Literal["success", "needs_retry"]
    rationale: str = Field(default="", max_length=300)


ALLOWED_ROUTES = {route.value for route in Route if route not in {Route.DEAD_LETTER, Route.DONE}}


def _message_content(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(str(item) for item in content).strip()
    return str(content).strip()


def _loads_json_object(content: str) -> dict[str, Any]:
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM response did not contain a JSON object")
    return json.loads(content[start : end + 1])


def _invoke_structured(
    llm: Any,
    schema: type[BaseModel],
    messages: list[tuple[str, str]],
) -> BaseModel:
    try:
        result = llm.with_structured_output(schema).invoke(messages)
        if isinstance(result, schema):
            return result
        return schema.model_validate(result)
    except Exception:
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        fallback_messages = [
            *messages,
            (
                "system",
                "Return only a JSON object that validates against this schema: "
                f"{schema_json}",
            ),
        ]
        response = llm.invoke(fallback_messages)
        try:
            return schema.model_validate(_loads_json_object(_message_content(response)))
        except Exception as fallback_exc:
            raise RuntimeError("LLM did not return valid structured output") from fallback_exc


def _normalize_route(route: str) -> str:
    normalized = route.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in ALLOWED_ROUTES:
        return normalized
    return Route.SIMPLE.value


def _latest_tool_result(state: AgentState) -> str:
    tool_results = state.get("tool_results") or []
    if not tool_results:
        return ""
    return tool_results[-1]


def _heuristic_evaluation(tool_result: str) -> str:
    text = tool_result.upper()
    if not tool_result or "ERROR" in text or "TIMEOUT" in text:
        return "needs_retry"
    return "success"


def _coerce_approval(value: Any) -> dict[str, Any]:
    if isinstance(value, ApprovalDecision):
        return value.model_dump()
    if isinstance(value, bool):
        return ApprovalDecision(
            approved=value,
            reviewer="human-reviewer",
            comment="Decision supplied by interrupt resume.",
        ).model_dump()
    if isinstance(value, dict):
        return ApprovalDecision.model_validate(
            {
                "approved": bool(value.get("approved", False)),
                "reviewer": value.get("reviewer", "human-reviewer"),
                "comment": value.get("comment", ""),
            }
        ).model_dump()
    return ApprovalDecision(
        approved=False,
        reviewer="human-reviewer",
        comment="No approval decision was supplied.",
    ).model_dump()


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── Workflow nodes ─────────────────────────────────────────────────


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM.

    *** MUST use a real LLM call — keyword-only heuristics will lose points. ***

    Use .with_structured_output() or equivalent to get reliable enum classification.
    The LLM should classify into one of: simple, tool, missing_info, risky, error.

    Hints:
    - See llm.py for the get_llm() helper
    - Use Pydantic model or TypedDict with .with_structured_output()
    - Set risk_level to "high" for risky routes, "low" otherwise
    - Priority guide: risky > tool > missing_info > error > simple

    Return: {"route": str, "risk_level": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    llm = get_llm(temperature=0.0)
    messages = [
        (
            "system",
            "You classify support tickets for a LangGraph workflow. "
            "Return one route only: simple, tool, missing_info, risky, or error. "
            "Priority order: risky > tool > missing_info > error > simple. "
            "Use risky for side-effecting actions such as refunds, deletions, cancellations, "
            "agent-performed account changes, or sending emails. Do not classify self-service "
            "how-to or account-help instructions as risky. Use tool for lookups such as "
            "order status, tracking, searches, or fetching customer records. "
            "Use missing_info for vague requests that lack enough context to act. Use error "
            "for system failures such as "
            "timeouts, crashes, service unavailable, retry exhaustion, or unrecoverable failures. "
            "Use simple for general support questions answerable without tools.",
        ),
        ("human", f"Ticket query: {query}"),
    ]
    classification = _invoke_structured(llm, RouteClassification, messages)
    route = _normalize_route(str(getattr(classification, "route", Route.SIMPLE.value)))
    risk_level = "high" if route == Route.RISKY.value else "low"
    rationale = str(getattr(classification, "rationale", "")).strip()
    return {
        "route": route,
        "risk_level": risk_level,
        "messages": [f"classify:{route}"],
        "events": [
            make_event(
                "classify",
                "completed",
                f"classified as {route}",
                route=route,
                risk_level=risk_level,
                rationale=rationale,
            )
        ],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call.

    Simulate transient failures for error-route scenarios to test retry loops.

    Requirements:
    - Read current attempt count from state
    - If route is "error" and attempt < 2: return error result (string containing "ERROR")
    - Otherwise: return a mock success result string
    - Append result to tool_results list

    Return: {"tool_results": [result_string], "events": [make_event(...)]}
    """
    route = state.get("route", Route.SIMPLE.value)
    attempt = int(state.get("attempt", 0) or 0)
    query = state.get("query", "")

    if route == Route.ERROR.value and attempt < 2:
        result = f"ERROR: transient support backend timeout on tool attempt {attempt + 1}"
        status = "error"
    elif route == Route.TOOL.value:
        result = (
            "SUCCESS: Mock lookup completed. Order/customer record is reachable and "
            "the requested status context is available."
        )
        status = "success"
    elif route == Route.RISKY.value:
        proposed_action = state.get("proposed_action") or f"Process requested action: {query}"
        result = f"SUCCESS: Approved action recorded for execution: {proposed_action}"
        status = "success"
    else:
        result = "SUCCESS: Mock support tool completed successfully."
        status = "success"

    return {
        "tool_results": [result],
        "messages": [f"tool:{status}"],
        "events": [
            make_event(
                "tool",
                "completed",
                status,
                attempt=attempt,
                route=route,
            )
        ],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the retry-loop gate.

    Check whether the latest tool result is satisfactory or needs retry.

    SHOULD use LLM-as-judge for bonus points. Heuristic (e.g., check for "ERROR" substring)
    is acceptable for base score.

    Requirements:
    - Read the latest entry from tool_results
    - Set evaluation_result to "needs_retry" or "success"
    - This field drives route_after_evaluate conditional edge

    Note: You may need to add 'evaluation_result' to AgentState if not present.

    Return: {"evaluation_result": str, "events": [make_event(...)]}
    """
    latest_result = _latest_tool_result(state)
    evaluation_result = _heuristic_evaluation(latest_result)
    rationale = "Heuristic check: latest tool result is usable."

    if os.getenv("LLM_EVALUATE", "").lower() in {"1", "true", "yes"}:
        try:
            llm = get_llm(temperature=0.0)
            judge = _invoke_structured(
                llm,
                ToolEvaluation,
                [
                    (
                        "system",
                        "You judge whether a support workflow tool result is usable. "
                        "Return needs_retry only when the result is missing, timed out, "
                        "or explicitly contains an error.",
                    ),
                    ("human", f"Tool result:\n{latest_result}"),
                ],
            )
            evaluation_result = str(getattr(judge, "evaluation_result", evaluation_result))
            rationale = str(getattr(judge, "rationale", rationale))
        except Exception as exc:
            rationale = f"LLM judge unavailable; used heuristic. Reason: {exc}"

    return {
        "evaluation_result": evaluation_result,
        "messages": [f"evaluate:{evaluation_result}"],
        "events": [
            make_event(
                "evaluate",
                "completed",
                evaluation_result,
                latest_result=latest_result,
                rationale=rationale,
            )
        ],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM.

    *** MUST use a real LLM call — hardcoded strings will lose points. ***

    The LLM should generate a helpful response grounded in available context:
    - tool_results (if any)
    - approval decision (if risky route)
    - original query

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    llm = get_llm(temperature=0.2)
    context = {
        "query": state.get("query", ""),
        "route": state.get("route", ""),
        "risk_level": state.get("risk_level", ""),
        "tool_results": state.get("tool_results", []),
        "approval": state.get("approval"),
        "errors": state.get("errors", []),
    }
    response = llm.invoke(
        [
            (
                "system",
                "You are a concise support-ticket assistant. Generate a helpful final "
                "response grounded only in the provided workflow context. If tool results "
                "exist, base the answer on them. If a risky action was approved, mention "
                "that approval was recorded before the action proceeded. Do not invent "
                "external facts or claim real-world completion beyond the mock tool result.",
            ),
            (
                "human",
                "Workflow context JSON:\n"
                f"{json.dumps(context, ensure_ascii=False, indent=2)}\n\n"
                "Write the final customer-facing answer in 2-4 sentences.",
            ),
        ]
    )
    final_answer = _message_content(response)
    return {
        "final_answer": final_answer,
        "messages": ["answer:completed"],
        "events": [make_event("answer", "completed", "answer generated")],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    Generate a specific clarification question based on the vague/incomplete query.

    Note: You may need to add 'pending_question' to AgentState if not present.

    Return: {"pending_question": str, "final_answer": str, "events": [make_event(...)]}
    """
    approval = state.get("approval") or {}
    if approval and not approval.get("approved", False):
        question = (
            "The requested action was not approved. What alternative support action "
            "would you like me to take?"
        )
    else:
        question = (
            "Could you provide the affected account, order ID, or exact issue details "
            "so I can help without guessing?"
        )
    return {
        "pending_question": question,
        "final_answer": question,
        "messages": ["clarify:pending"],
        "events": [make_event("clarify", "completed", "clarification requested")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval.

    Describe the proposed action and why it requires approval.

    Note: You may need to add 'proposed_action' to AgentState if not present.

    Return: {"proposed_action": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    proposed_action = (
        f"Review and, if approved, perform the requested side-effecting support action: {query}"
    )
    return {
        "proposed_action": proposed_action,
        "messages": ["risky_action:prepared"],
        "events": [
            make_event(
                "risky_action",
                "completed",
                "approval required",
                proposed_action=proposed_action,
            )
        ],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default behavior: mock approval (approved=True) so tests and CI run offline.
    Extension: if env LANGGRAPH_INTERRUPT=true, use langgraph.types.interrupt() for real HITL.

    Return: approval decision and audit events.
    """
    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() in {"1", "true", "yes"}:
        from langgraph.types import interrupt

        decision = interrupt(
            {
                "kind": "approval_request",
                "scenario_id": state.get("scenario_id"),
                "query": state.get("query"),
                "proposed_action": state.get("proposed_action"),
                "risk_level": state.get("risk_level"),
            }
        )
        approval = _coerce_approval(decision)
        event_type = "interrupt"
    else:
        approval = ApprovalDecision(
            approved=True,
            reviewer="mock-reviewer",
            comment="Auto-approved for deterministic lab execution.",
        ).model_dump()
        event_type = "completed"

    return {
        "approval": approval,
        "messages": [f"approval:{approval['approved']}"],
        "events": [
            make_event(
                "approval",
                event_type,
                "approval recorded",
                approved=approval["approved"],
                reviewer=approval["reviewer"],
            )
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt.

    Increment the attempt counter and log the transient failure.

    Requirements:
    - Read current attempt from state, increment by 1
    - Add an error message to errors list
    - Return updated attempt count

    Return: {"attempt": int, "errors": [str], "events": [make_event(...)]}
    """
    next_attempt = int(state.get("attempt", 0) or 0) + 1
    latest_result = _latest_tool_result(state)
    error = latest_result or "Initial error route requires a tool retry attempt."
    return {
        "attempt": next_attempt,
        "errors": [f"attempt {next_attempt}: {error}"],
        "messages": [f"retry:{next_attempt}"],
        "events": [
            make_event(
                "retry",
                "completed",
                "retry attempt recorded",
                attempt=next_attempt,
                max_attempts=state.get("max_attempts", 3),
            )
        ],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded.

    This is the third layer: retry → fallback → dead letter.
    Log the failure and set a final_answer explaining that the request could not be completed.

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    attempt = int(state.get("attempt", 0) or 0)
    max_attempts = int(state.get("max_attempts", 3) or 3)
    final_answer = (
        "I could not complete this request because the support workflow exhausted "
        f"its retry limit ({attempt}/{max_attempts}). I have routed it to the "
        "dead-letter path for manual investigation."
    )
    return {
        "evaluation_result": "failed",
        "final_answer": final_answer,
        "errors": [f"dead_letter: exhausted retries at attempt {attempt}"],
        "messages": ["dead_letter:completed"],
        "events": [
            make_event(
                "dead_letter",
                "completed",
                "retry limit exhausted",
                attempt=attempt,
                max_attempts=max_attempts,
            )
        ],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END.

    Return: {"events": [make_event("finalize", "completed", "workflow finished")]}
    """
    return {
        "messages": ["finalize:completed"],
        "events": [
            make_event(
                "finalize",
                "completed",
                "workflow finished",
                route=state.get("route"),
                has_answer=bool(state.get("final_answer")),
            )
        ],
    }
