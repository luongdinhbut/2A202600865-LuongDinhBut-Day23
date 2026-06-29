# Day 08 Lab Report

## 1. Team / student

- Name:
- Repo/commit:
- Date:

## 2. Architecture

The workflow is a typed LangGraph `StateGraph` for support-ticket orchestration. It starts with `intake`, uses `classify` to choose a supported route, and then fans into direct answering, tool execution, clarification, risky-action approval, retry, or dead-letter handling. Every route converges through `finalize` before `END`, so all paths leave an audit event trail.

The graph uses conditional edges after classification, tool evaluation, retry accounting, and approval. The retry loop is bounded by `attempt < max_attempts`; once the limit is reached the workflow emits a dead-letter response.

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
| Total scenarios | 7 |
| Success rate | 100.00% |
| Average nodes visited | 6.43 |
| Total retries | 3 |
| Total interrupts / approvals | 2 |
| Resume or history evidence | yes |

## 5. Scenario results

| Scenario | Expected route | Actual route | Success | Retries | Interrupts | Errors |
|---|---|---|---:|---:|---:|---|
| S01_simple | simple | simple | yes | 0 | 0 |  |
| S02_tool | tool | tool | yes | 0 | 0 |  |
| S03_missing | missing_info | missing_info | yes | 0 | 0 |  |
| S04_risky | risky | risky | yes | 0 | 1 |  |
| S05_error | error | error | yes | 2 | 0 | attempt 1: Initial error route requires a tool retry attempt.<br>attempt 2: ERROR: transient support backend timeout on tool attempt 2 |
| S06_delete | risky | risky | yes | 0 | 1 |  |
| S07_dead_letter | error | error | yes | 1 | 0 | attempt 1: Initial error route requires a tool retry attempt.<br>dead_letter: exhausted retries at attempt 1 |

## 6. Failure analysis

1. Retry or tool failure: transient tool failures are represented as `ERROR` tool results. `evaluate` marks them as `needs_retry`, `retry` increments `attempt`, and `route_after_retry` either returns to `tool` or moves to `dead_letter` when the limit is reached.
2. Risky action without approval: refund, delete, cancellation, and email-sending requests route to `risky_action` first. The workflow records a proposed action and must pass through `approval` before any tool execution. If approval is rejected, the graph asks for a safer alternative through `clarify`.

## 7. Persistence / recovery evidence

Each scenario uses a distinct `thread_id` in the LangGraph config. The checkpointer is injected at compile time, so the same graph works with memory, SQLite, or Postgres checkpoint backends. `resume_success` is set from state-history availability when scenario execution can inspect checkpoint history.

## 8. Extension work

The implementation includes a SQLite checkpointer path, a Fireworks/OpenAI-compatible LLM provider, mock HITL approval by default, optional real interrupts via `LANGGRAPH_INTERRUPT=true`, and optional LLM-as-judge evaluation via `LLM_EVALUATE=true`.

## 9. Improvement plan

With another day, the first production hardening step would be replacing mock support tools with typed tool adapters, adding trace IDs for external calls, and expanding eval scenarios for ambiguous multi-intent tickets.
