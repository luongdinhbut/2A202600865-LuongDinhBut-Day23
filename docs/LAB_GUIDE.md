# Lab Guide

## Step 1 — Understand the target graph

Target flow:

```text
START -> intake -> classify -> route
route simple       -> answer -> finalize -> END
route tool         -> tool -> evaluate -> answer -> finalize -> END
route tool (retry) -> tool -> evaluate -> retry -> tool -> evaluate -> ... (loop)
route missing_info -> clarify -> finalize -> END
route risky        -> risky_action -> approval -> tool -> evaluate -> answer -> finalize -> END
route error        -> retry -> tool -> evaluate -> retry -> ... (loop until success or max)
route (max retry)  -> retry -> dead_letter -> finalize -> END
```

## Step 2 — Implement TODOs in order

### Phase 1: State schema + nodes (0–90 min)

1. **`state.py`**: Review existing fields. You will need to ADD fields as you implement nodes:
   - `evaluation_result` — needed for retry loop gate in `route_after_evaluate`
   - `pending_question` — needed for clarification flow
   - `proposed_action` — needed for risky action flow
   - `approval` — needed for HITL approval decisions
   - Decide which new fields should be append-only (`Annotated[list, add]`) vs overwrite

2. **`llm.py`**: Review the LLM helper. Set up your `.env` with `FIREWORKS_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY`. For Fireworks DeepSeek Pro, install `pip install -e '.[fireworks]'` or `pip install langchain-openai`, then set `LLM_MODEL=accounts/fireworks/models/deepseek-v4-pro`.

3. **`nodes.py`**: Implement all node functions. Key requirements:
   - `classify_node`: **MUST use LLM** with structured output for intent classification
   - `answer_node`: **MUST use LLM** to generate grounded responses
   - `evaluate_node`: SHOULD use LLM-as-judge (heuristic OK for base score)
   - `tool_node`: mock tool with error simulation for retry testing
   - `approval_node`: mock approval (approved=True) by default
   - `dead_letter_node`: log failures when max retries exceeded

### Phase 2: Routing + graph wiring (90–150 min)

4. **`routing.py`**: Implement all 4 routing functions:
   - `route_after_classify`: map route string → next node name
   - `route_after_evaluate`: retry loop gate (needs_retry → retry, else → answer)
   - `route_after_retry`: bounded retry check (attempt < max → tool, else → dead_letter)
   - `route_after_approval`: approved → tool, rejected → clarify

5. **`graph.py`**: Build the complete StateGraph:
   - Import and register all 11 nodes
   - Wire fixed edges (START→intake, intake→classify, tool→evaluate, etc.)
   - Wire conditional edges using routing functions
   - Compile with checkpointer
   - Verify: all paths terminate at finalize → END

6. **Verify**: `make test` and `make run-scenarios`

### Phase 3: Persistence (150–180 min)

7. **`persistence.py`**: Implement SQLite checkpointer:
   - `"sqlite"` → `SqliteSaver` with `sqlite3.connect()` and WAL mode
   - Show evidence: thread_id per run, state history, or crash-resume

### Phase 4: Metrics, report, tests (180–240 min)

8. **`report.py`**: Implement `render_report()` using the template
9. **Run all scenarios**: `make run-scenarios` → generates `outputs/metrics.json`
10. **Validate**: `make grade-local` → checks metrics schema
11. **Write report**: Fill `reports/lab_report.md` with architecture, metrics, failure analysis

### Phase 5: Extensions (240+ min) — push toward 90+

Pick one or more:
- **Real HITL**: Set `LANGGRAPH_INTERRUPT=true`, use `interrupt()` in approval_node
- **Streamlit UI**: Build approval/reject interface with interrupt/resume
- **Time travel**: Use `get_state_history()` to replay from earlier checkpoint
- **Crash recovery**: Show SQLite checkpoint survives process kill + restart
- **Parallel fan-out**: Use `Send()` to run two tools concurrently
- **Graph diagram**: Export Mermaid diagram via `graph.get_graph().draw_mermaid()`

## Step 3 — Run and validate

```bash
make run-scenarios
make grade-local
```

## Step 4 — Extension tasks

See Phase 5 above.

## Submission checklist

- [ ] All `TODO(student)` sections implemented
- [ ] `.env` configured with LLM API key
- [ ] `make test` passes
- [ ] `make run-scenarios` writes `outputs/metrics.json`
- [ ] `make grade-local` validates metrics
- [ ] `reports/lab_report.md` is completed
- [ ] `classify_node` uses real LLM call (not keyword-only)
- [ ] `answer_node` uses real LLM call (not hardcoded)
- [ ] You can explain one route and one failure mode in demo
