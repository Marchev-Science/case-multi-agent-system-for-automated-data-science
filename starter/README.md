# Starter scaffold

This is a minimal, runnable skeleton. It boots, loads a config, calls the LLM, and walks an empty state machine. None of the **agents do anything yet**. That part is your job.

## What's done for you

| File | Status | What it gives you |
|---|---|---|
| `maads/__main__.py` | ✅ Working | CLI: `python -m maads run --case <name>` or `--config <path>` |
| `maads/config.py` | ✅ Working | Loads any `<case>.yaml` into a typed `CaseConfig` |
| `maads/state.py` | ✅ Working skeleton | Pydantic state with nested phase models for all 24 CRISP-DM outputs; `view_for(agent)` slicing for token economy |
| `maads/llm.py` | ✅ Working | Thin wrapper over OpenAI (with optional DeepSeek), retries, per-agent token accounting |
| `maads/tools.py` | ⚠️ Partial | `PythonExec` is working (subprocess sandbox, captured stdout/stderr, timeout). `FileIO` is working. `RAGRetriever` is a stub. |
| `maads/orchestrator.py` | 🟥 Stub | The state machine and routing logic are TODOs. |
| `maads/agents.py` | 🟥 Stubs | `Agent` base class is done. All five concrete agents are stubs, including the Developer's debugging-toolkit methods. |
| `maads/data_utils.py` | ✅ Working | Generic Kaggle download (any competition slug) + shorthands for the three demonstration cases. |
| `configs/*.yaml` | ✅ Working | One config per demonstration case. Schema is general; add a fourth by writing a new file. |
| `tests/test_smoke.py` | ✅ Working | Imports the package and runs a no-op pipeline. |

## What you implement

In order of suggested attack:

1. **`maads/agents.py`** — fill in each agent's `act()` method. Start with `ProjectManagerAgent`, then add the rest in CRISP-DM order. The Developer's debugging stubs (`classify_error`, `propose_fix`, `re_execute`, `repair_json`, `schema_check`) are first-class API methods, not nice-to-haves — implement them as you wire up the other agents, because every other agent's failures land in the Developer.
2. **`maads/orchestrator.py`** — write the loop that calls `pm.plan()` and dispatches; implement the four loop contours.
3. **`maads/tools.py`** — implement `RAGRetriever`. Build the corpus from `docs/RESOURCES.md` references and per-case notes.
4. Add a `SchemaChecker` and (optionally) a `LeakageCheck` tool.

**Before you write your first prompt, read [`../docs/TOKEN_BUDGET.md`](../docs/TOKEN_BUDGET.md).** Cache-friendly prompt structure, state slicing via `state.view_for(agent_name)`, output truncation, and JSON-mode caps are not optional — they're the difference between a small bill and a large one.

## Running it

```bash
# 1) Set up
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your OPENAI_API_KEY (and optionally DEEPSEEK_API_KEY).

# 2) Download data — one of the bundled demonstration cases, or any
#    arbitrary Kaggle competition by slug:
python -m maads data download --case titanic
python -m maads data download --competition spaceship-titanic

# 3) Run (will currently do nothing useful — that's your job)
python -m maads run --case titanic
python -m maads run --config configs/my_new_case.yaml
```

## Standardised interface for any Kaggle competition

The system is **general**. To point it at a Kaggle competition not in the bundled configs:

1. Write `configs/<name>.yaml` following the same schema as `titanic.yaml` (see `maads/config.py` for the typed schema).
2. `python -m maads data download --competition <kaggle-slug>`
3. `python -m maads run --config configs/<name>.yaml`

No agent code changes. If you find yourself wanting to edit an agent to handle a new dataset, lift the concept into the config file or the Domain Expert's RAG corpus instead.

## A few decisions baked in

- **Pydantic v2** is used for state and config. Stick with it for consistency.
- **State is append-only**. Lists in state only get appended to; existing entries aren't mutated. This makes a run replayable.
- **No agent reads another agent's prompt** — they only read state, via `state.view_for(agent_name)`.
- **All code execution goes through `PythonExec`.** No `eval()`, no `exec()`. The sandbox captures real tracebacks; agents must learn from them.
- **The Developer is on-call for every other agent's failures.** Design accordingly.

Read `../docs/ARCHITECTURE.md` before changing the contracts in `state.py` or `agents.py`.
