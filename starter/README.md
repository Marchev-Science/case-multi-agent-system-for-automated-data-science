# Starter — drop-in utilities

This directory provides a small set of utilities you can drop into whichever multi-agent framework you pick (CrewAI, LangGraph, OpenAI Agents SDK, etc.). It is **not** a fill-in-the-blanks skeleton; the multi-agent orchestration itself is something your framework provides for free.

## What's here

| File | What it gives you | Use it for |
|---|---|---|
| `maads/config.py` | Typed `CaseConfig` loader for any `<case>.yaml`. | Reading per-competition configuration into Python. |
| `maads/state.py` | A Pydantic `CrispDMState` with one field per CRISP-DM 1.0 reference-model output. | The shared object your agents read from and write to. Compatible with CrewAI's shared context, LangGraph's typed state, etc. |
| `maads/tools.py` | `PythonExec` (subprocess sandbox with timeout, captured stdout/stderr), `FileIO`, and a stub `RAGRetriever`. | Pass `PythonExec` to your code-running agents as a tool. The Python sandbox is the most expensive utility to get right; this one is small but correct. |
| `maads/data_utils.py` | Kaggle downloader. `download_kaggle_competition(slug, out)` works for any competition; `download_case_data(case)` is a shorthand for the three demonstration cases. | One command, any competition. |
| `maads/llm.py` | Minimal OpenAI / DeepSeek wrapper with retries and per-call token accounting. | Optional — most frameworks have their own LLM client. Use this if you want a single thin interface across both providers. |
| `configs/*.yaml` | One config per demonstration case. Schema is general; add a fourth by writing a new file. | The required-input schema your system will read. |
| `tests/test_smoke.py` | Smoke tests covering the utilities above. | Sanity check after installation. |

## What is intentionally **not** here

There is no `Agent` base class, no orchestrator loop, no agent stubs. That's the work the framework does for you — defining what an agent is, how they communicate, how the loop runs. If you find yourself wanting to write that code by hand, stop and check whether your framework already provides it (it does). Your time is better spent on prompts, agent roles, and the CRISP-DM mapping.

## Running it

```bash
# 1) Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# add your framework of choice, e.g.:
pip install crewai
# or
pip install langgraph

cp .env.example .env
# Edit .env with your OPENAI_API_KEY (and optionally DEEPSEEK_API_KEY).

# 2) Download data for a demonstration case, or any Kaggle competition by slug:
python -m maads data download --case titanic
python -m maads data download --competition spaceship-titanic

# 3) Run the smoke tests
pytest tests/

# 4) Wire your framework up against these utilities. See ../docs/ARCHITECTURE.md
#    for what the five agents need.
```

The bundled `python -m maads run` command is a placeholder — it loads a config and walks an empty no-op orchestrator that just halts. Once you've wired your framework to do the real work, replace the placeholder body in `maads/__main__.py`'s `cmd_run` with a call into your framework's entry point.

## How to add a fourth Kaggle competition

The system is general. To point it at any Kaggle competition:

1. Write `configs/<name>.yaml` following the same schema as `titanic.yaml` (see `maads/config.py` for the typed schema).
2. `python -m maads data download --competition <kaggle-slug>`
3. Run your system with `--config configs/<name>.yaml`.

No agent code should change. If you find yourself wanting to special-case a dataset in an agent, lift the concept into the config file or the Domain Expert's RAG corpus instead.

## A few conventions baked in

- **Pydantic v2** for state and config. Stick with it.
- **State is append-only**. Lists in state only get appended to; existing entries aren't mutated. This makes a run replayable.
- **All code execution goes through `PythonExec`** (or whatever sandbox your framework provides). No `eval()`, no `exec()`. The sandbox captures real tracebacks; agents must learn from them.
- **No agent reads another agent's prompt** — they only read state. Whatever your framework calls "state", give each agent a minimal slice (see `state.view_for(agent_name)` for one approach).

Read `../docs/ARCHITECTURE.md` and `../docs/TOKEN_BUDGET.md` before doing anything with prompts.
