# Resources

A curated reading list. The first three are the minimum reading before starting.

## Minimum reading

1. **CRISP-DM 1.0**, the *detailed* version with substeps and loop contours. The Wikipedia article covers the six-phase summary but is too shallow. The original 2000 SPSS / CRISP-DM consortium document (a freely-findable PDF) is the source; any "CRISP-DM detailed reference" that lists all 24 substeps and their outputs is acceptable. The case is organised around these substeps; every agent prompt template references them.
2. The **quickstart for the framework you pick**. CrewAI: <https://docs.crewai.com/quickstart>. LangGraph: <https://langchain-ai.github.io/langgraph/tutorials/introduction/>. About one hour each.
3. The **`starter/` scaffold's own README**. It tells you which utilities are ready to drop into your framework.

## Choosing a framework

The case encourages using a multi-agent framework. Pick the one whose mental model fits your team. A short tour:

### CrewAI — recommended default

- **Mental model**: roles, goals, tools, and tasks. You define `Agent`s with role/goal/backstory, give them `Task`s with descriptions and expected outputs, and run them as a `Crew` with a `Process` (sequential or hierarchical).
- **Why it fits this case**: the role abstraction maps 1:1 onto the five CRISP-DM agents. The hierarchical process maps onto the PM-led hub-and-spoke. Tool integration is straightforward.
- **Docs**: <https://docs.crewai.com>
- **Examples worth reading**: the `examples/` folder on GitHub. Look for the multi-agent ones with explicit role + task definitions.

### LangGraph — recommended for explicit-control teams

- **Mental model**: a typed `State`, a graph of nodes, conditional edges. Each node reads from / writes to state. The four CRISP-DM loops are four conditional edges.
- **Why it fits this case**: the explicit state machine matches the CRISP-DM substep-by-substep walk. You can see exactly when a back-edge fires.
- **Docs**: <https://langchain-ai.github.io/langgraph/>
- **Cost**: more code per agent than CrewAI, steeper learning curve.

### OpenAI Agents SDK — minimal vendor SDK

- **Mental model**: agents with tools and handoffs. No heavy framework on top.
- **Why it fits**: smallest surface to learn. Good if the team is comfortable composing things themselves.
- **Cost**: less help with multi-agent coordination; you'll write more glue.
- **Docs**: <https://github.com/openai/openai-agents-python>

### Pydantic AI — type-first

- **Mental model**: agents return strictly-typed Pydantic outputs.
- **Why it fits**: strong type discipline catches mismatches early; output validation is automatic.
- **Cost**: younger ecosystem, fewer examples for multi-agent patterns.
- **Docs**: <https://ai.pydantic.dev/>

### What about AutoGen / AG2?

The original Microsoft AutoGen is in maintenance mode; the community fork AG2 (<https://ag2.ai/>) is where current activity is. If the team wants to use the AutoGen style (conversational agents), check AG2's current status and recent releases before committing.

### Mixing approaches

You can — and probably should — use a framework for the agents and orchestration, plus bring your own small utilities for things the framework doesn't handle well (the CRISP-DM-shaped state, the Kaggle download, a Python execution sandbox). The `starter/` directory in this repo gives you exactly those drop-in utilities.

## Background — agentic systems

- **ReAct: Synergizing Reasoning and Acting in Language Models** (Yao et al., 2022). Short, foundational, still worth reading. <https://arxiv.org/abs/2210.03629>
- **Reflexion: Language Agents with Verbal Reinforcement Learning** (Shinn et al., 2023). Relevant to the Validator's "try to find a problem" mandate.
- Any recent (2025–2026) multi-agent-system survey for context.

## Background — automated data science

- **AutoML lineage** (TPOT, auto-sklearn, H2O AutoML). Classical AutoML. Understand it well enough to explain in the paper how an agentic approach is *different* — more flexible, more interpretable, more expensive per run.
- **LLM-as-data-scientist papers**: search recent (2024–2026) literature for "Data Interpreter", "DS-Agent", "AutoGen Data Analyst", and similar. At least one should appear in the paper's related-work section.

## Background — Kaggle craft

- **"Approaching (Almost) Any Machine Learning Problem"** — Abhishek Thakur. Free PDF + Kaggle Learn. Excellent reference; consider including selected chapters in the Domain Expert's RAG corpus.
- **Per-dataset starting notebooks** (don't copy code, but read for context):
  - Titanic: dozens of canonical tutorials. They've been memorised by LLMs — useful only as background.
  - House Prices: Pedro Marcelino's "Comprehensive Data Exploration" is the canonical EDA reference; consider RAG-seeding it.
  - Disaster Tweets: the official Kaggle tutorial notebook is a fine baseline reference.

## Libraries you'll likely want

The scaffold's `requirements.txt` covers the basics. Beyond your chosen framework:

- `openai` — used for both OpenAI and DeepSeek (the latter is OpenAI-API-compatible).
- `pydantic`, `pyyaml` — typed config and state.
- `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost` — classical ML.
- `tiktoken` — token counting for cost accounting.
- `faiss-cpu` or hand-rolled cosine similarity in numpy — for RAG retrieval if your framework doesn't ship one.
- `optuna` — hyperparameter search if you go there.
- `pytest` — testing.
- `kaggle` — the official CLI for data download.  
- openshell by nVidia - sandboxing for agents

## What not to bother with

- Building a Streamlit / Gradio UI. Not part of the deliverable.
- Fine-tuning a transformer on Disaster Tweets. Burns budget for marginal gain.
- Rolling your own agent framework when established ones exist. *That is exactly what this case asks you not to do.*
