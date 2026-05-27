# Resources

A curated reading list. The first three are the minimum reading before writing any agent code.

## Minimum reading

1. **CRISP-DM 1.0**, the *detailed* version with substeps and loop contours. The Wikipedia article covers the six-phase summary but is too shallow. The original 2000 SPSS / CRISP-DM consortium document (a freely-findable PDF) is the source; any "CRISP-DM detailed reference" that lists all 24 substeps and their outputs is acceptable. The case is organised around these substeps; every agent prompt template references them.
2. The **OpenAI Python SDK quickstart** — <https://github.com/openai/openai-python>. You will write directly against it.
3. The **`starter/` scaffold's own README**. It tells you which files are already working and which are TODO-stubs.

## Why no framework — and what the popular frameworks would have done for you

The case forbids using a multi-agent framework, but the team should know what it is *not* using:

| Framework | What it would have done | Why not for this case |
|---|---|---|
| **CrewAI** | Role-based agent definitions in Python/YAML, automatic task delegation, shared context. Around 31k stars. | Hides the orchestration. The educational point is to *build* the orchestration. |
| **LangGraph** | Explicit graph-of-nodes state machine; very production-grade. | Same — and adds a steep learning curve to the no-go. |
| **AutoGen / AG2** | Conversational multi-agent. | Effectively in maintenance mode; not recommended for new builds. |
| **OpenClaw** | Self-hosted autonomous-agent runtime; the largest open-source agent project of its kind. Uses Markdown `SKILL.md` files for capabilities. | OpenClaw is a runtime, not a library. You use it to *run* an agent on your machine, not to embed one in a Python application. Wrong shape for this case. |
| **Hermes Agent** | Open-source agent from Nous Research; strong persistent memory, multi-platform reach, growing multi-agent feature. | Same shape as OpenClaw — operator-facing runtime, not an embeddable library. |
| **OpenAI Agents SDK** | Vendor-native, minimal. | Closest to "no framework", but still imposes its own loop. Replicating its essential ideas yourself is, in fact, the exercise. |
| **Pydantic AI** | Type-safe agent primitives. | Tempting and minimal — but **using it counts as using a framework** for this case. You may borrow conceptual ideas from it; you may not import it as your orchestration layer. |

A useful frame: an industrial team would pick CrewAI and ship in three days. An educational build does the same thing from scratch and *understands* the failure modes of every layer.

## Background — agentic systems

- **ReAct: Synergizing Reasoning and Acting in Language Models** (Yao et al., 2022). Short, foundational, still worth reading. <https://arxiv.org/abs/2210.03629>
- **Reflexion: Language Agents with Verbal Reinforcement Learning** (Shinn et al., 2023). Relevant to the Validator's "try to find a problem" mandate.
- A recent multi-agent-system survey (e.g. from DataCamp, Lindy, or any 2025–2026 comparison). Useful as background even though you're not using them.

## Background — automated data science

- **AutoML lineage** (TPOT, auto-sklearn, H2O AutoML). Classical AutoML. Understand it well enough to explain in the paper how an agentic approach is *different* — more flexible, more interpretable, more expensive per run.
- **LLM-as-data-scientist papers**: search recent (2024–2026) literature for "Data Interpreter", "DS-Agent", "AutoGen Data Analyst", and similar. At least one should appear in the paper's related-work section.

## Background — Kaggle craft

- **"Approaching (Almost) Any Machine Learning Problem"** — Abhishek Thakur. Free PDF + Kaggle Learn. Excellent reference; consider including selected chapters in the Domain Expert's RAG corpus.
- **Per-dataset starting notebooks** (don't copy code, but read for context):
  - Titanic: dozens of canonical tutorials. They've been memorised by LLMs — useful only as background.
  - House Prices: Pedro Marcelino's "Comprehensive Data Exploration" is the canonical EDA reference; consider RAG-seeding it.
  - Disaster Tweets: the official Kaggle tutorial notebook is a fine baseline reference.

## Tools and libraries

Allowed without question (these are not "frameworks"):

- `openai` — the SDK. Used for both OpenAI and DeepSeek (the latter is OpenAI-API-compatible).
- `pydantic`, `pyyaml` — typed config and state.
- `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost` — classical ML.
- `tiktoken` — token counting for cost accounting.
- `faiss-cpu` or hand-rolled cosine similarity in numpy — RAG retrieval.
- `optuna` — hyperparameter search if you go there.
- `pytest` — testing.
- `kaggle` — the official CLI for data download.

Not in keeping with the from-scratch spirit of the case (avoid using these as the orchestration layer):

- `crewai`, `langgraph`, `langchain`, `autogen`, `pyautogen`, `pydantic-ai`, `instructor` (when used as an agent framework), or any other published multi-agent framework.

Acceptable but discouraged:

- `litellm` for multi-provider routing. Tempting, but the scaffold's small `llm.py` does what you need without the dependency. Use it only if there's a specific reason.
- A vector-DB service (Pinecone, Weaviate). Overkill for the corpus size.

## What not to bother with

- Building a Streamlit / Gradio UI. Not part of the deliverable.
- Fine-tuning a transformer on Disaster Tweets. Burns budget for marginal gain.
- A 200-line config-validation library. Skip the nice-to-haves.
