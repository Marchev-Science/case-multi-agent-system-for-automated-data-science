# Architecture

This document is a **design sketch**, not a specification. The team is expected to adapt every decision below to its taste and to the framework it picks, and to defend the result in the final architecture write-up. What is non-negotiable is that all five (or optionally six) agents and all four CRISP-DM loops are implemented and observable.

The shape of this document changed in the last revision: it used to prescribe a specific Python class hierarchy. Since the case now encourages using a mature multi-agent framework, the prescription moved up a level. We describe **what** each agent does and **what** the shared state looks like; **how** that lands in code is a function of the framework you pick.

## 1. The shared state — what flows between agents

Whatever framework you choose, the five agents need a shared notion of "what we know so far". In CrewAI this lives in the `Crew`'s shared context; in LangGraph it is the typed `State` of the graph; in the OpenAI Agents SDK you carry it explicitly via tool calls. The *shape* below is what matters — call it a dict, a dataclass, a pydantic model, a graph state, whatever your framework prefers.

The naming follows the CRISP-DM 1.0 Reference Model outputs verbatim, so a prompt can reference an output by its canonical name. Suggested shape:

```
state:
    case_id, config, phase, substep, loop_history, halted, halt_reason

    bu — Business Understanding outputs
        background, business_objectives, business_success_criteria,
        inventory_of_resources, requirements_assumptions_constraints,
        risks_and_contingencies, terminology, costs_and_benefits,
        data_mining_goals, data_mining_success_criteria,
        project_plan, initial_assessment_of_tools_and_techniques

    du — Data Understanding outputs
        initial_data_collection_report, data_description_report,
        data_exploration_report, data_quality_report

    dp — Data Preparation outputs
        rationale_for_inclusion_exclusion, data_cleaning_report,
        derived_attributes, generated_records, merged_data,
        reformatted_data, dataset, dataset_description

    md — Modeling outputs
        modeling_technique, modeling_assumptions, test_design,
        models[], chosen_model

    ev — Evaluation outputs
        assessment_of_dm_results, approved_models,
        review_of_process, list_of_possible_actions, decision

    dep — Deployment outputs
        deployment_plan, monitoring_and_maintenance_plan,
        final_report_path, final_presentation_path,
        experience_documentation, submission_path

    log[], token_spend{}
```

The `starter/` directory ships a typed Pydantic version of this (`starter/maads/state.py`) you can drop in regardless of the framework you pick. Most frameworks will happily accept a Pydantic model as their state object.

Three guidelines around the state:

- **Append-only logs.** `log`, `loop_history`, and `md.models` should only grow during a run. Earlier entries are not mutated. This makes a run replayable and inspectable.
- **No agent receives the whole state.** Send each agent only the slice it needs. In CrewAI this is a matter of what you put in the `Task` description; in LangGraph, what each node reads from the state. The `starter/` includes a `view_for(agent_name)` helper that returns a minimal slice for each role — see [`TOKEN_BUDGET.md`](TOKEN_BUDGET.md) for why this matters.
- **Outputs are first-class.** Every CRISP-DM substep produces a named output. The corresponding state field should be filled by the substep's owner. The orchestrator's prereq checks should refuse to dispatch a substep whose inputs haven't been written yet.

## 2. The agents

The five agents and their CRISP-DM ownership are described in the README. Here we sketch what each one looks like in practice — context, tools, and the kind of prompt that suits the role. Adapt to your framework.

### 2.1 Project Manager

- **Owns**: phase transitions, all four loop contours, plus 1.4 (Project Plan), 5.2 (Review Process), 5.3 (Determine Next Steps).
- **Context it needs**: current phase/substep, recent log entries (trimmed), latest data-quality blockers, latest model assessment, the loop history.
- **Tools**: just the LLM, plus the ability to read the shared state and signal a decision back to the orchestrator. No code execution.
- **Prompt shape**: stable system prompt that lays out the substeps for the current phase, the exit conditions, and the four loop triggers. The user message contains the minimal state view. The model's output is a structured decision: advance / loop_back / halt.
- **Guardrails**: hard caps on total phase transitions, on visits to any phase, on Loop B iterations. These belong in the orchestrator (or the framework's transition logic) — *not* in the prompt. Asking the model nicely doesn't bound anything.

### 2.2 Domain Knowledge Expert

- **Owns**: full Business Understanding (1.1–1.3); contributes to Data Understanding (2.1, 2.2, 2.3).
- **Context it needs**: the problem statement from the config, any feature hints, and anything the Data Engineer has already written to the state.
- **Tools**: a RAG retriever over a small corpus (CRISP-DM excerpts, problem-specific notes, prior Kaggle write-ups). Most frameworks include a retriever tool you can plug in.
- **Prompt shape**: "Translate a business problem and a dataset into well-formed data-mining goals and hypotheses." This is the one agent that should be allowed to *speculate* — its job is to generate ideas, not to commit to them.

### 2.3 Data Engineer

- **Owns**: full Data Understanding (2.1, 2.2, 2.4) and full Data Preparation (3.1–3.5).
- **Context it needs**: the raw data paths, the data-mining goals, anything already written to `du` and `dp`.
- **Tools**: a Python execution sandbox. This is non-negotiable — the agent must actually run pandas, not "imagine" it has. Frameworks differ in what they ship: CrewAI has tools for code execution; LangGraph nodes can call any Python function directly. Either way, the rule holds: only commit to state what survived execution.
- **Prompt shape**: "Profile, clean, and prepare data for modelling. Your outputs are structured reports, not free-form prose."

### 2.4 Data Scientist

- **Owns**: a modelling-lens contribution to 2.3 (Explore Data); full Modeling (4.1–4.4); validation in 4.4 and 5.1 unless a separate Validator is introduced.
- **Context it needs**: data-mining goals, the prepared dataset paths, the data description report, recent model attempts, the test design.
- **Tools**: Python sandbox with sklearn, xgboost, lightgbm, catboost.
- **Prompt shape**: "Pick one technique from a constrained menu, justify the choice, build, score on the test design. On poor performance, write a concrete diagnostic — do not silently keep trying more models."
- **Constraint**: cap inner-loop iterations at three to keep token cost bounded.

### 2.5 Developer

- **Owns**: cross-cutting tool development and debugging through Phases 2–5; full Deployment (6.1–6.4).
- **Two distinct roles**:
  1. **Development** during Phases 2–5: writes helper Python that other agents need (custom encoders, feature builders, integration glue, the production pipeline). Called by the orchestrator (or the PM) when another agent's task requires a piece of code beyond a one-liner.
  2. **Debugging** whenever any agent's code execution fails. This is the on-call responsibility — every other agent's failures surface here. The Developer's prompt should be **diagnose-first**: read the stack trace, classify the error, propose the smallest fix, re-execute. A retry budget of three is usually enough.
- **Tools**: Python sandbox, file IO, packaging utilities, a JSON-repair helper for fixing malformed agent outputs, a schema checker that verifies column references against the data description report.
- **Debugging capabilities** (worth listing because they're the most likely to be skipped):
  - **Error classification** — given stderr, label as schema / shape / type / leakage / library-version / OOM / timeout / syntax / JSON-parse / other.
  - **Schema check** — given a code snippet, list any column names it references that aren't in `du.data_description_report`.
  - **Iterative re-execution** — propose a fix, run it, capture the new result, repeat up to N times, then surface as a "stuck on substep" diagnostic.
  - **JSON repair** — strip Markdown fencing, fix unbalanced braces, escape control characters, then re-parse.
  - In CrewAI, these are just tools you attach to the Developer agent. In LangGraph, they're functions called from the Developer's node.

The Developer is the agent that prevents "five agents each fail in their own way" from turning into ten interacting failure modes. Without it, every other agent has to handle errors itself, and they will not handle them as well.

### 2.6 Optional sixth: Validator

If the team chooses a 6-agent design, the Validator owns 4.4 and 5.1 separately, with its own context, its own prompt template, and an adversarial mandate ("try to reject"). The Data Scientist then loses 4.4/5.1 ownership. This is the "two-keys-to-launch" pattern — it reduces silent failures, at the cost of one more agent's worth of prompt engineering.

## 3. The orchestrator and the four loops

The Project Manager doesn't itself execute the substeps — it decides which substep is next and who should run it. The framework you pick provides the actual orchestration; your job is to encode the CRISP-DM-specific decisions:

- **Which agent owns which substep.** A small lookup table. Some frameworks (CrewAI) let you define this declaratively via the `Task → Agent` binding.
- **What "done enough" means for each phase.** A function of the substep's expected outputs being present in state.
- **When each loop fires.** The triggers in the README's loop table become conditional transitions in the orchestrator.
- **Hard caps that bound the system.** Total phase transitions, visits per phase, iterations of Loop B.

### Implementing the four loops

| Loop | Trigger | Implementation (concept) |
|---|---|---|
| **A — 2 → 1** | After 2.4, if `du.data_quality_report["blockers"]` is non-empty, or if Domain Expert hypotheses are contradicted by the actual schema | The PM emits a "loop A" decision; the orchestrator routes back to substep 1.3. Domain Expert re-runs *only* 1.3, not the full Phase 1. |
| **B — 4 → 3** | After 4.4, if the latest `ModelRun.assessment` flags a specific preparation deficit *and* CV score is below threshold | Route back to the affected substep of Phase 3. Cap at three iterations. |
| **C — 5 → 1** | After 5.1, if business success criteria are not met *and* Loop A has not already fired twice | Route back to 1.3. Triggers a fundamental rethink of data-mining goals. |
| **D — 6 → 1** | After 6.4 (optional, stretch goal) | The experience documentation is appended to the *next dataset*'s RAG corpus. Implements cross-dataset learning. |

A run that fires zero loops on Titanic but at least one on House Prices or Disaster Tweets is a sign the system is actually thinking, not just executing.

In **CrewAI**, loops manifest as conditional task hand-offs — the PM agent's task returns a structured decision that the orchestration code reads to decide which task to schedule next. In **LangGraph**, loops are first-class: a conditional edge from the PM node back to a Phase-3 (or Phase-1) entry node, with the condition read from the state.

## 4. Tools the agents need

Roughly:

| Tool | Status in `starter/` | Purpose |
|---|---|---|
| Python execution sandbox | **Working** (`PythonExec` in `tools.py`) | Sole way agents touch data. |
| LLM client wrapper | **Working** (`llm.py`) | OpenAI / DeepSeek under one interface. Your framework probably has its own — use whichever fits. |
| File IO helpers | **Working** (`FileIO`) | Persisting reports, model artefacts, submissions. |
| RAG retriever | **Stub** (`RAGRetriever`) | Domain Expert's grounding. Most frameworks have a built-in retriever you can use instead. |
| Schema checker | Not in scaffold | Validates that any column referenced in agent output actually exists in the data. Strongly recommended. |
| Leakage check | Not in scaffold | Validator-side; detects target leakage and train/test contamination. |
| Error classifier | Not in scaffold | Developer-side; categorises failed `PythonExec` calls and proposes fix templates. |
| Kaggle client | Not in scaffold | Optional: submit and read back the public-leaderboard score. The Kaggle CLI is usually enough. |

## 5. Communication pattern

For the first working version, use **hub-and-spoke**: agents only talk to / through the PM. Agents do not call each other directly. This makes the system easy to debug (every transition is logged in one place) and easy to reason about (only the PM fires loops).

CrewAI defaults to a process-based hub-and-spoke (the `Process.sequential` and `Process.hierarchical` patterns). LangGraph lets you draw whatever shape you want; start simple.

If, after the first end-to-end pipeline works, the team wants to experiment with peer-to-peer (e.g. Data Engineer ↔ Data Scientist negotiating feature engineering directly), do it on a branch and keep the hub-and-spoke version on `main`.

## 6. Cost budget

Token-economy details live in [`TOKEN_BUDGET.md`](TOKEN_BUDGET.md). Read it before writing your first prompt. Quick caps to enforce in code from the start:

- Cap total LLM calls per dataset run at ~150.
- Cap PM calls at ~30 (they use the most expensive model).
- Cap Loop B (4 → 3) iterations at 3.
- Cap any single agent call's output at ~2000 tokens.
- Use a mid-tier model for Data Engineer and Developer; reserve top-tier for PM and (optional) Validator. CrewAI lets you set `llm` per-agent; LangGraph lets you choose per-node.

## 7. Failure modes (read before you start coding)

These are the bugs you will hit. They are not rare:

- **Hallucinated columns.** The modeller writes `df["FamilySize"]` where no such column exists. Mitigation: a schema checker tool the Developer or Validator calls before any model is fit.
- **Train/test leakage.** Encoder/scaler fit on train+test concatenated. Mitigation: enforce sklearn `Pipeline` shape in the Data Scientist's prompt; an independent "first-half vs second-half" sanity test.
- **Submission schema mismatch.** Wrong column names, dtype, or row count. Mitigation: validate the final `submission.csv` against the competition's `sample_submission.csv` *as schema* before writing.
- **Infinite replanning loops.** The PM bounces between phases 3 and 4 forever. Mitigation: hard caps on phase visits and on each loop contour, enforced in the orchestrator.
- **Cost runaway.** A single run accidentally costs many dollars. Mitigation: per-run token cap enforced in code, not just requested in the prompt.
- **Agent-writes-pseudocode.** Code that looks right but doesn't run. Mitigation: every agent that emits code must execute it through the Python sandbox and report the captured output, not the generated text.
- **Phase-jumping.** Data Scientist starts modelling without waiting for Data Preparation to finish. Mitigation: the orchestrator refuses to dispatch to an agent for a substep whose prerequisites aren't satisfied.
- **JSON-output-parse failures.** A model occasionally returns text wrapped in Markdown fencing, with trailing commas, or with control characters. Mitigation: the Developer's JSON-repair routine plus a strict output mode in your framework / API.

## 8. What the final architecture write-up must contain

The architecture document you submit alongside the code should include:

1. A diagram showing your agents, their tools, and the message flow.
2. **Framework choice and why** — what you picked and what the alternatives would have cost.
3. For each agent: ownership in CRISP-DM, prompt template, inputs and outputs in terms of state fields, tools attached.
4. The state schema you actually used (whatever the framework calls it).
5. How each of the four loop contours is implemented (as conditional transitions, task hand-offs, whatever).
6. The LLM model assigned to each agent and provider (OpenAI vs DeepSeek) and why.
7. **A failure-modes log** — what broke, how you knew, what you did about it. The most important section. An honest one-page failure log beats a five-page success story.
8. A reproducibility appendix: exact commands to run on a fresh checkout.
