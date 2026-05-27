# Architecture

This document is a **detailed suggestion**, not a specification. The team is expected to adapt every decision below to its taste and defend the result in the final architecture write-up. What is non-negotiable is that all five (or optionally six) agents and all four CRISP-DM loops are implemented and observable.

## 1. The shared state

Everything the agents share — current phase, current substep, artefacts produced, decisions made, the active back-edge if any — lives in one object. There is no implicit memory. Every agent reads from this object and writes back to it.

A suggested shape (Pydantic model; the scaffold ships a working version of this). The naming follows the CRISP-DM 1.0 Reference Model outputs verbatim, so a prompt can reference an output by its canonical name and the code matches:

```python
class CrispDMState(BaseModel):
    case_id: str                       # arbitrary, comes from the config file
    config: CaseConfig
    phase: Phase                       # 1..6
    substep: str                       # "1.1", "1.2", ..., "6.4"
    loop_history: list[LoopEvent]      # each entry has label "A" | "B" | "C" | "D"
    halted: bool
    halt_reason: str | None

    bu:  BusinessUnderstanding         # 1.1 .. 1.4 outputs
    du:  DataUnderstanding             # 2.1 .. 2.4 outputs
    dp:  DataPreparation               # 3.1 .. 3.5 outputs (+ dataset, dataset_description)
    md:  Modeling                      # 4.1 .. 4.4 outputs (+ models, chosen_model)
    ev:  Evaluation                    # 5.1 .. 5.3 outputs
    dep: Deployment                    # 6.1 .. 6.4 outputs (+ submission_path)

    log: list[LogEntry]                # append-only; trim before showing to LLM
    token_spend: dict[str, int]        # per-agent token totals
```

Each nested sub-model has one field per output named in the CRISP-DM reference model. For instance:

```python
class BusinessUnderstanding(BaseModel):
    # 1.1 Determine Business Objectives
    background: str | None
    business_objectives: str | None
    business_success_criteria: str | None
    # 1.2 Assess Situation
    inventory_of_resources: dict | None
    requirements_assumptions_constraints: dict | None
    risks_and_contingencies: list[str]
    terminology: dict[str, str]
    costs_and_benefits: dict | None
    # 1.3 Determine Data Mining Goals
    data_mining_goals: str | None
    data_mining_success_criteria: str | None
    # 1.4 Produce Project Plan
    project_plan: list[str]
    initial_assessment_of_tools_and_techniques: dict | None
```

Three design rules:

- **Append-only logs.** `log`, `loop_history`, and `md.models` only get appended to. This makes a run replayable and inspectable.
- **No agent reads another agent's prompt.** They only read state. This keeps the agents loosely coupled — any one is swappable.
- **No agent receives the whole state.** Every agent uses `state.view_for(agent_name)` to get a minimal slice. Sending the entire state to every LLM call burns the token budget — see [`TOKEN_BUDGET.md`](TOKEN_BUDGET.md).

## 2. The agents

Each agent is a Python class implementing two methods: `plan(state) -> Plan` (what would you do next) and `act(state) -> StateDelta` (do it). The `Plan` from `plan()` is what the Project Manager evaluates when deciding whether to let this agent run, whether to fire a loop, or whether to move on.

### 2.1 Project Manager

- **CRISP-DM ownership**: phase transitions, all four loop contours (A, B, C, D). Directly produces 1.4 (Project Plan), 5.2 (Review Process), 5.3 (Determine Next Steps).
- **Tools**: read state, write phase/substep, query each agent's `plan()`, fire a loop, halt the run.
- **Inputs**: a minimal state view (`state.view_for("pm")`) containing the current phase/substep, the latest data-quality blockers, the latest model assessment, the loop history, and a trimmed recent log.
- **Outputs**: a phase-transition decision; a loop-firing decision; a final go/no-go.
- **Prompt structure** (high level):
  - Role: "You are the PM of a CRISP-DM run. Your job is to walk the cycle and fire loops when warranted."
  - Phase rules: the substeps for the current phase, the exit conditions, and the loop conditions copied from the README's loop table.
  - Decision schema: must return JSON `{action, target_substep, loop_to_phase, reason}`.
- **Anti-loop guardrails**: hard cap on total phase transitions (e.g. 12). Hard cap on visits to each phase (e.g. 3). The cap fires before the model can talk itself into a fifth try.

### 2.2 Domain Knowledge Expert

- **CRISP-DM ownership**: 1.1, 1.2, 1.3; contributes to 2.1, 2.2, 2.3.
- **Tools**: RAG retriever over a small corpus (CRISP-DM spec excerpts, problem-specific notes, prior Kaggle write-ups for similar problems); read state; write Business Understanding fields and contribute to Data Understanding.
- **Prompt structure**:
  - Role: "You translate a business problem and a dataset into well-formed data-mining goals and hypotheses."
  - Inputs from state: `config.problem_statement`, `raw_data_paths`, anything the Data Engineer has already put into `du`.
  - Output schema: structured JSON with each Phase-1 substep filled in, plus a list of typed feature hypotheses for Phase 2.

### 2.3 Data Engineer

- **CRISP-DM ownership**: full 2.1, 2.2, 2.4 and full 3.1–3.5.
- **Tools**: Python execution sandbox (already in the scaffold), file IO, schema introspection helpers.
- **Workflow**:
  - 2.1: load files according to the paths declared in `config.data`.
  - 2.2: emit a structured `data_description_report` (shapes, dtypes, missingness, cardinality, simple statistics).
  - 2.4: produce a `data_quality_report` with two lists, "blockers" and "tolerable". Blockers trigger Loop A.
  - 3.x: write the cleaned `train.parquet` and `test.parquet`. 3.3 Construct is where most feature engineering happens.
- **Hard rule**: every code block the Data Engineer emits must be **executed** by the Python sandbox; the agent only commits to state what survived execution.

### 2.4 Data Scientist

- **CRISP-DM ownership**: contributes to 2.3 (modelling-lens EDA); owns 4.1–4.4 and 5.1.
- **Tools**: Python sandbox with sklearn, xgboost, lightgbm, catboost.
- **Workflow**:
  - 4.1: pick a modelling technique from a constrained menu, with a justification logged.
  - 4.2: produce a test design (e.g. 5-fold stratified CV for classification, 5-fold KFold for regression, plus a held-out final score). The test design is stored in state and *re-used* across the inner loop.
  - 4.3 / 4.4: train, score on the test design, append `ModelRun` records to `md.models`. The PM decides whether Loop B fires; this agent does not.
- **Constraint**: cap inner-loop iterations at three to keep token cost bounded.

### 2.5 Developer

- **CRISP-DM ownership**: cross-cutting through 2–5; full 6.1–6.4.
- **Tools**: Python sandbox, file IO, packaging, plus the debugging toolkit described below.
- **Workflow during Phases 2–5**: writes whatever utility code other agents need (a custom encoder, a feature builder, an integration helper) and — critically — **handles errors** when other agents' code fails to execute.
- **Workflow during Phase 6**:
  - 6.1: write the submission file from `chosen_model`'s predictions; validate against the competition's `sample_submission.csv` shape before writing.
  - 6.2: a short monitoring plan (text — what should be monitored in production).
  - 6.3: produce `final_report.md` from the state log. This becomes the input to the paper.
  - 6.4: write the project review (what worked, what didn't), into `dep.experience_documentation`.

#### Debugging toolkit

When any agent's `PythonExec` call fails, the Developer is invoked to diagnose. Build at least these capabilities:

1. **Error classifier** — given stderr from a failed `PythonExec`, classify into one of: schema error (column not found, wrong dtype), shape/dimension error (sklearn shape mismatch), leakage signal, library version mismatch, OOM, timeout, syntax error, JSON-output-parse error, "other". Each category has a known fix template.
2. **Schema check** — given a code snippet and `state.du.data_description_report`, verify every column referenced exists in the schema; emit a list of unresolved names.
3. **Iterative re-execution** — given a failure, propose a fix, re-run, capture the new outcome. Cap at three attempts per call; on the third failure, surface the issue to the PM as a stuck-on-substep diagnostic so the PM can decide to fire Loop B or halt.
4. **JSON repair** — when another agent's structured output fails to parse, attempt a single repair (trim Markdown fencing, fix unbalanced braces, escape quoted strings). On failure, request a re-emission with a stricter prompt.
5. **Bisection** — when a pipeline produces wrong-looking numbers, the Developer can run the pipeline up to step N and compare with a reference (e.g. shape-check at every step). Useful for catching silent feature-engineering bugs.

The Developer is the agent that prevents "five agents each fail in their own way" from turning into ten interacting failure modes. Without it, the others must each handle their own errors, and they will not handle them as well.

### 2.6 Optional sixth: Validator

If the team chooses a 6-agent design, the Validator owns 4.4 and 5.1 separately, with its own context, its own prompt template, and an adversarial mandate ("try to reject"). The Data Scientist then loses 4.4/5.1 ownership. This is the "two-keys-to-launch" pattern; it reduces silent failures, at the cost of one more agent's worth of prompt engineering.

## 3. The orchestrator and the loop contours

The PM doesn't itself execute Phases 1–6 — it invokes the responsible agent and reads what they wrote into state. The control loop is roughly:

```
while not state.halted:
    plan = pm.plan(state)              # PM looks at state, decides next move
    if plan.action == "halt":
        break
    if plan.action == "advance":
        next_agent = router(plan.target_substep)
        next_agent.act(state)
    elif plan.action == "loop_back":
        state.record_loop(label, from_phase, to_phase, reason)
        state.phase = plan.target_phase
    if hard_limits_exceeded(state):
        force_halt(state)
```

The `router` is a tiny mapping from "substep ID" to "which agent owns this substep" (drawn from the table in the README). It is *not* an LLM call — it's a lookup.

### Implementing the four loops

The loop labels match the README diagram (Loop A, B, C, D).

| Loop | Trigger | Implementation |
|---|---|---|
| **A — 2 → 1** | After 2.4, if `state.du.data_quality_report["blockers"]` is non-empty, or if Domain Expert hypotheses are contradicted by the actual schema in `state.du.data_description_report` | PM calls `state.record_loop(label="A", from_phase=2, to_phase=1, reason=...)` and sets `state.phase=1, state.substep="1.3"`. Domain Expert re-runs 1.3 only, not the full Phase 1. |
| **B — 4 → 3** | After 4.4, if the latest `ModelRun.assessment` flags a specific preparation deficit *and* CV score is below threshold | PM records `LoopEvent(label="B", ...)` and sets `state.phase=3` at the affected substep (often 3.3 Construct). Cap: 3 iterations. |
| **C — 5 → 1** | After 5.1, if `state.ev.assessment_of_dm_results` indicates the business success criterion is not met *and* Loop A has not already fired twice | PM records `LoopEvent(label="C", ...)` and sets `state.phase=1, state.substep="1.3"`. Triggers a fundamental rethink of data-mining goals. |
| **D — 6 → 1** | Optional, stretch goal | After 6.4, the experience documentation is appended to the *next dataset*'s RAG corpus. Implements cross-dataset learning. |

A run that fires zero loops on Titanic but at least one loop on House Prices or Disaster Tweets is a sign the system is actually thinking, not just executing.

## 4. Tools

You will build / wire up roughly:

| Tool | Status in scaffold | Purpose |
|---|---|---|
| `PythonExec` | **Working.** Subprocess-based, captures stdout/stderr, enforces timeout. | Sole way agents touch data. |
| `LLM` | **Working.** Thin wrapper over OpenAI / DeepSeek with retries and per-agent cost accounting. | Sole way agents call the model. |
| `FileIO` | **Working.** Read/write helpers under a per-run artifact directory. | Persisting reports, model artefacts, submissions. |
| `RAGRetriever` | **Stubbed.** Empty class; the team builds the indexer and retriever. | Domain Expert's grounding. |
| `SchemaChecker` | **Not present.** | Validates that any column referenced in agent output actually exists in the data. *Recommended early.* |
| `LeakageCheck` | **Not present.** | Validator-side; detects target leakage and train/test contamination. |
| `ErrorClassifier` | **Not present.** | Developer-side; categorises `PythonExec` failures and proposes fix templates. |
| `KaggleClient` | **Not present.** | Optional: submit and read back the public-leaderboard score. The Kaggle CLI is enough; wrapping it as a tool is a nice-to-have. |

## 5. Communication pattern

For the first working version, use a **hub-and-spoke** pattern: agents only talk to / through the PM. Agents do not call each other directly. This makes the system easy to debug (every transition is logged in one place) and easy to reason about (only the PM fires loops).

If, after the first end-to-end pipeline works, the team wants to experiment with peer-to-peer (e.g. Data Engineer ↔ Data Scientist negotiating feature engineering directly), do it on a branch and keep the hub-and-spoke version on `main`.

## 6. Cost budget

Token-economy details live in [`TOKEN_BUDGET.md`](TOKEN_BUDGET.md) — read it before writing the first prompt. Quick caps to enforce in code from the start:

- Cap total LLM calls per dataset run at ~150.
- Cap PM calls at ~30.
- Cap Loop B (4 → 3) iterations at 3.
- Cap any single agent call's output at ~2000 tokens.
- Use a mid-tier model for Data Engineer and Developer; reserve top-tier for PM and (optional) Validator.

## 7. Failure modes (read before you start coding)

These are the bugs you will hit. They are not rare:

- **Hallucinated columns.** The modeller writes `df["FamilySize"]` where no such column exists. Mitigation: a `SchemaChecker` tool, called by the Developer or Validator before any model is fit.
- **Train/test leakage.** Encoder/scaler fit on train+test concatenated. Mitigation: enforce sklearn `Pipeline` shape in the Data Scientist's prompt; check independently with a "first-half vs second-half" sanity test.
- **Submission schema mismatch.** Wrong column names, dtype, or row count. Mitigation: the Developer validates the final `submission.csv` against the competition's `sample_submission.csv` *as schema* before writing.
- **Infinite replanning loops.** The PM bounces between phases 3 and 4 forever. Mitigation: hard caps on phase visits and on each loop contour, enforced in the orchestrator.
- **Cost runaway.** A single run accidentally costs many dollars. Mitigation: per-run token cap enforced in `llm.py`, not just requested in the prompt.
- **Agent-writes-pseudocode.** Code that "looks right" but doesn't run. Mitigation: every agent that emits code must execute it through `PythonExec` and report the captured output, not the generated text.
- **Phase-jumping.** Data Scientist starts modelling without waiting for Data Preparation to finish. Mitigation: the router refuses to dispatch to an agent for a substep whose prerequisites aren't satisfied in state.
- **JSON-output-parse failures.** A model occasionally returns text wrapped in Markdown fencing, with trailing commas, or with control characters. Mitigation: the Developer's JSON-repair routine, plus `response_format={"type":"json_object"}` on every structured call.

## 8. What the final architecture write-up must contain

The architecture document you submit alongside the code must include:

1. A diagram showing your agents, their tools, and the message flow.
2. For each agent: ownership in CRISP-DM, prompt template, inputs and outputs in terms of state fields.
3. The state schema you actually used.
4. How each of the four loop contours is implemented in code.
5. The LLM model assigned to each agent (and provider — OpenAI or DeepSeek) and why.
6. A failure-modes log: what broke, how you knew, what you did about it. **This is the most important section.** An honest one-page failure log beats a five-page success story.
7. A reproducibility appendix: exact commands to run on a fresh checkout.
