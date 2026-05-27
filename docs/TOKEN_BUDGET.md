# Token Budget — How to Not Burn the API Allowance

A naïve multi-agent system burns tokens. With 5 agents, 24 substeps, and 4 loops, the call graph can easily multiply. Read this **before** writing your first agent. The techniques below are listed roughly in order of "biggest savings for least effort". Apply at least the first six from the start.

> **Rule of thumb.** A single end-to-end run on Titanic should fit comfortably under **$0.50** of API spend, and well under 200 000 total tokens. If your numbers blow past this during development, stop and revisit this document before adding more features.

---

## 1. Hard caps in code, not in prompts (mandatory)

The single most important rule. Don't ask the model nicely to keep things short — *enforce* it. The scaffold's `llm.py` already has a `MAX_TOKENS_PER_RUN` env-var cap. Add three more:

```python
# In llm.py or your agent base class
MAX_OUTPUT_TOKENS_PER_CALL = 1500      # passed as max_tokens
MAX_CALLS_PER_AGENT_PER_PHASE = 4
MAX_INPUT_TOKENS_PER_CALL = 8000       # truncate or summarise before sending
```

If a cap is exceeded, *raise an exception* — don't just log. A loud failure early is far cheaper than a silent burn later.

## 2. Prompt caching (free, automatic if you structure things right)

OpenAI (and DeepSeek, in the same way) automatically caches the front of any prompt over a certain size, with a substantial discount on the cached portion and a TTL of a few minutes. The cache hits only if the *prefix* is **byte-identical** across calls. To benefit:

- Put your **stable system prompt at the very front** of every agent's call — never interpolate the run timestamp, the case ID, or anything that varies.
- Put variable content (state excerpts, the user message, recent log lines) **at the back**.
- Keep each agent's system prompt the same shape across all 24 substeps; differentiate by user message instead of by system message.

A single agent making 8 calls per run with a 1500-token system prompt will see roughly half the cost vanish from caching alone.

## 3. Don't pass data — pass schemas and summaries

The biggest unforced error in multi-agent data science is dropping `df.to_string()` or `df.head(20).to_markdown()` into a prompt. Don't.

| Instead of | Send |
|---|---|
| `df.head(20).to_string()` | `df.describe(include="all").to_dict()` + `df.dtypes.to_dict()` |
| A full CSV | The output of a small profiling script (column names, dtypes, missingness %, top-k values per category) |
| A list of 1000 strings | A random sample of 5, plus distribution stats |
| A pandas DataFrame with 8000 tweets | `{n_rows: 8000, vocabulary_size: 21500, avg_tokens: 16.4, n_classes: 2, class_balance: {...}}` |

The agent rarely needs the data itself; it needs *a typed description of the data* so it can decide which Python to write. The Python it writes will then touch the real data through `PythonExec`.

## 4. Send each agent only the slice of state it needs

The CRISP-DM state has many fields. Passing the entire state to every agent on every call is a waste.

Use the `state.view_for(agent_name) -> dict` helper that the scaffold ships:

| Agent | What it needs from state |
|---|---|
| PM | `phase`, `substep`, last few log entries (truncated), latest data_quality_report summary, latest model assessment summary |
| Domain Expert | `config`, BU outputs so far, DU outputs that are filled |
| Data Engineer | `config`, paths to raw data, `bu.data_mining_goals`, DU outputs filled so far |
| Data Scientist | `config`, paths to cleaned data, `bu.data_mining_goals`, `du.data_description_report`, recent candidate models |
| Developer | `config`, paths needed for the current task, `md.chosen_model` (if Phase 6) |
| Validator | models, chosen_model, data_description_report, test_design |

This alone can cut input tokens by half on later-phase calls.

## 5. Truncate the log before feeding it to any LLM

`state.log` grows unboundedly during a run. Don't pass it whole. When the PM needs context:

```python
def summarise_log_for_prompt(log: list[LogEntry], max_entries: int = 8) -> str:
    recent = log[-max_entries:]
    lines = []
    for e in recent:
        msg = e.message[:200]  # truncate per-entry
        lines.append(f"[{e.agent}] {msg}")
    return "\n".join(lines)
```

If older context matters (a hypothesis from Phase 1 that affects Phase 4), put it in a *structured* state field — not in the log. The log is for traceability, not memory.

## 6. Truncate tool output (especially Python stdout)

`PythonExec` results can be megabytes. Before passing back to the LLM:

```python
def trim_output(s: str, max_chars: int = 2000) -> str:
    if len(s) <= max_chars:
        return s
    head = s[: max_chars // 2]
    tail = s[-max_chars // 2 :]
    return f"{head}\n\n... [{len(s) - max_chars} chars omitted] ...\n\n{tail}"
```

Apply to both stdout and stderr. The agent doesn't need the middle 4 MB of a Pandas error stack trace.

## 7. Use JSON mode and tight output schemas

Free-form prose is verbose. For every structured decision (the PM's `{action, target_substep, reason}`, the Domain Expert's hypotheses list, the Data Scientist's model selection) use:

- `response_format={"type": "json_object"}` in the call (already wired into the scaffold's `llm.py`).
- A short JSON schema described in the system prompt.
- `max_tokens` set to a tight bound (200–500 for short decisions, ~1500 for longer reports).

JSON mode also stops the model from prefacing answers with "Certainly! Here's the…" — three free tokens saved on every call.

## 8. Model tiering

Don't use the most capable model for everything. The scaffold's `llm_for(agent_name)` does this:

| Agent | Default tier | Why |
|---|---|---|
| PM | top | Planning and loop-firing decisions are high-stakes |
| Validator (if used) | top | Adversarial reasoning |
| Data Scientist | top or mid | Modelling decisions benefit from reasoning |
| Domain Expert | mid | Mostly RAG-grounded |
| Data Engineer | mid | Code gen against a known schema |
| Developer | mid | Code gen, packaging, debugging |

For the cheapest agents — embedding-driven RAG calls, simple classifications, code-gen with a clear template — a `mini`-tier model is fine. If the OpenAI budget tightens, route the high-volume code-gen agents (Data Engineer, Developer) to **DeepSeek-V3** through the same wrapper; it's roughly an order of magnitude cheaper on input tokens.

## 9. RAG instead of in-prompt stuffing

The Domain Expert agent's reference corpus (CRISP-DM excerpts, dataset notes, Abhishek Thakur chapters) should **not** be in the system prompt. Index it once:

- Chunk into ~200-token pieces.
- Embed with `text-embedding-3-small` (very cheap).
- At call time, retrieve top-3 to top-5 chunks for the current query.
- Append them at the *back* of the user message (so they don't bust the system-prompt cache).

This is the difference between a 30 000-token Domain Expert prompt and a 3 000-token one.

## 10. Memoise expensive calls within a run

Within a single dataset run, the same factual question may be asked twice (e.g. "what does this column mean" in 2.2 and again in 3.3). Memoise on a hash of the prompt:

```python
@functools.lru_cache(maxsize=128)
def cached_chat(system_prompt: str, user_prompt: str, model: str) -> str:
    return llm.chat(system=system_prompt, user=user_prompt, ...).text
```

If you re-run on the same dataset during development, persist the cache to a JSON file keyed on `(model, prompt_hash)`. A simple `~/.cache/maads/calls.json` saves the bulk of LLM cost during development iteration.

## 11. Develop on data subsets

During development, default to **the first 200 rows** of each training file. Switch to the full set only for a real submission run. One line of code; large savings.

```python
TRAIN_SUBSET = int(os.environ.get("TRAIN_SUBSET", "200"))  # set to 0 for full
df = pd.read_csv(...).head(TRAIN_SUBSET) if TRAIN_SUBSET else pd.read_csv(...)
```

## 12. Batch embeddings

If you embed your RAG corpus piece by piece you'll make 50 API calls where one would do. The OpenAI embeddings endpoint takes a list of strings. Batch by 50–100 at a time.

## 13. Don't loop on rate-limit retries during cost runaway

`llm.py` retries on `RateLimitError` with backoff. That's fine. But if you ever hit a *quota* error (different from rate-limit), do not retry — halt the run. Quota errors mean the budget is actually gone; retrying is throwing tokens at a wall.

## 14. Hold a recorded backup run

Once you have a successful end-to-end run, record its full state JSON, logs, and submission files into the repo. If the LLM provider has an outage during a demo, you can replay this recording. Costs you one extra run; saves the demo.

---

## A simple per-run budget worksheet

Fill this in for each dataset run, *before* you start. If you cannot make these numbers add up, redesign — don't run.

| Agent | Expected calls | Avg input tokens (after cache) | Avg output tokens | Subtotal (input + output) |
|---|---|---|---|---|
| PM | 10 | 1500 | 200 | 17 000 |
| Domain Expert | 6 | 2500 | 600 | 18 600 |
| Data Engineer | 12 | 1800 | 800 | 31 200 |
| Data Scientist | 8 | 2200 | 600 | 22 400 |
| Developer | 8 | 1500 | 500 | 16 000 |
| **Total** | **44** | — | — | **≈ 105 000 tokens / run** |

At current OpenAI mini-tier prices, that's roughly **$0.05–$0.10 per Titanic run**. Three datasets, ten development iterations: well under $5 of budget total. Being off by 5× lands you under $25 — survivable. Being off by 50× because you forgot to truncate stdout is how budgets die.

## Day-one checklist before your first commit

- [ ] `MAX_TOKENS_PER_RUN` in `.env` is set
- [ ] Every agent has a max_output_tokens cap
- [ ] System prompts are identical across substeps (cache-friendly)
- [ ] State views are agent-specific (`state.view_for(agent_name)`)
- [ ] `PythonExec` output is trimmed before being shown to the LLM
- [ ] `df.describe().to_dict()` is your default; raw CSVs are not in prompts
- [ ] Token spend per agent is logged after every call

If any of these is unchecked, fix it before writing the next agent.
