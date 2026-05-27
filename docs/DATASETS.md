# Demonstration suite — three Kaggle competitions

The system you are building is **general** — it should run on any Kaggle competition that fits the small configuration schema below. This case requires that you demonstrate the system on three specific competitions, chosen for diversity along the axes that matter for an agentic data science system. They are not the system's scope; they are its acceptance test.

| # | Competition | Problem | Data type | What it stresses |
|---|---|---|---|---|
| 1 | Titanic | Binary classification | Tabular, mixed types, small | Vanilla end-to-end. The "does anything work at all" sanity check. |
| 2 | House Prices | Regression | Tabular, 79 features, ordinals-as-strings | The Data Engineer's preparation logic. A different metric and loss. |
| 3 | Disaster Tweets | Binary classification | Text (NLP) | A fundamentally different feature space; tests whether the system generalises beyond tabular without dataset-specific code. |

A run that succeeds on Titanic and House Prices but cannot even start on Disaster Tweets means the architecture has assumed "tabular" somewhere it should not have. Catching this is part of the value of the exercise.

---

## 1. Titanic — Machine Learning from Disaster

- **Link**: <https://www.kaggle.com/competitions/titanic>
- **Type**: binary classification
- **Target**: `Survived` (0 / 1)
- **Metric**: accuracy
- **Size**: 891 training rows, 418 test rows, 11 features
- **Why first**: the canonical first Kaggle problem. Mixed dtypes (numeric, categorical, free text), real-but-bounded missingness. The Domain Expert can lean on a century of well-documented context for the 1.1 / 1.2 substeps.
- **Baselines to beat**:
  - Random guess: 0.50
  - "All female survive, all male die": ~0.766
  - Decent tuned GBDT: ~0.78–0.82
  - Top of leaderboard: ~0.84 (anything higher usually means leakage from the public test set)
- **System target**: ≥ 0.77 accuracy on the public leaderboard. Higher is gravy.
- **Things to watch**:
  - `Cabin` is ~77% missing — agent must decide drop vs. deck-letter extraction.
  - `Name` has the title (Mr, Mrs, Master, etc.) — feature engineering target for the Data Engineer.
  - `Age` is partially missing — imputation choice matters.

## 2. House Prices — Advanced Regression Techniques

- **Link**: <https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques>
- **Type**: regression
- **Target**: `SalePrice`
- **Metric**: RMSE on `log(SalePrice)`
- **Size**: 1460 training rows, 1459 test rows, **79 features** (numeric + ordinal-as-string + nominal)
- **Why second**: this is the stress test for Data Preparation. 79 features with non-trivial missingness patterns, many ordinal-but-string variables (`ExterQual: Ex/Gd/TA/Fa/Po`), and meaningful feature-engineering opportunities (total square footage, age at sale, time since remodel). If the Data Engineer's profiling step is weak, this dataset exposes it immediately.
- **Baselines to beat**:
  - Mean predictor on log-price: ~0.40 RMSE
  - Naïve linear regression: ~0.18–0.20
  - Tuned GBDT with sensible preprocessing: ~0.12–0.14
  - Top of leaderboard: ~0.10 (often via stacking)
- **System target**: ≤ 0.15 RMSE on the public leaderboard. Lower is gravy.
- **Things to watch**:
  - The target is right-skewed; logging it before training is almost always the right call.
  - Several columns use "NA" as a category meaning "absence of the feature" (e.g. "no basement"), not a missing value. The Domain Expert can flag this from the data dictionary; the Data Engineer must respect it.
  - Many ordinals look nominal until the Domain Expert flags them.

## 3. Natural Language Processing with Disaster Tweets

- **Link**: <https://www.kaggle.com/competitions/nlp-getting-started>
- **Type**: binary classification
- **Target**: `target` (0 = not a real disaster, 1 = real disaster)
- **Metric**: F1 score
- **Size**: 7613 training rows, 3263 test rows, ~5 columns (`id`, `keyword`, `location`, `text`, `target`)
- **Why third**: this is the diversity test. The main feature is **free text**. If your agents have implicitly assumed `pandas.get_dummies()` solves all categorical encoding, this dataset will say no. The Data Engineer must either build a text representation (TF-IDF is fine; sentence-transformer embeddings via OpenAI's embeddings endpoint is also fine), or the Developer must wire up that capability for it.
- **Baselines to beat**:
  - Majority class: ~0.43 F1
  - TF-IDF + logistic regression with light cleaning: ~0.78–0.80 F1
  - Sentence-embedding-based classifier: ~0.82–0.84 F1
  - Top of leaderboard with fine-tuned transformer: ~0.85
- **System target**: ≥ 0.78 F1 on the public leaderboard. Higher is gravy.
- **Things to watch**:
  - `keyword` and `location` are weakly informative but often missing; decide explicitly.
  - The training set was scraped and contains some labelling noise — there are known label disagreements. Don't chase a 0.99 score; it doesn't exist on this dataset.
  - Don't fine-tune a transformer for this. It will eat the budget. TF-IDF or OpenAI embeddings + classical ML is the sweet spot.

---

## Adding a fourth competition

Because the system is general, adding a fourth (or fifth) competition is intended to be trivial. The flow:

1. Write a config file at `starter/configs/<your-case>.yaml` following the schema in the three existing configs. The required fields are:
   - `case_id`
   - `kaggle_competition` (the slug, e.g. `spaceship-titanic`)
   - `problem_statement`
   - `problem_type` (`binary_classification` | `multiclass_classification` | `regression`)
   - `target_column`, `id_column`, `evaluation_metric`
   - `data.train_csv`, `data.test_csv`, `data.sample_submission_csv` (paths relative to the repo root)
   - `feature_hints` (optional, may be empty)
   - `success_criterion: {metric, threshold}` — used by the Evaluation phase to decide whether Loop C fires
2. Download the data: `python -m maads data download --competition <slug>`
3. Run: `python -m maads run --config configs/<your-case>.yaml`

The agents themselves should be entirely untouched.

## Across-dataset constraints

The same agent code must run all three (or four, or five). This means:

- **No dataset-specific code in agents.** Anything dataset-specific lives in the config file.
- **No dataset-specific prompts.** Agent system prompts are identical across runs. Only the state's `config` differs.
- **One command per dataset**:
  ```
  python -m maads run --case titanic
  python -m maads run --case house_prices
  python -m maads run --case disaster_tweets
  ```

If you find yourself writing `if state.case_id == "titanic":` in agent code, lift the concept into config or into the Domain Expert's RAG corpus instead.

## Acquiring the data

The scaffold ships a `data/` directory with `.gitkeep` files. The team is expected to either:

- Use the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) to download into `data/<case>/`. (Requires each user to have accepted the competition rules on Kaggle.) The `python -m maads data download` subcommand wraps the CLI.
- Or commit the raw CSVs to the repo (small enough — Titanic ~60 KB, House Prices ~960 KB, Disaster Tweets ~1.5 MB), respecting each competition's terms of use.

The first option is cleaner. `python -m maads data download --case <name>` handles it once Kaggle credentials are configured. The same command accepts `--competition <slug>` for any Kaggle competition not yet in the bundled configs.
