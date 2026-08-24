# BotCompletion

Code, evolved prompts, and training-set curation pipeline for:

> **BotCompletion: Optimizing Context and Instructions for LLM-Based Bot Detection**
> Pedro Sanvido and Isabel Manssour — *The 18th International Conference on
> Advances in Social Networks Analysis and Mining (ASONAM 2026)*

## Repository layout

```
twibot/                     Python package
    context.py              post selection + DSPy example construction
    gepa_metric.py          structured-feedback GEPA metric
    account_classifier.py   heuristic account-type classifier
    dspy_components/        DSPy signatures and helpers
    cli/embed_tweets.py     Jina-V3 embedding, centrality sort, K-Means clustering
    utils/, config.py, results_db.py
scripts/
    preprocessing/          raw TwiBot-22 → experiment files (4 steps; see data/README.md)
    curation/               stratified training-set curation (select_*, verify_*, assemble_v4)
    run_baseline_trainset.py  baseline over the train pool (feeds curation)
    run_baseline_15configs.py unevolved baseline across 15 (mode, k) configs
    run_gepa.py             GEPA optimization, six paper configurations
    eval_gepa_table.py      3-pass evaluation of the evolved prompts
    eval_handwritten_prompt.py  no-GEPA thresholds ablation
    run_budget_sweep.py     post-budget sensitivity sweep
    plot_budget_sweep.py    paper figure (small multiples)
prompts/                    the six evolved instructions + frozen hand-written prompt
data/                       built locally from TwiBot-22 (see data/README.md)
configs/twibot22.yaml       embedding/dataset defaults
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # or: uv venv && source .venv/bin/activate
pip install -r requirements.txt                      # or: uv pip install -r requirements.txt
```

Create a `.env` at the repository root:

```
OPENROUTER_API_KEY=sk-or-...
```

All model calls go through OpenRouter. The task model defaults to
`openai/gpt-oss-120b` (temperature 1.0, high reasoning effort); reflection
models default to Claude Opus 4.6 and Gemini 3.1 Pro. Both can be overridden
with environment variables, so the pipeline can run with your own models:

```bash
export TASK_MODEL="openrouter-model-id"        # e.g. qwen/qwen3-235b-a22b
export REFLECTION_MODEL="openrouter-model-id"  # applies to every GEPA run
export OPENROUTER_PROVIDER="provider-name"     # default: google-vertex
```

## 1. Data

TwiBot-22 content cannot be redistributed, so this repository contains **no
user data**; every input file is rebuilt locally from your own copy of the
dataset. See [`data/README.md`](data/README.md) for access instructions and
the four preprocessing steps.

## 2. Training-set curation

The GEPA training set is built from baseline misclassifications on BotSay's
2,694-user TwiBot-22 training pool:

```bash
# Baseline predictions over the train pool (SQLite: results/baseline_trainset/)
python scripts/run_baseline_trainset.py

# Stratified selection per account type (each writes data/gepa_trainsets/v4_curation/<stratum>.json)
python scripts/curation/select_sparse.py
python scripts/curation/select_verified.py
python scripts/curation/select_standard.py
python scripts/curation/select_high_url.py
python scripts/curation/select_high_rt.py
python scripts/curation/select_small_groups.py

# Assemble the final trainset (+ optional verify_* sanity reports)
python scripts/curation/assemble_v4.py

# Newest-first variant for the Latest strategy
python scripts/curation/build_latest_trainset.py   # needs TWIBOT22_RAW_DIR
```

The strata, thresholds, and error-category shares referenced in the paper
are in `twibot/account_classifier.py` and the `select_*` scripts.

## 3. GEPA optimization

```bash
python scripts/run_gepa.py
```

Runs the six paper configurations ({Latest@30, Centrality@10, Cluster} × {Opus,
Gemini}): 8,000 metric calls per run, reflection minibatch 3, Pareto candidate
selection with merge enabled, seed 42. Each run writes the evolved program,
a metadata JSON, and per-user predictions under `results/gepa/`. Expect
$10–30 in API spend per run, dominated by the reflection model.

## 4. Evaluation

```bash
python scripts/run_baseline_15configs.py    # unevolved baseline, 15 (mode, k) cells
python scripts/eval_gepa_table.py           # 3 passes per evolved prompt
python scripts/eval_handwritten_prompt.py   # no-GEPA thresholds ablation
python scripts/run_budget_sweep.py          # k ∈ {5,10,20,30,40} sweep
python scripts/plot_budget_sweep.py         # paper figure
```

`eval_gepa_table.py` evaluates the released prompts in `prompts/evolved/`
directly, so it can be run without re-running GEPA.


## Citation

```bibtex
@inproceedings{Sanvido2026botcompletion,
  author    = {Sanvido, Pedro and Manssour, Isabel},
  title     = {BotCompletion: Optimizing Context and Instructions for LLM-Based Bot Detection},
  booktitle = {Advances in Social Networks Analysis and Mining (ASONAM 2026)},
  publisher = {Springer},
  year      = {2026}
}
```

## License

[MIT](LICENSE). TwiBot-22 data is governed by its own terms; obtain it from
its authors (see [`data/README.md`](data/README.md)).
