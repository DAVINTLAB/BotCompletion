#!/usr/bin/env python3
"""GEPA instruction optimization for the six paper configurations (Sec. 3.4).

Each run evolves the detection instruction with GEPA against the curated
347-user training set, using the structured-feedback metric
(twibot/gepa_metric.py), then evaluates the optimized program on BotSay-340.

The BotDetector uses dspy.Predict instead of dspy.ChainOfThought: reasoning
models like GPT-OSS already reason internally (controlled here via the
`Reasoning: high` system-message directive injected by ReasoningAdapter).
Forcing them to also emit a visible "Reasoning:" section via ChainOfThought
likely produces a post-hoc rationalization that GEPA's reflection LM may
optimize against misleadingly.

Outputs per run (under results/gepa/):
  <exp_id>_optimized.json  — the evolved DSPy program (instruction)
  <exp_id>_meta.json       — run configuration and test metrics
  gepa_twibot22.db         — per-user predictions (SQLite)
"""

import os
import sys
import json
import hashlib
import sqlite3
import datetime
import logging
from pathlib import Path

# Mute noisy loggers
os.environ["LITELLM_LOG"] = "ERROR"
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("dspy").setLevel(logging.WARNING)

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
import dspy
from sklearn.metrics import f1_score, accuracy_score

from twibot.context import normalize_label, create_example_for_ablation
from twibot.results_db import AblationResultsDB
from twibot.dspy_components import BotDetectionSignatureWithInstruction
from twibot.gepa_metric import bot_detection_metric


# ─── Configuration ──────────────────────────────────────────────────────

TASK_MODEL = os.environ.get("TASK_MODEL", "openai/gpt-oss-120b")
PROVIDER = os.environ.get("OPENROUTER_PROVIDER", "google-vertex")
EXP_VARIANT = "release"

# The six paper runs: {Latest@30, Centrality@10, Cluster} × {Opus, Gemini}.
# Latest entries use a tweets-newest-first trainset (built by
# scripts/curation/build_latest_trainset.py) and test_latest.json; centrality
# and cluster use the standard curated trainset and test_clustered.json.
# The actual select_tweets_for_mode call uses 'centrality_top' for latest
# entries (taking front-of-list = newest, since the latest data files are
# pre-sorted newest-first).
# The Cluster + Gemini configuration optimizes at max_tweets=20 (deployment
# uses k=30); Cluster + Opus optimizes at max_tweets=30.
RUNS = [
    {"selection_mode": "latest",         "max_tweets": 30, "reflection_model": "anthropic/claude-opus-4.6"},
    {"selection_mode": "centrality_top", "max_tweets": 10, "reflection_model": "anthropic/claude-opus-4.6"},
    {"selection_mode": "cluster",        "max_tweets": 30, "reflection_model": "anthropic/claude-opus-4.6"},
    {"selection_mode": "latest",         "max_tweets": 30, "reflection_model": "google/gemini-3.1-pro-preview"},
    {"selection_mode": "centrality_top", "max_tweets": 10, "reflection_model": "google/gemini-3.1-pro-preview"},
    {"selection_mode": "cluster",        "max_tweets": 20, "reflection_model": "google/gemini-3.1-pro-preview"},
]

# A REFLECTION_MODEL environment variable overrides the reflection model for
# every run (deduplicating configurations that then only differ by it).
_reflection_override = os.environ.get("REFLECTION_MODEL")
if _reflection_override:
    for _run in RUNS:
        _run["reflection_model"] = _reflection_override
    _seen = set()
    RUNS = [r for r in RUNS if tuple(sorted(r.items())) not in _seen
            and not _seen.add(tuple(sorted(r.items())))]

DATA_ROOT = project_root / "data"
DATASET = "twibot22"

# Standard paths: 347-user curated trainset, BotSay-340 test split.
TRAINSET_PATH_STD     = DATA_ROOT / "gepa_trainsets" / "gepa_trainset_twibot22_v4.json"
TESTSET_PATH_STD      = DATA_ROOT / "twibot22" / "test_clustered.json"

# Latest paths: tweets pre-sorted newest-first per user.
TRAINSET_PATH_LATEST  = DATA_ROOT / "gepa_trainsets" / "gepa_trainset_twibot22_v4_latest.json"
TESTSET_PATH_LATEST   = DATA_ROOT / "twibot22" / "test_latest.json"

# Module-level globals (swapped per-run by run_one).
TRAINSET_PATH = TRAINSET_PATH_STD
TESTSET_PATH  = TESTSET_PATH_STD
VALSET_PATH   = TRAINSET_PATH_STD  # GEPA candidate selection uses the trainset

REFERENCE_DATE = datetime.datetime(2022, 1, 26, tzinfo=datetime.timezone.utc)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

OUTPUT_DIR = project_root / "results" / "gepa"
SEED = 42


# ─── Custom Chat Adapter ────────────────────────────────────────────────

class ReasoningAdapter(dspy.ChatAdapter):
    """Injects 'Reasoning: high' suffix into system messages for GPT-OSS."""

    def format_system_message(self, signature) -> str:
        base = super().format_system_message(signature)
        return f"{base}\n\nReasoning: high"


# ─── Data Loading ────────────────────────────────────────────────────────

def load_trainset(selection_mode: str, max_tweets: int) -> list[dspy.Example]:
    """Load curated trainset and convert to DSPy Examples."""
    with open(TRAINSET_PATH) as f:
        raw = json.load(f)
    examples = []
    for user in raw:
        ex = create_example_for_ablation(
            user, max_tweets, selection_mode, REFERENCE_DATE, DATE_FORMAT
        )
        examples.append(ex)
    return examples


def load_valset(selection_mode: str, max_tweets: int) -> list[dspy.Example]:
    """Load valset for GEPA candidate selection."""
    with open(VALSET_PATH) as f:
        raw = json.load(f)
    examples = []
    for user in raw:
        ex = create_example_for_ablation(
            user, max_tweets, selection_mode, REFERENCE_DATE, DATE_FORMAT
        )
        examples.append(ex)
    return examples


def load_testset(selection_mode: str, max_tweets: int) -> tuple[list[dspy.Example], list[str]]:
    """Load test set and convert to DSPy Examples. Returns (examples, user_ids)."""
    with open(TESTSET_PATH) as f:
        raw = json.load(f)
    examples = []
    user_ids = []
    for i, user in enumerate(raw):
        uid = user.get("id", user.get("ID", str(i)))
        user_ids.append(uid)
        ex = create_example_for_ablation(
            user, max_tweets, selection_mode, REFERENCE_DATE, DATE_FORMAT
        )
        examples.append(ex)
    return examples, user_ids


# ─── DSPy Program ────────────────────────────────────────────────────────

class BotDetector(dspy.Module):
    def __init__(self):
        # dspy.Predict instead of ChainOfThought — let the reasoning model think
        # internally without forcing a visible reasoning trace in the output.
        self.classify = dspy.Predict(BotDetectionSignatureWithInstruction)

    def forward(self, **kwargs):
        return self.classify(**kwargs)


# ─── Experiment ID ────────────────────────────────────────────────────────

def make_experiment_id(selection_mode: str, max_tweets: int, reflection_model: str) -> str:
    key = f"gepa_v4trainset_{EXP_VARIANT}_{DATASET}_{TASK_MODEL}_{PROVIDER}_{selection_mode}_{max_tweets}_{reflection_model}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


# ─── Evaluate on Test Set ─────────────────────────────────────────────────

def evaluate_program(
    program: dspy.Module,
    test_examples: list[dspy.Example],
    user_ids: list[str],
    db: AblationResultsDB,
    experiment_id: str,
) -> dict:
    """Run the optimized program on the test set and record results."""
    exec_pairs = [(program, ex) for ex in test_examples]
    parallel = dspy.Parallel(
        num_threads=16,
        max_errors=len(test_examples),
        return_failed_examples=True,
        provide_traceback=False,
    )
    results, failed, exceptions = parallel(exec_pairs)

    gold_labels, pred_labels = [], []
    correct, errors = 0, 0

    for idx, (uid, ex, result) in enumerate(zip(user_ids, test_examples, results)):
        gold = ex.label
        pred = None
        err = None

        if result is not None:
            pred = normalize_label(result.label)
            if pred is None:
                err = f"Invalid label: {result.label}"
        else:
            if idx in exceptions:
                err = str(exceptions[idx])[:200]

        db.add_prediction(experiment_id, uid, ex.username, gold, pred, err)

        if pred:
            gold_labels.append(gold)
            pred_labels.append(pred)
            if pred == gold:
                correct += 1
        if err:
            errors += 1

    acc = accuracy_score(gold_labels, pred_labels) if pred_labels else 0.0
    f1 = f1_score(gold_labels, pred_labels, average="macro", labels=["bot", "human"]) if pred_labels else 0.0

    return {"accuracy": acc, "f1": f1, "correct": correct, "errors": errors, "total": len(test_examples)}


# ─── Run One GEPA Experiment ──────────────────────────────────────────────

def run_one(
    selection_mode: str,
    max_tweets: int,
    reflection_model: str,
    api_key: str,
) -> None:
    # Swap module-level paths and effective selection mode for 'latest' runs.
    # The 'latest' files are pre-sorted newest-first; we use 'centrality_top'
    # internally so select_tweets_for_mode returns tweets[:k] (the newest k).
    global TRAINSET_PATH, TESTSET_PATH, VALSET_PATH
    if selection_mode == "latest":
        TRAINSET_PATH = TRAINSET_PATH_LATEST
        TESTSET_PATH  = TESTSET_PATH_LATEST
        VALSET_PATH   = TRAINSET_PATH_LATEST
        effective_mode = "centrality_top"
    else:
        TRAINSET_PATH = TRAINSET_PATH_STD
        TESTSET_PATH  = TESTSET_PATH_STD
        VALSET_PATH   = TRAINSET_PATH_STD
        effective_mode = selection_mode

    exp_id = make_experiment_id(selection_mode, max_tweets, reflection_model)
    label = f"{selection_mode}@{max_tweets} + {reflection_model.split('/')[-1]}"

    # Check if already done
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    db_path = OUTPUT_DIR / f"gepa_{DATASET}.db"
    db = AblationResultsDB(str(db_path))
    with sqlite3.connect(db.db_path) as conn:
        count = conn.execute(
            "SELECT COUNT(DISTINCT user_id) FROM predictions WHERE experiment_id = ?",
            (exp_id,),
        ).fetchone()[0]
    # Load test set size for skip check
    with open(TESTSET_PATH) as _tf:
        _test_size = len(json.load(_tf))
    if count >= _test_size:
        print(f"\n  SKIP {label} — {count} predictions exist")
        return

    print(f"\n{'='*70}")
    print(f"  GEPA: {label}")
    print(f"  Experiment ID: {exp_id}")
    print(f"{'='*70}")

    # Load data (using effective_mode for select_tweets_for_mode; latest data
    # files are pre-sorted newest-first so centrality_top returns the newest).
    print(f"  Trainset path: {TRAINSET_PATH.name}")
    print("  Loading trainset...")
    trainset = load_trainset(effective_mode, max_tweets)
    print(f"  Trainset: {len(trainset)} examples")

    print(f"  Valset path: {VALSET_PATH.name}")
    print("  Loading valset...")
    valset = load_valset(effective_mode, max_tweets)
    print(f"  Valset: {len(valset)} examples")

    print(f"  Testset path: {TESTSET_PATH.name}")
    print("  Loading testset...")
    test_examples, user_ids = load_testset(effective_mode, max_tweets)
    print(f"  Testset: {len(test_examples)} examples")

    # Configure task LM
    is_gpt_oss = TASK_MODEL.startswith("openai/gpt-oss")
    extra_body = {
        "provider": {
            "order": [PROVIDER],
            "allow_fallbacks": True,
            "require_parameters": True,
            "quantizations": ["bf16"],      # ensure quality across providers
        }
    }
    if not is_gpt_oss:
        # Non-GPT-OSS models use OpenRouter's native reasoning parameter.
        extra_body["reasoning"] = {"effort": "high"}
        extra_body["include_reasoning"] = True

    task_lm = dspy.LM(
        model=f"openrouter/{TASK_MODEL}",
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
        cache=False,
        temperature=1.0,
        max_tokens=32768,
        extra_body=extra_body,
    )
    # GPT-OSS uses a chat-message suffix trick; other models use native reasoning.
    adapter = ReasoningAdapter() if is_gpt_oss else dspy.ChatAdapter()
    dspy.configure(lm=task_lm, adapter=adapter)

    # Reflection LM at GEPA defaults; temperature/top_p follow Google's
    # recommended values for Gemini and are kept for both reflection models.
    reflection_lm = dspy.LM(
        model=f"openrouter/{reflection_model}",
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
        cache=False,
        temperature=1.0,
        top_p=0.95,
        max_tokens=8192,
    )

    # Run GEPA
    print("  Starting GEPA optimization...")
    program = BotDetector()
    print(f"  Fresh GEPA from bare signature (instructions={len(program.classify.signature.instructions)} chars)")

    optimizer = dspy.GEPA(
        metric=bot_detection_metric,
        max_metric_calls=8000,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=3,
        candidate_selection_strategy="pareto",
        num_threads=64,
        seed=SEED,
        track_stats=True,
        log_dir=str(OUTPUT_DIR / "logs" / exp_id),
    )

    optimized = optimizer.compile(program, trainset=trainset, valset=valset)
    print("  GEPA optimization complete.")

    # Save optimized program
    program_path = OUTPUT_DIR / f"{exp_id}_optimized.json"
    optimized.save(str(program_path))
    print(f"  Program saved to {program_path}")

    # Evaluate on test set
    print("  Evaluating on test set...")
    db.start_experiment(
        exp_id, DATASET, TASK_MODEL, "reasoning", selection_mode, max_tweets
    )
    metrics = evaluate_program(optimized, test_examples, user_ids, db, exp_id)
    db.complete_experiment(
        exp_id, metrics["accuracy"], metrics["f1"],
        metrics["total"], metrics["correct"], metrics["errors"],
    )

    print(f"  Results: Acc={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}  Errors={metrics['errors']}/{metrics['total']}")

    # Save run metadata
    meta = {
        "experiment_id": exp_id,
        "dataset": DATASET,
        "task_model": TASK_MODEL,
        "provider": PROVIDER,
        "reflection_model": reflection_model,
        "selection_mode": selection_mode,
        "max_tweets": max_tweets,
        "trainset_size": len(trainset),
        "testset_size": len(test_examples),
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "errors": metrics["errors"],
        "timestamp": datetime.datetime.now().isoformat(),
    }
    meta_path = OUTPUT_DIR / f"{exp_id}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    load_dotenv(project_root / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        return 1

    for run in RUNS:
        run_one(
            run["selection_mode"], run["max_tweets"], run["reflection_model"], api_key
        )

    print("\nAll runs complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
