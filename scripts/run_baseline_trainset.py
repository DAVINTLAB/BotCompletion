#!/usr/bin/env python3
"""
Run the unoptimized baseline classifier on the full TwiBot-22 training pool to
identify misclassifications for GEPA training-set curation (Sec. 3.5).

Uses centrality@10 with the unoptimized BotDetectionSignatureGPTOSS instruction.
Saves predictions to SQLite (results/baseline_trainset/baseline_twibot22.db),
which the scripts in scripts/curation/ read for stratified sampling.
"""

import os
import sys
import json
import hashlib
import sqlite3
import datetime
import logging
from pathlib import Path

# Mute DSPy/litellm output
os.environ["DSP_CACHEBOOL"] = "false"
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
from twibot.dspy_components import BotDetectionSignatureGPTOSS


# ─── Configuration ──────────────────────────────────────────────────────

MODEL_NAME = os.environ.get("TASK_MODEL", "openai/gpt-oss-120b")
PROVIDER = os.environ.get("OPENROUTER_PROVIDER", "google-vertex")
SELECTION_MODE = "centrality_top"
MAX_TWEETS = 10
NUM_THREADS = 4

DATA_ROOT = project_root / "data"
OUTPUT_DIR = project_root / "results" / "baseline_trainset"

DATASET_CONFIG = {
    "twibot22": {
        "path": str(DATA_ROOT / "twibot22" / "train_clustered.json"),
        "reference_date": datetime.datetime(2022, 1, 26, tzinfo=datetime.timezone.utc),
        "date_format": "%Y-%m-%d %H:%M:%S",
    },
}


def load_dataset(name: str) -> tuple:
    cfg = DATASET_CONFIG[name]
    with open(cfg["path"], "r") as f:
        data = json.load(f)
    return data, cfg["reference_date"], cfg["date_format"]


def make_experiment_id(dataset: str) -> str:
    key = f"{dataset}_{MODEL_NAME}_{SELECTION_MODE}_{MAX_TWEETS}_baseline_train"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def run_baseline(dataset_name: str, api_key: str):
    print(f"\n{'#'*60}")
    print(f"# Baseline on {dataset_name} training set")
    print(f"# Selection: {SELECTION_MODE}@{MAX_TWEETS}")
    print(f"{'#'*60}")

    train_data, ref_date, date_fmt = load_dataset(dataset_name)
    print(f"  Training set size: {len(train_data)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    db_path = OUTPUT_DIR / f"baseline_{dataset_name}.db"
    db = AblationResultsDB(str(db_path))

    exp_id = make_experiment_id(dataset_name)

    # Check how many predictions already exist (for resumption)
    with sqlite3.connect(db.db_path) as conn:
        existing = conn.execute(
            "SELECT COUNT(DISTINCT user_id) FROM predictions WHERE experiment_id = ?",
            (exp_id,)
        ).fetchone()[0]

    if existing >= len(train_data):
        print(f"  SKIP — already have {existing} predictions")
        return

    if existing > 0:
        with sqlite3.connect(db.db_path) as conn:
            done_ids = set(
                row[0] for row in conn.execute(
                    "SELECT DISTINCT user_id FROM predictions WHERE experiment_id = ?",
                    (exp_id,)
                ).fetchall()
            )
        print(f"  Resuming — {existing} done, {len(train_data) - existing} remaining")
    else:
        done_ids = set()
        db.start_experiment(
            exp_id, dataset_name, MODEL_NAME, "reasoning", SELECTION_MODE, MAX_TWEETS
        )

    # Configure DSPy
    lm = dspy.LM(
        model=f"openrouter/{MODEL_NAME}",
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
        cache=False,
        temperature=0.0,
        max_tokens=32768,
        extra_body={
            "provider": {
                "order": [PROVIDER],
                "allow_fallbacks": False,
                "require_parameters": True,
            }
        },
    )
    dspy.configure(lm=lm, adapter=dspy.ChatAdapter())
    predictor = dspy.Predict(BotDetectionSignatureGPTOSS)

    # Build examples for remaining users
    examples = []
    user_ids = []
    for i, user_data in enumerate(train_data):
        uid = user_data.get("ID", user_data.get("id", str(i)))
        if uid in done_ids:
            continue
        user_ids.append(uid)
        ex = create_example_for_ablation(
            user_data, MAX_TWEETS, SELECTION_MODE, ref_date, date_fmt
        )
        examples.append(ex)

    print(f"  Running {len(examples)} predictions...")

    # Process in batches for incremental progress saving
    BATCH_SIZE = 200
    total_correct = 0
    total_errors = 0
    all_gold = []
    all_pred = []

    for batch_start in range(0, len(examples), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(examples))
        batch_examples = examples[batch_start:batch_end]
        batch_uids = user_ids[batch_start:batch_end]

        print(f"\n  Batch {batch_start}-{batch_end} / {len(examples)}")

        exec_pairs = [(predictor, ex) for ex in batch_examples]
        parallel = dspy.Parallel(
            num_threads=NUM_THREADS,
            max_errors=len(batch_examples),
            return_failed_examples=True,
            provide_traceback=False,
        )
        results, failed, exceptions = parallel(exec_pairs)

        for idx, (uid, ex, result) in enumerate(zip(batch_uids, batch_examples, results)):
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

            db.add_prediction(exp_id, uid, ex.username, gold, pred, err)

            if pred:
                all_gold.append(gold)
                all_pred.append(pred)
                if pred == gold:
                    total_correct += 1
            if err:
                total_errors += 1

        print(f"  Running total: {total_correct}/{len(all_gold)} correct, {total_errors} errors")

    # Final metrics
    if all_pred:
        acc = accuracy_score(all_gold, all_pred)
        f1 = f1_score(all_gold, all_pred, average="macro", labels=["bot", "human"])
        print(f"\n  FINAL: Acc={acc:.4f}  F1={f1:.4f}  Errors={total_errors}/{len(examples)}")
        db.complete_experiment(exp_id, acc, f1, len(train_data), total_correct, total_errors)
    else:
        print(f"\n  No new predictions collected")


def main():
    load_dotenv(project_root / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        return 1

    run_baseline("twibot22", api_key)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
