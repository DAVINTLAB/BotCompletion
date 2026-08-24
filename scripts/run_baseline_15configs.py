#!/usr/bin/env python3
"""Run the unoptimized baseline prompt on BotSay-340 across 15 (mode, k) configs.

Modes: centrality_top, cluster, latest
k values: 5, 10, 20, 30, 40

Output: results/baseline_15configs.json
This feeds Sec. 5.1 (Post-Selection Validation) of the paper.
"""
import os
import sys
import json
import datetime
import logging
from pathlib import Path

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
from twibot.dspy_components import BotDetectionSignatureWithInstruction


TASK_MODEL = os.environ.get("TASK_MODEL", "openai/gpt-oss-120b")
PROVIDER = os.environ.get("OPENROUTER_PROVIDER", "google-vertex")
DATA_ROOT = project_root / "data"
TESTSET_STD = DATA_ROOT / "twibot22" / "test_clustered.json"
TESTSET_LATEST = DATA_ROOT / "twibot22" / "test_latest.json"
OUT_PATH = project_root / "results" / "baseline_15configs.json"

REFERENCE_DATE = datetime.datetime(2022, 1, 26, tzinfo=datetime.timezone.utc)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

K_VALUES = [5, 10, 20, 30, 40]
MODES = ["centrality_top", "cluster", "latest"]
CONFIGS = [{"mode": m, "k": k} for m in MODES for k in K_VALUES]


class ReasoningAdapter(dspy.ChatAdapter):
    """Injects 'Reasoning: high' suffix into system messages for GPT-OSS."""

    def format_system_message(self, signature) -> str:
        base = super().format_system_message(signature)
        return f"{base}\n\nReasoning: high"


class BotDetector(dspy.Module):
    def __init__(self):
        self.classify = dspy.Predict(BotDetectionSignatureWithInstruction)

    def forward(self, **kwargs):
        return self.classify(**kwargs)


def load_test(mode: str, k: int) -> list[dspy.Example]:
    """Load test set. For latest mode, use test_latest.json + centrality_top
    internally (file is pre-sorted newest-first, so top-k = newest k)."""
    if mode == "latest":
        path = TESTSET_LATEST
        effective_mode = "centrality_top"
    else:
        path = TESTSET_STD
        effective_mode = mode
    with open(path) as f:
        raw = json.load(f)
    return [create_example_for_ablation(u, k, effective_mode, REFERENCE_DATE, DATE_FORMAT) for u in raw]


def evaluate(program: dspy.Module, examples: list[dspy.Example]) -> dict:
    exec_pairs = [(program, ex) for ex in examples]
    parallel = dspy.Parallel(
        num_threads=64,
        max_errors=len(examples),
        return_failed_examples=True,
        provide_traceback=False,
    )
    results, _, exceptions = parallel(exec_pairs)

    gold_labels, pred_labels = [], []
    n_errors = 0
    for idx, (ex, result) in enumerate(zip(examples, results)):
        if result is not None:
            p = normalize_label(result.label)
            if p is not None:
                gold_labels.append(ex.label)
                pred_labels.append(p)
                continue
        n_errors += 1

    acc = accuracy_score(gold_labels, pred_labels) if pred_labels else 0.0
    f1 = f1_score(gold_labels, pred_labels, average="macro", labels=["bot", "human"]) if pred_labels else 0.0
    return {
        "acc": acc,
        "f1_macro": f1,
        "n_scored": len(pred_labels),
        "n_errors": n_errors,
        "n_total": len(examples),
    }


def main():
    load_dotenv(project_root / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        return 1

    task_lm = dspy.LM(
        model=f"openrouter/{TASK_MODEL}",
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
        cache=False,
        temperature=1.0,
        max_tokens=32768,
        extra_body={
            "provider": {
                "order": [PROVIDER],
                "allow_fallbacks": True,
                "require_parameters": True,
                "quantizations": ["bf16"],
            }
        },
    )
    dspy.configure(lm=task_lm, adapter=ReasoningAdapter())

    program = BotDetector()
    base_inst = program.classify.signature.instructions
    print(f"baseline instruction ({len(base_inst)} chars): {base_inst!r}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = []

    for i, cfg in enumerate(CONFIGS, 1):
        mode, k = cfg["mode"], cfg["k"]
        print(f"\n[{i}/{len(CONFIGS)}] {mode}@{k}")
        examples = load_test(mode, k)
        print(f"  loaded {len(examples)} examples")
        metrics = evaluate(program, examples)
        print(f"  Acc={metrics['acc']:.4f}  F1={metrics['f1_macro']:.4f}  "
              f"scored={metrics['n_scored']}/{metrics['n_total']}  errors={metrics['n_errors']}")

        results.append({
            "mode": mode,
            "k": k,
            **metrics,
            "timestamp": datetime.datetime.now().isoformat(),
        })
        # Incremental save so we never lose progress if interrupted
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\nwrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
