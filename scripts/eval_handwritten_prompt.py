#!/usr/bin/env python3
"""No-GEPA ablation: evaluate the hand-written threshold prompt (frozen in
prompts/handwritten_thresholds.json, written once from the paper's stated
rules before any evaluation call) three times on BotSay-340 at the headline
deployment configuration (cluster, k=30). See Sec. 5.2 of the paper.

Harness copied from eval_gepa_table.py; only the program source, the
PROGRAMS list, and OUT_PATH differ.

Output: results/handwritten_prompt_3runs.json (incremental save per run).
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
DATA_ROOT = project_root / "data"
TESTSET_STD = DATA_ROOT / "twibot22" / "test_clustered.json"
PROMPT_PATH = project_root / "prompts" / "handwritten_thresholds.json"
OUT_PATH = project_root / "results" / "handwritten_prompt_3runs.json"
REFERENCE_DATE = datetime.datetime(2022, 1, 26, tzinfo=datetime.timezone.utc)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
N_RUNS = 3

CONFIG = {"exp_id": "handwritten_thresholds", "label": "Hand-written thresholds",
          "mode": "cluster", "eff_mode": "cluster", "source": "clustered", "k": 30}


class ReasoningAdapter(dspy.ChatAdapter):
    def format_system_message(self, signature):
        return f"{super().format_system_message(signature)}\n\nReasoning: high"


class BotDetector(dspy.Module):
    def __init__(self):
        self.classify = dspy.Predict(BotDetectionSignatureWithInstruction)

    def forward(self, **kwargs):
        return self.classify(**kwargs)


def load_program():
    program = BotDetector()
    seed = json.load(open(PROMPT_PATH))
    instr = seed["classify"]["signature"]["instructions"]
    program.classify.signature = program.classify.signature.with_instructions(instr)
    return program, len(instr)


def load_examples(eff_mode, k):
    raw = json.load(open(TESTSET_STD))
    return [create_example_for_ablation(u, k, eff_mode, REFERENCE_DATE, DATE_FORMAT) for u in raw]


def evaluate(program, examples):
    exec_pairs = [(program, ex) for ex in examples]
    parallel = dspy.Parallel(num_threads=64, max_errors=len(examples),
                              return_failed_examples=True, provide_traceback=False)
    results, _, _ = parallel(exec_pairs)
    gold, pred = [], []
    n_err = 0
    for ex, r in zip(examples, results):
        if r is not None:
            p = normalize_label(r.label)
            if p is not None:
                gold.append(ex.label); pred.append(p)
                continue
        n_err += 1
    acc = accuracy_score(gold, pred) if pred else 0.0
    f1 = f1_score(gold, pred, average="macro", labels=["bot", "human"]) if pred else 0.0
    per_class = f1_score(gold, pred, average=None, labels=["bot", "human"]) if pred else [0.0, 0.0]
    return {
        "acc": acc, "f1_macro": f1,
        "f1_bot": float(per_class[0]), "f1_human": float(per_class[1]),
        "n_scored": len(pred), "n_errors": n_err, "n_total": len(examples),
    }


def main():
    load_dotenv(project_root / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    assert api_key, "OPENROUTER_API_KEY missing"

    task_lm = dspy.LM(
        model=f"openrouter/{TASK_MODEL}", api_key=api_key,
        api_base="https://openrouter.ai/api/v1", cache=False,
        temperature=1.0, max_tokens=32768,
        extra_body={"provider": {"sort": "throughput", "allow_fallbacks": True}})
    dspy.configure(lm=task_lm, adapter=ReasoningAdapter())

    program, instr_len = load_program()
    print(f"Loaded hand-written threshold prompt ({instr_len} chars)")

    examples = load_examples(CONFIG["eff_mode"], CONFIG["k"])
    print(f"{len(examples)} examples (cluster, k={CONFIG['k']})")

    if OUT_PATH.exists():
        rows = json.loads(OUT_PATH.read_text())
        done = {r["run"] for r in rows}
        print(f"Resuming: {len(rows)} runs already complete")
    else:
        rows = []
        done = set()

    for run_id in range(1, N_RUNS + 1):
        if run_id in done:
            print(f"  run{run_id}: skipping (already done)")
            continue
        print(f"  run{run_id}: evaluating...")
        m = evaluate(program, examples)
        print(f"    Acc={m['acc']:.4f}  F1={m['f1_macro']:.4f}  "
              f"(bot {m['f1_bot']:.4f}, hum {m['f1_human']:.4f})  errors={m['n_errors']}")
        rows.append({
            "exp_id": CONFIG["exp_id"],
            "label": CONFIG["label"],
            "mode": CONFIG["mode"],
            "k": CONFIG["k"],
            "run": run_id,
            **m,
            "timestamp": datetime.datetime.now().isoformat(),
        })
        OUT_PATH.write_text(json.dumps(rows, indent=2))

    f1s = [r["f1_macro"] for r in rows]
    if f1s:
        import statistics
        mean = statistics.mean(f1s)
        std = statistics.stdev(f1s) if len(f1s) > 1 else 0.0
        print(f"\nFINAL: macro F1 mean={mean:.4f} std={std:.4f} runs={sorted(round(x,4) for x in f1s)}")
    print(f"Wrote {OUT_PATH} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
