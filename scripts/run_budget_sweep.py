#!/usr/bin/env python3
"""Sweep each of the 6 GEPA-evolved prompts across k ∈ {5,10,20,30,40}
on BotSay-340, at the program's native post-selection mode (Sec. 5.3 of
the paper, post-budget sensitivity).

For the latest mode, we use the test_latest.json file (newest-first
ordering) and select with 'centrality_top' (first-k = newest k).
For cluster and centrality_top modes, we use test_clustered.json.

Output: results/budget_sweep_6programs.json
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
PROVIDER = os.environ.get("OPENROUTER_PROVIDER", "google-vertex/global")
DATA_ROOT = project_root / "data"
PROMPTS_DIR = project_root / "prompts" / "evolved"
TESTSET_STD = DATA_ROOT / "twibot22" / "test_clustered.json"
TESTSET_LATEST = DATA_ROOT / "twibot22" / "test_latest.json"
OUT_PATH = project_root / "results" / "budget_sweep_6programs.json"
REFERENCE_DATE = datetime.datetime(2022, 1, 26, tzinfo=datetime.timezone.utc)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

K_VALUES = [5, 10, 20, 30, 40]

# train_k is the (mode, k) configuration each prompt was optimized at.
PROGRAMS = [
    {"prompt": "latest_opus.json",       "label": "Latest@30 + Opus",
     "mode": "latest",         "eff_mode": "centrality_top", "source": "latest",    "train_k": 30},
    {"prompt": "centrality_opus.json",   "label": "Centrality@10 + Opus",
     "mode": "centrality_top", "eff_mode": "centrality_top", "source": "clustered", "train_k": 10},
    {"prompt": "cluster_opus.json",      "label": "Cluster@30 + Opus",
     "mode": "cluster",        "eff_mode": "cluster",        "source": "clustered", "train_k": 30},
    {"prompt": "latest_gemini.json",     "label": "Latest@30 + Gemini",
     "mode": "latest",         "eff_mode": "centrality_top", "source": "latest",    "train_k": 30},
    {"prompt": "centrality_gemini.json", "label": "Centrality@10 + Gemini",
     "mode": "centrality_top", "eff_mode": "centrality_top", "source": "clustered", "train_k": 10},
    {"prompt": "cluster_gemini.json",    "label": "Cluster@30 + Gemini (headline)",
     "mode": "cluster",        "eff_mode": "cluster",        "source": "clustered", "train_k": 20},
]


class ReasoningAdapter(dspy.ChatAdapter):
    def format_system_message(self, signature):
        return f"{super().format_system_message(signature)}\n\nReasoning: high"


class BotDetector(dspy.Module):
    def __init__(self):
        self.classify = dspy.Predict(BotDetectionSignatureWithInstruction)

    def forward(self, **kwargs):
        return self.classify(**kwargs)


def load_program(prompt_file):
    path = PROMPTS_DIR / prompt_file
    program = BotDetector()
    seed = json.load(open(path))
    instr = seed["classify"]["signature"]["instructions"]
    program.classify.signature = program.classify.signature.with_instructions(instr)
    return program, len(instr)


def load_examples(source, eff_mode, k):
    path = TESTSET_LATEST if source == "latest" else TESTSET_STD
    raw = json.load(open(path))
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
    assert api_key

    task_lm = dspy.LM(
        model=f"openrouter/{TASK_MODEL}", api_key=api_key,
        api_base="https://openrouter.ai/api/v1", cache=False,
        temperature=1.0, max_tokens=32768,
        # Google Vertex reports quant="unknown", so we omit the quantizations filter
        # when pinning to it. require_parameters drops too: Google's supported_parameters
        # list omits some defaults DSPy may include.
        extra_body={"provider": {"order": [PROVIDER], "allow_fallbacks": False}})
    dspy.configure(lm=task_lm, adapter=ReasoningAdapter())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for i, cfg in enumerate(PROGRAMS, 1):
        print(f"\n========== [{i}/{len(PROGRAMS)}] {cfg['label']} ({cfg['prompt']}) ==========")
        program, instr_len = load_program(cfg["prompt"])
        print(f"  loaded instruction ({instr_len} chars)")

        for j, k in enumerate(K_VALUES, 1):
            examples = load_examples(cfg["source"], cfg["eff_mode"], k)
            print(f"  [{j}/{len(K_VALUES)}] k={k}: {len(examples)} examples")
            m = evaluate(program, examples)
            print(f"    Acc={m['acc']:.4f}  F1={m['f1_macro']:.4f}  "
                  f"(bot {m['f1_bot']:.4f}, hum {m['f1_human']:.4f})  errors={m['n_errors']}")
            rows.append({
                "prompt": cfg["prompt"],
                "label": cfg["label"],
                "mode": cfg["mode"],
                "train_k": cfg["train_k"],
                "k": k,
                **m,
                "timestamp": datetime.datetime.now().isoformat(),
            })
            OUT_PATH.write_text(json.dumps(rows, indent=2))

    print(f"\nwrote {OUT_PATH} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
