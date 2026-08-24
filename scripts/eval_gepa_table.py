#!/usr/bin/env python3
"""Run each of the 6 GEPA-evolved prompts three times at their deployment
(mode, k) configuration, reproducing the GEPA-results table of the paper
(Sec. 5.2): mean ± sample standard deviation over three inference passes.

Provider config tuned for max throughput:
  - sort: throughput (OpenRouter routes to highest-TPS available provider)
  - allow_fallbacks: True (no quantization or precision pinning)
  - temperature: 1.0

Output: results/gepa_table_3runs.json (incremental save after every cell).
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
PROMPTS_DIR = project_root / "prompts" / "evolved"
TESTSET_STD = DATA_ROOT / "twibot22" / "test_clustered.json"
TESTSET_LATEST = DATA_ROOT / "twibot22" / "test_latest.json"
OUT_PATH = project_root / "results" / "gepa_table_3runs.json"
REFERENCE_DATE = datetime.datetime(2022, 1, 26, tzinfo=datetime.timezone.utc)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
N_RUNS = 3

# Six rows of the GEPA table — each evaluated at its deployment (mode, k).
PROGRAMS = [
    {"prompt": "latest_opus.json",       "label": "Latest + Opus",
     "mode": "latest",         "eff_mode": "centrality_top", "source": "latest",    "k": 30},
    {"prompt": "centrality_opus.json",   "label": "Centrality + Opus",
     "mode": "centrality_top", "eff_mode": "centrality_top", "source": "clustered", "k": 10},
    {"prompt": "cluster_opus.json",      "label": "Cluster + Opus",
     "mode": "cluster",        "eff_mode": "cluster",        "source": "clustered", "k": 30},
    {"prompt": "latest_gemini.json",     "label": "Latest + Gemini",
     "mode": "latest",         "eff_mode": "centrality_top", "source": "latest",    "k": 30},
    {"prompt": "centrality_gemini.json", "label": "Centrality + Gemini",
     "mode": "centrality_top", "eff_mode": "centrality_top", "source": "clustered", "k": 10},
    {"prompt": "cluster_gemini.json",    "label": "Cluster + Gemini",
     "mode": "cluster",        "eff_mode": "cluster",        "source": "clustered", "k": 30},
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
    assert api_key, "OPENROUTER_API_KEY missing"

    task_lm = dspy.LM(
        model=f"openrouter/{TASK_MODEL}", api_key=api_key,
        api_base="https://openrouter.ai/api/v1", cache=False,
        temperature=1.0, max_tokens=32768,
        extra_body={"provider": {"sort": "throughput", "allow_fallbacks": True}})
    dspy.configure(lm=task_lm, adapter=ReasoningAdapter())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Resume if partial output exists
    if OUT_PATH.exists():
        rows = json.loads(OUT_PATH.read_text())
        done = {(r["prompt"], r["run"]) for r in rows}
        print(f"Resuming: {len(rows)} cells already complete")
    else:
        rows = []
        done = set()

    total_cells = len(PROGRAMS) * N_RUNS
    cell_n = 0
    for cfg in PROGRAMS:
        program, instr_len = load_program(cfg["prompt"])
        print(f"\nLoaded {cfg['label']} ({cfg['prompt']}, {instr_len} chars)")

        examples = load_examples(cfg["source"], cfg["eff_mode"], cfg["k"])

        for run_id in range(1, N_RUNS + 1):
            cell_n += 1
            if (cfg["prompt"], run_id) in done:
                print(f"  [{cell_n}/{total_cells}] skipping {cfg['label']} run{run_id} (already done)")
                continue
            print(f"  [{cell_n}/{total_cells}] {cfg['label']} run{run_id}: k={cfg['k']}, {len(examples)} examples")
            m = evaluate(program, examples)
            print(f"    Acc={m['acc']:.4f}  F1={m['f1_macro']:.4f}  "
                  f"(bot {m['f1_bot']:.4f}, hum {m['f1_human']:.4f})  errors={m['n_errors']}")
            rows.append({
                "prompt": cfg["prompt"],
                "label": cfg["label"],
                "mode": cfg["mode"],
                "k": cfg["k"],
                "run": run_id,
                **m,
                "timestamp": datetime.datetime.now().isoformat(),
            })
            OUT_PATH.write_text(json.dumps(rows, indent=2))

    print(f"\nWrote {OUT_PATH} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
