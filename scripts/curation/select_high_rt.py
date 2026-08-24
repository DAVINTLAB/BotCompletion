"""Select high-RT examples for trainset v4.

Target: 25 bots, 27 humans = 52.
High-RT = amplification pattern (>70% retweets).
Known FP problem: 16 humans consistently misclassified as amplification bots.
Selection principle: contrastive pairs — amplification bots vs human heavy-retweeters.
"""

import json
import sqlite3
import random
import sys
import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
random.seed(42)

from twibot.account_classifier import extract_post_features, classify_account
from twibot.context import create_example_for_ablation

REPO_ROOT = Path(__file__).resolve().parents[2]

REF = datetime.datetime(2022, 1, 26, tzinfo=datetime.timezone.utc)
FMT = "%Y-%m-%d %H:%M:%S"

with open(REPO_ROOT / "data/twibot22/train_clustered.json") as f:
    train_pool = json.load(f)

conn = sqlite3.connect(REPO_ROOT / "results/baseline_trainset/baseline_twibot22.db")
baseline_preds = {}
for uid, gold, pred in conn.execute("SELECT user_id, gold_label, pred_label FROM predictions"):
    baseline_preds[uid] = pred
conn.close()


def get_fields(u, source):
    uid = u.get("id", u.get("ID", ""))
    v = u.get("verified", False)
    if isinstance(v, str):
        v = v.strip().lower() == "true"
    fol = u.get("followers_count", 0) or 0
    fing = u.get("following_count", 0) or 0
    desc = str(u.get("description", "") or "")

    ex = create_example_for_ablation(u, 10, "centrality_top", REF, FMT)
    features = extract_post_features(str(getattr(ex, "tweets", "") or ""))
    acct_type = classify_account(
        verified=v, followers=fol, following=fing,
        description=desc, features=features,
    )
    return {
        "id": uid, "source": source, "label": u.get("label", "?"),
        "baseline": baseline_preds.get(uid, "n/a"),
        "fol": fol, "fing": fing,
        "sc": u.get("statuses_count", 0) or 0,
        "n_posts": len(u.get("tweets", []) or []),
        "desc": desc.strip(),
        "acct_type": acct_type,
        "url_pct": features.get("url_pct", 0),
        "rt_pct": features.get("rt_pct", 0),
        "verified": v,
        "data": u,
    }


print("Scanning pools for high-RT accounts...")
candidates = []
for u in train_pool:
    c = get_fields(u, "train")
    if "amplification" in c["acct_type"] and not c["verified"]:
        candidates.append(c)

bots = [c for c in candidates if c["label"] == "bot"]
humans = [c for c in candidates if c["label"] == "human"]

print(f"Total high-RT candidates: {len(candidates)}")
print(f"  Bots: {len(bots)}  Humans: {len(humans)}")


def allocate(items, target, prefer_wrong=True):
    if not items:
        return []
    correct = [c for c in items if c["baseline"] == c["label"]]
    wrong = [c for c in items if c["baseline"] != c["label"] and c["baseline"] != "n/a"]
    no_base = [c for c in items if c["baseline"] == "n/a"]

    if prefer_wrong:
        n_wrong = min(target // 2, len(wrong))
        n_correct = min(target - n_wrong, len(correct))
    else:
        n_correct = min(target // 2, len(correct))
        n_wrong = min(target - n_correct, len(wrong))

    n_fill = target - n_correct - n_wrong
    n_nobase = min(n_fill, len(no_base))

    picked = (
        random.sample(correct, n_correct)
        + random.sample(wrong, n_wrong)
        + random.sample(no_base, n_nobase)
    )
    remaining = target - len(picked)
    if remaining > 0:
        leftover = [c for c in items if c not in picked]
        if leftover:
            picked.extend(random.sample(leftover, min(remaining, len(leftover))))
    return picked


# Bucket by RT intensity and follower tier
def rt_bucket(c):
    if c["rt_pct"] >= 90:
        level = "full"  # 90-100%
    elif c["rt_pct"] >= 80:
        level = "heavy"  # 80-89%
    else:
        level = "moderate"  # 70-79%
    tier = "high-fol" if c["fol"] > 1000 else "low-fol"
    return level, tier


# --- BOTS (target 25) ---
bot_buckets = {k: [] for k in [
    ("full", "low-fol"), ("full", "high-fol"),
    ("heavy", "low-fol"), ("heavy", "high-fol"),
    ("moderate", "low-fol"), ("moderate", "high-fol"),
]}
for c in bots:
    bot_buckets[rt_bucket(c)].append(c)

print(f"\nBot buckets:")
for k, v in bot_buckets.items():
    print(f"  {k}: {len(v)}")

bot_targets = {
    ("full", "low-fol"): 5,
    ("full", "high-fol"): 3,
    ("heavy", "low-fol"): 5,
    ("heavy", "high-fol"): 4,
    ("moderate", "low-fol"): 4,
    ("moderate", "high-fol"): 4,
}

selected_bots = []
for key, target in bot_targets.items():
    picked = allocate(bot_buckets[key], target, prefer_wrong=True)
    selected_bots.extend(picked)
    print(f"  bots {key}: target={target} got={len(picked)}")

# --- HUMANS (target 27) ---
# Humans here are all heavy retweeters — need to teach that high RT% alone doesn't mean bot
human_buckets = {k: [] for k in [
    ("full", "low-fol"), ("full", "high-fol"),
    ("heavy", "low-fol"), ("heavy", "high-fol"),
    ("moderate", "low-fol"), ("moderate", "high-fol"),
]}
for c in humans:
    human_buckets[rt_bucket(c)].append(c)

print(f"\nHuman buckets:")
for k, v in human_buckets.items():
    print(f"  {k}: {len(v)}")

human_targets = {
    ("full", "low-fol"): 5,
    ("full", "high-fol"): 4,
    ("heavy", "low-fol"): 5,
    ("heavy", "high-fol"): 4,
    ("moderate", "low-fol"): 5,
    ("moderate", "high-fol"): 4,
}

selected_humans = []
for key, target in human_targets.items():
    picked = allocate(human_buckets[key], target, prefer_wrong=True)
    selected_humans.extend(picked)
    print(f"  humans {key}: target={target} got={len(picked)}")

print(f"\n{'='*70}")
print("HIGH-RT SUMMARY")
print("="*70)
print(f"Selected: {len(selected_bots)} bots + {len(selected_humans)} humans")
bot_wrong = sum(1 for c in selected_bots if c["baseline"] == "human")
human_wrong = sum(1 for c in selected_humans if c["baseline"] == "bot")
print(f"Baseline-wrong: {bot_wrong} FN bots, {human_wrong} FP humans")
out = {
    "category": "high-rt",
    "bots": [c["data"] for c in selected_bots],
    "humans": [c["data"] for c in selected_humans],
    "meta": {
        "bot_count": len(selected_bots),
        "human_count": len(selected_humans),
        "bot_ids": [c["id"] for c in selected_bots],
        "human_ids": [c["id"] for c in selected_humans],
    },
}

out_path = Path(REPO_ROOT / "data/gepa_trainsets/v4_curation/high_rt.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\nSaved to {out_path}")
