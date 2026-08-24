"""Select standard account examples for trainset v4.

Target: 60 bots, 65 humans = 125.
Standard = no strong archetype signal (not sparse, not verified, not high-RT/URL/template).
This is the largest and most heterogeneous group. Our persistent errors concentrate here:
  - 23 standard account FN bots
  - 5 standard account FP humans

Selection principle: cover a wide variety of sub-patterns
  - Mix baseline-correct and baseline-wrong
  - Ensure variety in follower counts, bio types, content patterns
  - Emphasize hard cases (baseline errors) for learning signal
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


def classify_standard(u):
    """Return True if this account is 'standard' (not in any specialized group)."""
    v = u.get("verified", False)
    if isinstance(v, str):
        v = v.strip().lower() == "true"
    if v:
        return False
    fol = u.get("followers_count", 0) or 0
    sc = u.get("statuses_count", 0) or 0
    fing = u.get("following_count", 0) or 0
    # Not sparse
    if fol < 20 and sc < 200:
        return False
    # Not follow-spam
    if fol < 10 and fing >= 100:
        return False

    # Check account type via metric classifier
    ex = create_example_for_ablation(u, 10, "centrality_top", REF, FMT)
    features = extract_post_features(str(getattr(ex, "tweets", "") or ""))
    desc = str(getattr(ex, "description", "") or "")
    acct_type = classify_account(
        verified=False,
        followers=fol,
        following=fing,
        description=desc,
        features=features,
    )
    return acct_type == "standard account"


def get_fields(u, source):
    uid = u.get("id", u.get("ID", ""))
    return {
        "id": uid, "source": source, "label": u.get("label", "?"),
        "baseline": baseline_preds.get(uid, "n/a"),
        "fol": u.get("followers_count", 0) or 0,
        "fing": u.get("following_count", 0) or 0,
        "sc": u.get("statuses_count", 0) or 0,
        "n_posts": len(u.get("tweets", []) or []),
        "desc": (u.get("description", "") or "").strip(),
        "data": u,
    }


print("Scanning train pool...")
train_standard = [get_fields(u, "train") for u in train_pool if classify_standard(u)]
print(f"  Found {len(train_standard)} standard accounts in train pool")

candidates = train_standard
bots = [c for c in candidates if c["label"] == "bot"]
humans = [c for c in candidates if c["label"] == "human"]

print(f"\nTotal candidates: {len(candidates)}")
print(f"  Bots: {len(bots)}")
print(f"  Humans: {len(humans)}")

# --- SUBTYPES for standard accounts ---
# Based on our error analysis:
# FN bots tend to be: high followers (>1k), established-looking, professional bios
# FP humans tend to be: small (<300 fol), low activity, thin bios
# Need coverage of:
#   - Small standard (low fol)
#   - Mid standard (fol 200-1k)
#   - High standard (fol 1k+) - especially important for FN coverage
#   - With bio vs without bio
#   - Mix of baseline-correct and baseline-wrong

def bucket(c):
    if c["fol"] < 200:
        tier = "small"
    elif c["fol"] < 2000:
        tier = "mid"
    else:
        tier = "high"
    has_bio = len(c["desc"]) > 20
    return tier, has_bio


def allocate(items, target, prefer_wrong=False):
    """Select items with preference for baseline-wrong (learning signal)."""
    if not items:
        return []
    correct = [c for c in items if c["baseline"] == c["label"]]
    wrong = [c for c in items if c["baseline"] != c["label"] and c["baseline"] != "n/a"]
    no_base = [c for c in items if c["baseline"] == "n/a"]  # no baseline prediction

    # Prefer wrong for learning signal
    if prefer_wrong:
        n_wrong = min(target // 2, len(wrong))
        n_correct = min(target - n_wrong, len(correct))
    else:
        n_correct = min(target // 2, len(correct))
        n_wrong = min(target - n_correct, len(wrong))
    n_filled = n_correct + n_wrong
    n_fill = target - n_filled
    n_nobase = min(n_fill, len(no_base))

    picked = (
        random.sample(correct, n_correct)
        + random.sample(wrong, n_wrong)
        + random.sample(no_base, n_nobase)
    )
    # Top up if still short
    remaining_target = target - len(picked)
    if remaining_target > 0:
        leftover = [c for c in items if c not in picked]
        if leftover:
            picked.extend(random.sample(leftover, min(remaining_target, len(leftover))))
    return picked


# --- BOTS (target 60) ---
# Distribution: small=12, mid=25, high=23 (emphasize high, where FN bots live)
# With/without bio: roughly 50/50

bot_buckets = {
    ("small", True): [],
    ("small", False): [],
    ("mid", True): [],
    ("mid", False): [],
    ("high", True): [],
    ("high", False): [],
}
for c in bots:
    bot_buckets[bucket(c)].append(c)

print(f"\nBot bucket counts:")
for k, v in bot_buckets.items():
    print(f"  {k}: {len(v)}")

# Allocations for bots (emphasize high tier for FN coverage, prefer baseline-wrong)
bot_targets = {
    ("small", True): 3,
    ("small", False): 3,
    ("mid", True): 12,
    ("mid", False): 8,
    ("high", True): 16,
    ("high", False): 8,
}

selected_bots = []
for key, target in bot_targets.items():
    pool = bot_buckets[key]
    picked = allocate(pool, target, prefer_wrong=True)
    selected_bots.extend(picked)
    print(f"  bots {key}: target={target} got={len(picked)}")

print(f"\nTotal bots selected: {len(selected_bots)}")

# --- HUMANS (target 65) ---
# Emphasize small-tier humans (FP source)
# Mix of professionals (with bios) and casual users

human_buckets = {
    ("small", True): [],
    ("small", False): [],
    ("mid", True): [],
    ("mid", False): [],
    ("high", True): [],
    ("high", False): [],
}
for c in humans:
    human_buckets[bucket(c)].append(c)

print(f"\nHuman bucket counts:")
for k, v in human_buckets.items():
    print(f"  {k}: {len(v)}")

# Humans: spread across tiers, but emphasize small (FP source)
human_targets = {
    ("small", True): 12,
    ("small", False): 10,
    ("mid", True): 13,
    ("mid", False): 8,
    ("high", True): 14,
    ("high", False): 8,
}

selected_humans = []
for key, target in human_targets.items():
    pool = human_buckets[key]
    picked = allocate(pool, target, prefer_wrong=True)
    selected_humans.extend(picked)
    print(f"  humans {key}: target={target} got={len(picked)}")

print(f"\nTotal humans selected: {len(selected_humans)}")

# Summary
print("\n" + "=" * 70)
print("STANDARD CATEGORY SUMMARY")
print("=" * 70)
print(f"Selected: {len(selected_bots)} bots + {len(selected_humans)} humans = {len(selected_bots) + len(selected_humans)} examples")

bot_wrong = sum(1 for c in selected_bots if c["baseline"] == "human")
human_wrong = sum(1 for c in selected_humans if c["baseline"] == "bot")
print(f"Baseline-wrong: {bot_wrong} FN bots (learning signal), {human_wrong} FP humans (learning signal)")

# Save
out = {
    "category": "standard",
    "bots": [c["data"] for c in selected_bots],
    "humans": [c["data"] for c in selected_humans],
    "meta": {
        "bot_count": len(selected_bots),
        "human_count": len(selected_humans),
        "bot_ids": [c["id"] for c in selected_bots],
        "human_ids": [c["id"] for c in selected_humans],
    },
}

out_path = Path(REPO_ROOT / "data/gepa_trainsets/v4_curation/standard.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\nSaved to {out_path}")
