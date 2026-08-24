"""Select verified examples for trainset v4.

Target: 0 bots, 45 humans.
All verified accounts in TwiBot-22 are humans (the dataset has 0 verified bots).
Selection principle: teach GEPA the "verified = human" hard rule with diverse examples
  (celebrities, news outlets, corporate brands, government accounts).
"""

import json
import sqlite3
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

random.seed(42)

with open(REPO_ROOT / "data/twibot22/train_clustered.json") as f:
    train_pool = json.load(f)

conn = sqlite3.connect(REPO_ROOT / "results/baseline_trainset/baseline_twibot22.db")
baseline_preds = {}
for uid, gold, pred in conn.execute("SELECT user_id, gold_label, pred_label FROM predictions"):
    baseline_preds[uid] = pred
conn.close()


def is_verified(u):
    v = u.get("verified", False)
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return bool(v)


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


candidates = []
for u in train_pool:
    if is_verified(u):
        candidates.append(get_fields(u, "train"))

print(f"Total verified candidates: {len(candidates)}")
print(f"  Bots: {sum(1 for c in candidates if c['label'] == 'bot')}")
print(f"  Humans: {sum(1 for c in candidates if c['label'] == 'human')}")

verified_humans = [c for c in candidates if c["label"] == "human"]

# Categorize verified accounts by archetype
# - high-follower celebrities/brands (fol > 100k)
# - mid-range verified (fol 10k-100k)
# - smaller verified (fol < 10k)
# - verified with templated content (potential source of misclassification)
celebs = []  # fol > 100k
mid = []  # 10k-100k
small = []  # < 10k
for c in verified_humans:
    if c["fol"] > 100_000:
        celebs.append(c)
    elif c["fol"] > 10_000:
        mid.append(c)
    else:
        small.append(c)

print(f"\nArchetype breakdown (humans only):")
print(f"  Celebrities/brands (>100k fol): {len(celebs)}")
print(f"  Mid-range (10k-100k): {len(mid)}")
print(f"  Small verified (<10k): {len(small)}")

# Prioritize baseline-wrong examples (edge cases where verified rule needed)
# Baseline should NEVER be wrong on verified humans (hard rule), but check
fp_verified = [c for c in verified_humans if c["baseline"] == "bot"]
print(f"  Baseline wrong (FP, model flagged verified human as bot): {len(fp_verified)}")

# Target: 45 verified humans
# Allocation:
# - 15 celebrities
# - 20 mid-range (most diverse, includes news/media)
# - 10 small verified (edge cases)
# - Prefer mix of baseline-correct and baseline-wrong

selected = []

# Take all FP verified first (learning signal - baseline broke the rule)
target_fp = min(10, len(fp_verified))
selected.extend(random.sample(fp_verified, target_fp))
print(f"\nTaking {target_fp} FP verified examples (where baseline broke hard rule)")

# Then balance across follower tiers
already_ids = {c["id"] for c in selected}
remaining_celebs = [c for c in celebs if c["id"] not in already_ids]
remaining_mid = [c for c in mid if c["id"] not in already_ids]
remaining_small = [c for c in small if c["id"] not in already_ids]

need_celebs = min(15 - sum(1 for c in selected if c["fol"] > 100_000), len(remaining_celebs))
need_mid = min(20 - sum(1 for c in selected if 10_000 < c["fol"] <= 100_000), len(remaining_mid))
need_small = min(10 - sum(1 for c in selected if c["fol"] <= 10_000), len(remaining_small))

selected.extend(random.sample(remaining_celebs, need_celebs))
selected.extend(random.sample(remaining_mid, need_mid))
selected.extend(random.sample(remaining_small, need_small))

# Cap at 45
selected = selected[:45]

print(f"\nSelected verified humans: {len(selected)}")
# Breakdown
print(f"  By tier: celebs={sum(1 for c in selected if c['fol'] > 100_000)}, "
      f"mid={sum(1 for c in selected if 10_000 < c['fol'] <= 100_000)}, "
      f"small={sum(1 for c in selected if c['fol'] <= 10_000)}")
print(f"  Baseline-wrong (FP): {sum(1 for c in selected if c['baseline'] == 'bot')}")

print("\n" + "=" * 70)
print("SELECTED VERIFIED HUMANS (sample of 20)")
print("=" * 70)
for c in selected[:20]:
    print(f"  [{c['source']}] id={c['id']:<22} baseline={c['baseline']:<6} fol={c['fol']:>8} desc=\"{c['desc'][:55]}\"")

# Save
out = {
    "category": "verified",
    "bots": [],
    "humans": [c["data"] for c in selected],
    "meta": {
        "bot_count": 0,
        "human_count": len(selected),
        "human_ids": [c["id"] for c in selected],
    },
}

out_path = Path(REPO_ROOT / "data/gepa_trainsets/v4_curation/verified.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\nSaved to {out_path}")
