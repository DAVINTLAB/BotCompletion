"""Assemble trainset v4 from all category files."""

import json
import random
from pathlib import Path
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parents[2]

random.seed(42)

CATEGORIES = [
    "sparse", "verified", "standard", "high_url", "high_rt",
    "template", "moderate_rt", "follow_spam", "institutional", "support",
]

CURATION_DIR = Path(REPO_ROOT / "data/gepa_trainsets/v4_curation")

all_examples = []
category_counts = {}
for cat in CATEGORIES:
    path = CURATION_DIR / f"{cat}.json"
    with open(path) as f:
        data = json.load(f)
    bots = data["bots"]
    humans = data["humans"]
    for b in bots:
        b["_group"] = cat
    for h in humans:
        h["_group"] = cat
    all_examples.extend(bots)
    all_examples.extend(humans)
    category_counts[cat] = (len(bots), len(humans))

# Deduplicate by ID
seen_ids = set()
deduped = []
for u in all_examples:
    uid = u.get("id", u.get("ID"))
    if uid in seen_ids:
        continue
    seen_ids.add(uid)
    deduped.append(u)

# Shuffle
random.shuffle(deduped)

# Safety check: the trainset must be disjoint from the evaluation split
with open(REPO_ROOT / "data/twibot22/test_clustered.json") as f:
    test = json.load(f)
test_ids = {u.get("id", u.get("ID")) for u in test}
overlap = sorted(uid for uid in seen_ids if uid in test_ids)
assert not overlap, (
    f"{len(overlap)} trainset users are in the evaluation split: {overlap[:5]}"
)

# Save final trainset
out_path = Path(REPO_ROOT / "data/gepa_trainsets/gepa_trainset_twibot22_v4.json")
with open(out_path, "w") as f:
    json.dump(deduped, f, indent=2, default=str)

print("=" * 70)
print("TRAINSET V4 ASSEMBLY")
print("=" * 70)
print(f"\nTotal examples (after dedup): {len(deduped)}")
print(f"  (Before dedup: {len(all_examples)})")

labels = Counter(u.get("label") for u in deduped)
print(f"\nLabel balance:")
for l, c in labels.most_common():
    print(f"  {l}: {c}")

print(f"\nPer-category breakdown:")
print(f"{'Category':<20} {'Bots':>6} {'Humans':>7} {'Total':>6}")
print("-" * 42)
for cat in CATEGORIES:
    b, h = category_counts[cat]
    print(f"{cat:<20} {b:>6} {h:>7} {b+h:>6}")

# Check distribution vs test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import datetime
from twibot.account_classifier import extract_post_features, classify_account
from twibot.context import create_example_for_ablation

REF = datetime.datetime(2022, 1, 26, tzinfo=datetime.timezone.utc)
FMT = "%Y-%m-%d %H:%M:%S"

def get_group(u):
    v = u.get("verified", False)
    if isinstance(v, str):
        v = v.strip().lower() == "true"
    fol = u.get("followers_count", 0) or 0
    fing = u.get("following_count", 0) or 0
    desc = str(u.get("description", "") or "")
    sc = u.get("statuses_count", 0) or 0

    if v: return "verified"
    if fol < 20 and sc < 200: return "sparse"
    if fol < 10 and fing >= 100: return "follow_spam"

    ex = create_example_for_ablation(u, 10, "centrality_top", REF, FMT)
    features = extract_post_features(str(getattr(ex, "tweets", "") or ""))
    acct = classify_account(verified=v, followers=fol, following=fing, description=desc, features=features)
    if "amplification" in acct: return "high_rt"
    if "feed/aggregation" in acct: return "high_url"
    if "heavy curator" in acct: return "moderate_rt"
    if "template" in acct: return "template"
    if "institutional" in acct: return "institutional"
    if "support" in acct: return "support"
    if "follow-spam" in acct: return "follow_spam"
    return "standard"


test_counts = Counter()
for u in test:
    test_counts[get_group(u)] += 1

print(f"\n\nDistribution comparison vs test set:")
print(f"{'Group':<20} {'Train %':>10} {'Test %':>10} {'Δ':>8}")
print("-" * 50)
total_train = len(deduped)
total_test = len(test)
for cat in CATEGORIES:
    b, h = category_counts[cat]
    train_pct = (b + h) / total_train * 100
    test_pct = test_counts.get(cat, 0) / total_test * 100
    delta = train_pct - test_pct
    print(f"{cat:<20} {train_pct:>9.1f}% {test_pct:>9.1f}% {delta:>+7.1f}%")

print(f"\nSaved to {out_path}")
