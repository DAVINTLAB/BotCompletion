"""Verify the high-RT category selections."""

import json
import sys
import datetime
import statistics
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from twibot.account_classifier import extract_post_features
from twibot.context import create_example_for_ablation

REPO_ROOT = Path(__file__).resolve().parents[2]

REF = datetime.datetime(2022, 1, 26, tzinfo=datetime.timezone.utc)
FMT = "%Y-%m-%d %H:%M:%S"

with open(REPO_ROOT / "data/gepa_trainsets/v4_curation/high_rt.json") as f:
    data = json.load(f)
bots = data["bots"]
humans = data["humans"]

print("=" * 70)
print("HIGH-RT CATEGORY VERIFICATION")
print("=" * 70)
print(f"\nTotal: {len(bots)} bots + {len(humans)} humans = {len(bots)+len(humans)}")

# Check 1: All have RT% > 70
print("\n--- Check 1: All examples have RT% > 70 ---")
violations = []
for u in bots + humans:
    ex = create_example_for_ablation(u, 10, "centrality_top", REF, FMT)
    features = extract_post_features(str(getattr(ex, "tweets", "") or ""))
    if features.get("rt_pct", 0) <= 70:
        violations.append((u.get("id"), features.get("rt_pct", 0), u.get("label")))
if violations:
    print(f"  VIOLATIONS: {len(violations)}")
    for v in violations[:5]:
        print(f"    {v}")
else:
    print(f"  ✓ All {len(bots)+len(humans)} have RT% > 70")

# Check 2: Duplicates
ids = [u.get("id") for u in bots + humans]
dupes = [i for i in set(ids) if ids.count(i) > 1]
print(f"\n--- Check 2: Duplicates: {'NONE' if not dupes else dupes} ---")

# Check 3: Label sanity — suspicious bots and humans
print("\n--- Check 3: Label sanity ---")
# Suspicious bots: high-follower with substantial varied content
susp_bots = []
for u in bots:
    fol = u.get("followers_count", 0) or 0
    desc = (u.get("description", "") or "").strip()
    sc = u.get("statuses_count", 0) or 0
    if fol > 5000 and sc > 5000 and len(desc) > 40:
        susp_bots.append({"id": u.get("id"), "fol": fol, "sc": sc, "desc": desc[:70]})
print(f"  High-profile bots with substantial bios: {len(susp_bots)}")
for s in susp_bots[:3]:
    print(f"    {s}")

# Suspicious humans: low-activity, empty profile
susp_humans = []
for u in humans:
    fol = u.get("followers_count", 0) or 0
    desc = (u.get("description", "") or "").strip()
    sc = u.get("statuses_count", 0) or 0
    if fol < 20 and sc < 100 and len(desc) < 10:
        susp_humans.append({"id": u.get("id"), "fol": fol, "sc": sc, "desc": desc[:50]})
print(f"\n  Humans with very low activity (potentially mislabeled): {len(susp_humans)}")
for s in susp_humans[:3]:
    print(f"    {s}")

# Check 4: RT distribution
print("\n--- Check 4: RT% distribution ---")
rt_buckets = Counter()
for u in bots + humans:
    ex = create_example_for_ablation(u, 10, "centrality_top", REF, FMT)
    features = extract_post_features(str(getattr(ex, "tweets", "") or ""))
    rt_pct = features.get("rt_pct", 0)
    if rt_pct >= 90: bucket = "90-100%"
    elif rt_pct >= 80: bucket = "80-89%"
    else: bucket = "70-79%"
    label_tag = u.get("label")
    rt_buckets[(bucket, label_tag)] += 1

for b in ["90-100%", "80-89%", "70-79%"]:
    bot_c = rt_buckets.get((b, "bot"), 0)
    hum_c = rt_buckets.get((b, "human"), 0)
    print(f"  {b}: bots={bot_c}, humans={hum_c}")

# Check 5: Follower tier
print("\n--- Check 5: Follower tier distribution ---")
bot_tiers = Counter()
human_tiers = Counter()
for u in bots:
    fol = u.get("followers_count", 0) or 0
    bot_tiers["high" if fol > 1000 else "low"] += 1
for u in humans:
    fol = u.get("followers_count", 0) or 0
    human_tiers["high" if fol > 1000 else "low"] += 1
print(f"  Bots:   low-fol={bot_tiers['low']}, high-fol={bot_tiers['high']}")
print(f"  Humans: low-fol={human_tiers['low']}, high-fol={human_tiers['high']}")

# Check 6: Stats
print("\n--- Check 6: Profile stats ---")
bot_fols = [u.get("followers_count", 0) or 0 for u in bots]
human_fols = [u.get("followers_count", 0) or 0 for u in humans]
print(f"  BOTS:   fol median={statistics.median(bot_fols):.0f}")
print(f"  HUMANS: fol median={statistics.median(human_fols):.0f}")

# Check 7: Learning signal
import sqlite3
conn = sqlite3.connect(REPO_ROOT / "results/baseline_trainset/baseline_twibot22.db")
baseline_preds = {uid: pred for uid, _, pred in conn.execute("SELECT user_id, gold_label, pred_label FROM predictions")}
conn.close()
bot_wrong = sum(1 for u in bots if baseline_preds.get(u.get("id")) == "human")
human_wrong = sum(1 for u in humans if baseline_preds.get(u.get("id")) == "bot")
print(f"\n--- Check 7: Learning signal ---")
print(f"  FN bots: {bot_wrong}, FP humans: {human_wrong}")
