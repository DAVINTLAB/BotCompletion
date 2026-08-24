"""Verify the high-URL category selections."""

import json
import sys
import datetime
import statistics
from pathlib import Path
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from twibot.account_classifier import extract_post_features
from twibot.context import create_example_for_ablation

REF = datetime.datetime(2022, 1, 26, tzinfo=datetime.timezone.utc)
FMT = "%Y-%m-%d %H:%M:%S"

with open(REPO_ROOT / "data/gepa_trainsets/v4_curation/high_url.json") as f:
    data = json.load(f)
bots = data["bots"]
humans = data["humans"]

print("=" * 70)
print("HIGH-URL CATEGORY VERIFICATION")
print("=" * 70)
print(f"\nTotal: {len(bots)} bots + {len(humans)} humans = {len(bots)+len(humans)}")

# Check 1: All genuinely high-URL (url_pct > 60)
print("\n--- Check 1: All examples have URL% > 60 ---")
violations = []
for u in bots + humans:
    ex = create_example_for_ablation(u, 10, "centrality_top", REF, FMT)
    features = extract_post_features(str(getattr(ex, "tweets", "") or ""))
    if features.get("url_pct", 0) <= 60:
        violations.append((u.get("id"), features.get("url_pct", 0), u.get("label")))
if violations:
    print(f"  VIOLATIONS: {len(violations)}")
    for v in violations[:5]:
        print(f"    {v}")
else:
    print(f"  ✓ All {len(bots)+len(humans)} have URL% > 60")

# Check 2: Duplicates
ids = [u.get("id") for u in bots + humans]
dupes = [i for i in set(ids) if ids.count(i) > 1]
print(f"\n--- Check 2: Duplicates: {'NONE' if not dupes else dupes} ---")

# Check 3: Label sanity
# Look at bots that might be legit brands (high fol, news-ish bio)
print("\n--- Check 3: Potentially mislabeled bots (high-profile institutional) ---")
susp_bots = []
for u in bots:
    fol = u.get("followers_count", 0) or 0
    desc = (u.get("description", "") or "").strip().lower()
    # "Official" or big institutional bots
    if fol > 10000 and ("official" in desc or "news" in desc[:30]):
        susp_bots.append({"id": u.get("id"), "fol": fol, "desc": desc[:80]})
print(f"  High-fol 'official' bots: {len(susp_bots)}")
for s in susp_bots[:5]:
    print(f"    {s}")

# Check 4: URL distribution
print("\n--- Check 4: URL percentage distribution ---")
url_buckets = Counter()
for u in bots + humans:
    ex = create_example_for_ablation(u, 10, "centrality_top", REF, FMT)
    features = extract_post_features(str(getattr(ex, "tweets", "") or ""))
    url_pct = features.get("url_pct", 0)
    if url_pct >= 90: bucket = "90-100%"
    elif url_pct >= 80: bucket = "80-89%"
    elif url_pct >= 70: bucket = "70-79%"
    else: bucket = "60-69%"
    label_tag = u.get("label")
    url_buckets[(bucket, label_tag)] += 1

for b in ["90-100%", "80-89%", "70-79%", "60-69%"]:
    bot_c = url_buckets.get((b, "bot"), 0)
    hum_c = url_buckets.get((b, "human"), 0)
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
print(f"  BOTS:   fol median={statistics.median(bot_fols):.0f}, range={min(bot_fols)}-{max(bot_fols)}")
print(f"  HUMANS: fol median={statistics.median(human_fols):.0f}, range={min(human_fols)}-{max(human_fols)}")

# Check 7: Learning signal
import sqlite3
conn = sqlite3.connect(REPO_ROOT / "results/baseline_trainset/baseline_twibot22.db")
baseline_preds = {uid: pred for uid, _, pred in conn.execute("SELECT user_id, gold_label, pred_label FROM predictions")}
conn.close()
bot_wrong = sum(1 for u in bots if baseline_preds.get(u.get("id")) == "human")
human_wrong = sum(1 for u in humans if baseline_preds.get(u.get("id")) == "bot")
print(f"\n--- Check 7: Learning signal ---")
print(f"  FN bots: {bot_wrong}, FP humans: {human_wrong}")
