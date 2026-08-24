"""Verify the standard category selections."""

import json
import sys
import datetime
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from twibot.account_classifier import extract_post_features, classify_account
from twibot.context import create_example_for_ablation

REF = datetime.datetime(2022, 1, 26, tzinfo=datetime.timezone.utc)
FMT = "%Y-%m-%d %H:%M:%S"

with open(REPO_ROOT / "data/gepa_trainsets/v4_curation/standard.json") as f:
    data = json.load(f)

bots = data["bots"]
humans = data["humans"]

print("=" * 70)
print("STANDARD CATEGORY VERIFICATION")
print("=" * 70)
print(f"\nTotal: {len(bots)} bots + {len(humans)} humans = {len(bots)+len(humans)}")


# Check 1: Genuinely "standard" (not sparse, not verified, not archetype)
print("\n--- Check 1: All examples are genuinely 'standard' ---")
violations = []
for u in bots + humans:
    v = u.get("verified", False)
    if isinstance(v, str):
        v = v.strip().lower() == "true"
    if v:
        violations.append(("verified", u.get("id")))
        continue

    fol = u.get("followers_count", 0) or 0
    sc = u.get("statuses_count", 0) or 0
    fing = u.get("following_count", 0) or 0

    if fol < 20 and sc < 200:
        violations.append(("sparse", u.get("id"))); continue
    if fol < 10 and fing >= 100:
        violations.append(("follow-spam", u.get("id"))); continue

    ex = create_example_for_ablation(u, 10, "centrality_top", REF, FMT)
    features = extract_post_features(str(getattr(ex, "tweets", "") or ""))
    desc = str(getattr(ex, "description", "") or "")
    acct_type = classify_account(
        verified=False, followers=fol, following=fing,
        description=desc, features=features,
    )
    if acct_type != "standard account":
        violations.append((acct_type, u.get("id")))

if violations:
    print(f"  VIOLATIONS: {len(violations)}")
    for v_type, vid in violations[:10]:
        print(f"    {v_type}: {vid}")
else:
    print(f"  ✓ All are genuinely 'standard'")

# Check 2: Duplicates
print("\n--- Check 2: Duplicates ---")
ids = [u.get("id") for u in bots + humans]
dupes = [i for i in set(ids) if ids.count(i) > 1]
if dupes:
    print(f"  DUPLICATES: {dupes}")
else:
    print(f"  ✓ No duplicates")

# Check 3: Label sanity
print("\n--- Check 3: Label sanity ---")
# Look for suspiciously human-looking bots
susp_bots = []
for u in bots:
    fol = u.get("followers_count", 0) or 0
    sc = u.get("statuses_count", 0) or 0
    desc = (u.get("description", "") or "").strip()
    if sc > 5000 and fol > 5000 and len(desc) > 60:
        susp_bots.append({"id": u.get("id"), "fol": fol, "sc": sc, "desc": desc[:80]})
print(f"  Highly-engaged bots with long bios (suspicious): {len(susp_bots)}")
for s in susp_bots[:5]:
    print(f"    {s}")

# Look for suspiciously bot-like humans
susp_humans = []
for u in humans:
    fol = u.get("followers_count", 0) or 0
    fing = u.get("following_count", 0) or 0
    desc = (u.get("description", "") or "").strip()
    sc = u.get("statuses_count", 0) or 0
    # Empty bio, low followers, high following, low activity
    if fol < 50 and fing > 200 and len(desc) < 10 and sc < 100:
        susp_humans.append({"id": u.get("id"), "fol": fol, "fing": fing, "sc": sc, "desc": desc[:60]})
print(f"\n  Humans with bot-like profile (empty bio, low fol, follow-ish): {len(susp_humans)}")
for s in susp_humans[:5]:
    print(f"    {s}")

# Check 4: Tier distribution
print("\n--- Check 4: Follower tier distribution ---")
bot_tiers = Counter()
for u in bots:
    fol = u.get("followers_count", 0) or 0
    if fol < 200: bot_tiers["small"] += 1
    elif fol < 2000: bot_tiers["mid"] += 1
    else: bot_tiers["high"] += 1

human_tiers = Counter()
for u in humans:
    fol = u.get("followers_count", 0) or 0
    if fol < 200: human_tiers["small"] += 1
    elif fol < 2000: human_tiers["mid"] += 1
    else: human_tiers["high"] += 1

print(f"  Bots:   small={bot_tiers['small']}  mid={bot_tiers['mid']}  high={bot_tiers['high']}")
print(f"  Humans: small={human_tiers['small']}  mid={human_tiers['mid']}  high={human_tiers['high']}")

# Check 5: Stats
print("\n--- Check 5: Profile stats ---")
bot_fols = [u.get("followers_count", 0) or 0 for u in bots]
human_fols = [u.get("followers_count", 0) or 0 for u in humans]
bot_scs = [u.get("statuses_count", 0) or 0 for u in bots]
human_scs = [u.get("statuses_count", 0) or 0 for u in humans]
print(f"  BOTS:   fol median={statistics.median(bot_fols):.0f}  sc median={statistics.median(bot_scs):.0f}")
print(f"  HUMANS: fol median={statistics.median(human_fols):.0f}  sc median={statistics.median(human_scs):.0f}")

# Check 6: Learning signal
print("\n--- Check 6: Baseline-wrong distribution (learning signal) ---")
import sqlite3
conn = sqlite3.connect(REPO_ROOT / "results/baseline_trainset/baseline_twibot22.db")
baseline_preds = {uid: pred for uid, _, pred in conn.execute("SELECT user_id, gold_label, pred_label FROM predictions")}
conn.close()

bot_wrong = sum(1 for u in bots if baseline_preds.get(u.get("id")) == "human")
human_wrong = sum(1 for u in humans if baseline_preds.get(u.get("id")) == "bot")
no_base_bots = sum(1 for u in bots if u.get("id") not in baseline_preds)
no_base_humans = sum(1 for u in humans if u.get("id") not in baseline_preds)

print(f"  FN bots (baseline called them human): {bot_wrong}")
print(f"  FP humans (baseline called them bot): {human_wrong}")
print(f"  No baseline prediction: bots={no_base_bots}, humans={no_base_humans}")

# Check 7: Eval-split disjointness
print("\n--- Check 7: Eval-split disjointness ---")
with open(REPO_ROOT / "data/twibot22/test_clustered.json") as f:
    test = json.load(f)
test_ids = set(u.get("id", u.get("ID")) for u in test)
overlap = [u.get("id") for u in bots + humans if u.get("id") in test_ids]
print(f"  Overlap with the evaluation split: {len(overlap)} (expected 0)")
