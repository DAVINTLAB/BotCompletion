"""Verify the verified category selections."""

import json
import statistics
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

with open(REPO_ROOT / "data/gepa_trainsets/v4_curation/verified.json") as f:
    data = json.load(f)

bots = data["bots"]
humans = data["humans"]

print("=" * 70)
print("VERIFIED CATEGORY VERIFICATION")
print("=" * 70)
print(f"\nTotal: {len(bots)} bots + {len(humans)} humans = {len(bots)+len(humans)}")

# Check 1: All are verified
print("\n--- Check 1: All marked as verified ---")
violations = []
for u in bots + humans:
    v = u.get("verified", False)
    if isinstance(v, str):
        v = v.strip().lower() == "true"
    if not v:
        violations.append(u.get("id"))
if violations:
    print(f"  VIOLATIONS (not verified): {violations}")
else:
    print(f"  ✓ All {len(bots)+len(humans)} are verified")

# Check 2: No bots (all verified should be human)
print("\n--- Check 2: Zero bots expected ---")
if len(bots) == 0:
    print(f"  ✓ 0 bots (correct — no verified bots in TwiBot-22)")
else:
    print(f"  WARNING: {len(bots)} bots found — check for mislabels")

# Check 3: Duplicates
print("\n--- Check 3: Duplicates ---")
ids = [u.get("id") for u in bots + humans]
dupes = [i for i in set(ids) if ids.count(i) > 1]
if dupes:
    print(f"  DUPLICATES: {dupes}")
else:
    print(f"  ✓ No duplicates")

# Check 4: Tier distribution (celebrities / mid / small)
print("\n--- Check 4: Follower tier distribution ---")
tiers = Counter()
for u in humans:
    fol = u.get("followers_count", 0) or 0
    if fol > 100_000: tiers["celebs (>100k)"] += 1
    elif fol > 10_000: tiers["mid (10k-100k)"] += 1
    else: tiers["small (<10k)"] += 1
for k in ["celebs (>100k)", "mid (10k-100k)", "small (<10k)"]:
    print(f"  {k}: {tiers[k]}")

# Check 5: Diversity - look at bio types
print("\n--- Check 5: Bio/account-type diversity (sample) ---")
keywords = Counter()
for u in humans:
    desc = (u.get("description", "") or "").lower()
    if "news" in desc or "journalist" in desc or "reporter" in desc or "editor" in desc:
        keywords["news/journalism"] += 1
    if "actor" in desc or "artist" in desc or "musician" in desc or "producer" in desc:
        keywords["entertainment"] += 1
    if "ceo" in desc or "founder" in desc or "director" in desc or "executive" in desc:
        keywords["business-leader"] += 1
    if "official" in desc:
        keywords["official-account"] += 1
    if "professor" in desc or "phd" in desc or "academic" in desc:
        keywords["academic"] += 1
    if "athlete" in desc or "soccer" in desc or "football" in desc or "basketball" in desc:
        keywords["sports"] += 1
    if "politic" in desc or "senator" in desc or "congress" in desc:
        keywords["politics"] += 1
for k, v in keywords.most_common():
    print(f"  {k}: {v}")

# Check 6: Stats
print("\n--- Check 6: Profile stats ---")
fols = [u.get("followers_count", 0) or 0 for u in humans]
fings = [u.get("following_count", 0) or 0 for u in humans]
scs = [u.get("statuses_count", 0) or 0 for u in humans]
print(f"  fol: median={statistics.median(fols):.0f}  range={min(fols)}-{max(fols)}")
print(f"  fing: median={statistics.median(fings):.0f}  range={min(fings)}-{max(fings)}")
print(f"  sc: median={statistics.median(scs):.0f}  range={min(scs)}-{max(scs)}")

# Check 7: Are any "obviously mislabeled" (empty profiles, no activity)
print("\n--- Check 7: Any suspicious humans? ---")
suspicious = []
for u in humans:
    desc = (u.get("description", "") or "").strip()
    sc = u.get("statuses_count", 0) or 0
    fol = u.get("followers_count", 0) or 0
    # Suspicious: verified but practically empty/inactive
    if sc < 100 and len(desc) < 15 and fol < 10_000:
        suspicious.append({"id": u.get("id"), "fol": fol, "sc": sc, "desc": desc[:60]})
if suspicious:
    print(f"  Found {len(suspicious)} suspicious verified humans:")
    for s in suspicious:
        print(f"    {s}")
else:
    print(f"  ✓ No obviously suspicious verified humans")

# Check 8: Show the 14 FP examples (where baseline broke the rule)
print("\n--- Check 8: FP verified examples (where baseline broke hard rule) ---")
print("  These are the valuable learning examples:")
import sqlite3
conn = sqlite3.connect(REPO_ROOT / "results/baseline_trainset/baseline_twibot22.db")
baseline_preds = {uid: pred for uid, _, pred in conn.execute("SELECT user_id, gold_label, pred_label FROM predictions")}
conn.close()

fp_count = 0
for u in humans:
    uid = u.get("id")
    bp = baseline_preds.get(uid)
    if bp == "bot":
        fp_count += 1
        if fp_count <= 5:
            print(f"    [baseline flagged as bot] id={uid} fol={u.get('followers_count',0)} desc=\"{(u.get('description','') or '')[:60]}\"")
print(f"  Total: {fp_count} FP verified examples in selection")
