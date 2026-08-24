"""Verify the sparse category selections.

Checks:
1. Every selected example actually meets sparse criteria (fol<20, sc<200)
2. Label is genuine (spot-check for obvious mislabels)
3. Proportions are reasonable
4. No duplicates
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

with open(REPO_ROOT / "data/gepa_trainsets/v4_curation/sparse.json") as f:
    data = json.load(f)

bots = data["bots"]
humans = data["humans"]

print("=" * 70)
print("SPARSE CATEGORY VERIFICATION")
print("=" * 70)
print(f"\nTotal: {len(bots)} bots + {len(humans)} humans = {len(bots)+len(humans)}")

# Check 1: All meet sparse criteria
print("\n--- Check 1: Sparse criteria (fol<20, sc<200) ---")
violations = []
for u in bots + humans:
    fol = u.get("followers_count", 0) or 0
    sc = u.get("statuses_count", 0) or 0
    if fol >= 20 or sc >= 200:
        violations.append((u.get("id"), fol, sc, u.get("label")))
if violations:
    print(f"  VIOLATIONS: {len(violations)}")
    for v in violations:
        print(f"    {v}")
else:
    print(f"  ✓ All {len(bots)+len(humans)} examples meet criteria")

# Check 2: Duplicates
print("\n--- Check 2: Duplicates ---")
ids = [u.get("id") for u in bots + humans]
dupes = [i for i in set(ids) if ids.count(i) > 1]
if dupes:
    print(f"  DUPLICATES: {dupes}")
else:
    print(f"  ✓ No duplicates")

# Check 3: Sanity check on bots — are they genuinely sparse bots?
print("\n--- Check 3: Bot label sanity (looking for obvious mislabels) ---")
suspicious_bots = []
for u in bots:
    fol = u.get("followers_count", 0) or 0
    desc = (u.get("description", "") or "").strip()
    sc = u.get("statuses_count", 0) or 0
    n_tweets = len(u.get("tweets", []) or [])

    # Bots with substantial content + meaningful bio are suspicious
    if sc > 50 and len(desc) > 30 and n_tweets > 20:
        suspicious_bots.append({
            "id": u.get("id"),
            "fol": fol, "sc": sc, "n_tweets": n_tweets,
            "desc": desc[:80],
        })

print(f"  Found {len(suspicious_bots)} bots with substantial content+bio (potential mislabel):")
for s in suspicious_bots:
    print(f"    id={s['id']} fol={s['fol']} sc={s['sc']} posts={s['n_tweets']} desc=\"{s['desc']}\"")

# Check 4: Humans — are they genuinely humans?
print("\n--- Check 4: Human label sanity ---")
suspicious_humans = []
for u in humans:
    fol = u.get("followers_count", 0) or 0
    desc = (u.get("description", "") or "").strip()
    sc = u.get("statuses_count", 0) or 0
    n_tweets = len(u.get("tweets", []) or [])
    fing = u.get("following_count", 0) or 0

    # Humans with zero activity AND classic follow-spam pattern are suspicious (probably bots)
    if fol == 0 and sc == 0 and n_tweets == 0 and fing > 50:
        suspicious_humans.append({
            "id": u.get("id"),
            "fol": fol, "fing": fing, "sc": sc,
            "desc": desc[:80],
        })

print(f"  Found {len(suspicious_humans)} humans with classic bot-like profile (potential mislabel):")
for s in suspicious_humans:
    print(f"    id={s['id']} fol={s['fol']} fing={s['fing']} sc={s['sc']} desc=\"{s['desc']}\"")

# Check 5: Archetype distribution
print("\n--- Check 5: Archetype distribution ---")
from collections import Counter
archetypes = Counter()
for u in bots:
    fol = u.get("followers_count", 0) or 0
    fing = u.get("following_count", 0) or 0
    desc = (u.get("description", "") or "").strip()
    sc = u.get("statuses_count", 0) or 0
    n_tweets = len(u.get("tweets", []) or [])
    has_bio = len(desc) > 5
    has_posts = n_tweets > 0
    follow_spam = fol < 5 and fing >= 100
    if follow_spam and not has_bio:
        archetypes["follow-spam-no-bio"] += 1
    elif not has_bio and not has_posts:
        archetypes["empty-no-posts"] += 1
    elif not has_bio and has_posts:
        archetypes["empty-some-posts"] += 1
    elif has_bio and not has_posts:
        archetypes["bio-no-posts"] += 1
    else:
        archetypes["bio-some-posts"] += 1
print(f"  Bot archetypes:")
for k, v in archetypes.most_common():
    print(f"    {k}: {v}")

# Check 6: Overall profile stats
print("\n--- Check 6: Overall profile stats ---")
import statistics
bot_fols = [u.get("followers_count", 0) or 0 for u in bots]
bot_fings = [u.get("following_count", 0) or 0 for u in bots]
bot_scs = [u.get("statuses_count", 0) or 0 for u in bots]
human_fols = [u.get("followers_count", 0) or 0 for u in humans]
human_fings = [u.get("following_count", 0) or 0 for u in humans]
human_scs = [u.get("statuses_count", 0) or 0 for u in humans]

print(f"  BOTS:   fol median={statistics.median(bot_fols):.0f}  fing median={statistics.median(bot_fings):.0f}  sc median={statistics.median(bot_scs):.0f}")
print(f"  HUMANS: fol median={statistics.median(human_fols):.0f}  fing median={statistics.median(human_fings):.0f}  sc median={statistics.median(human_scs):.0f}")

# Compare to test set sparse for reference
print("\n--- Check 7: Compare to test sparse distribution ---")
with open(REPO_ROOT / "data/twibot22/test_clustered.json") as f:
    test = json.load(f)

test_sparse_bots = [u for u in test if u.get("label") == "bot"
                    and (u.get("followers_count",0) or 0) < 20
                    and (u.get("statuses_count",0) or 0) < 200]
test_sparse_humans = [u for u in test if u.get("label") == "human"
                      and (u.get("followers_count",0) or 0) < 20
                      and (u.get("statuses_count",0) or 0) < 200]

print(f"  Test set sparse: {len(test_sparse_bots)} bots, {len(test_sparse_humans)} humans")
if test_sparse_bots:
    tbf = [u.get("followers_count", 0) or 0 for u in test_sparse_bots]
    tbs = [u.get("statuses_count", 0) or 0 for u in test_sparse_bots]
    print(f"  Test bot medians:   fol={statistics.median(tbf):.0f}  sc={statistics.median(tbs):.0f}")
if test_sparse_humans:
    thf = [u.get("followers_count", 0) or 0 for u in test_sparse_humans]
    ths = [u.get("statuses_count", 0) or 0 for u in test_sparse_humans]
    print(f"  Test human medians: fol={statistics.median(thf):.0f}  sc={statistics.median(ths):.0f}")
