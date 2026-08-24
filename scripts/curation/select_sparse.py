"""Carefully select sparse examples for trainset v4.

Target: 40 sparse bots, 6 sparse humans.
Selection principles:
  1. Archetypal examples (clear bot/human signals)
  2. Mix of baseline-correct (guardrail) and baseline-wrong (learning signal)
  3. Variety within group (different subtypes of sparse)
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


def is_sparse(u):
    fol = u.get("followers_count", 0) or 0
    sc = u.get("statuses_count", 0) or 0
    return fol < 20 and sc < 200


def get_fields(u, source):
    uid = u.get("id", u.get("ID", ""))
    return {
        "id": uid,
        "source": source,
        "label": u.get("label", "?"),
        "baseline": baseline_preds.get(uid, "n/a"),
        "fol": u.get("followers_count", 0) or 0,
        "fing": u.get("following_count", 0) or 0,
        "sc": u.get("statuses_count", 0) or 0,
        "n_posts": len(u.get("tweets", []) or []),
        "desc": (u.get("description", "") or "").strip(),
        "data": u,
    }


# Collect all sparse candidates
candidates = []
for u in train_pool:
    if is_sparse(u):
        candidates.append(get_fields(u, "train"))

print(f"Total sparse candidates: {len(candidates)}")

# --- SPARSE BOTS: target 40 ---
sparse_bots = [c for c in candidates if c["label"] == "bot"]
print(f"\nSparse bot candidates: {len(sparse_bots)}")

# Sub-categorize by archetype
archetype_empty_nopost = []  # Empty bio, 0 posts — classic
archetype_empty_fewpost = []  # Empty bio, 1-50 posts
archetype_bio_nopost = []  # Has bio, 0 posts
archetype_bio_fewpost = []  # Has bio, few posts
archetype_followspam = []  # <5 followers, 100+ following

for c in sparse_bots:
    has_bio = len(c["desc"]) > 5
    has_posts = c["n_posts"] > 0
    follow_spam = c["fol"] < 5 and c["fing"] >= 100

    if follow_spam and not has_bio:
        archetype_followspam.append(c)
    elif not has_bio and not has_posts:
        archetype_empty_nopost.append(c)
    elif not has_bio and has_posts:
        archetype_empty_fewpost.append(c)
    elif has_bio and not has_posts:
        archetype_bio_nopost.append(c)
    else:
        archetype_bio_fewpost.append(c)

print(f"\nArchetype breakdown:")
print(f"  follow-spam (<5 fol, 100+ fing, no bio): {len(archetype_followspam)}")
print(f"  empty bio, 0 posts: {len(archetype_empty_nopost)}")
print(f"  empty bio, some posts: {len(archetype_empty_fewpost)}")
print(f"  has bio, 0 posts: {len(archetype_bio_nopost)}")
print(f"  has bio, some posts: {len(archetype_bio_fewpost)}")

# Allocation: spread across archetypes proportionally
# Target 40 bots, reserve space for each archetype
target_allocations = {
    "followspam": 10,
    "empty_nopost": 15,
    "empty_fewpost": 8,
    "bio_nopost": 4,
    "bio_fewpost": 3,
}

archetype_pools = {
    "followspam": archetype_followspam,
    "empty_nopost": archetype_empty_nopost,
    "empty_fewpost": archetype_empty_fewpost,
    "bio_nopost": archetype_bio_nopost,
    "bio_fewpost": archetype_bio_fewpost,
}

selected_bots = []
for arch, target in target_allocations.items():
    pool = archetype_pools[arch]
    if len(pool) == 0:
        print(f"  WARN: {arch} pool is empty")
        continue
    # Prefer mix of baseline-correct and baseline-wrong
    correct = [c for c in pool if c["baseline"] == "bot"]
    wrong = [c for c in pool if c["baseline"] == "human"]
    no_baseline = [c for c in pool if c["baseline"] == "n/a"]  # no baseline prediction

    # Try for 50/50 correct/wrong, with no_baseline filling gaps
    n_correct = min(target // 2, len(correct))
    n_wrong = min(target - n_correct, len(wrong))
    n_filled = n_correct + n_wrong
    n_fill = target - n_filled
    n_nobase = min(n_fill, len(no_baseline))

    picked = (
        random.sample(correct, n_correct)
        + random.sample(wrong, n_wrong)
        + random.sample(no_baseline, n_nobase)
    )
    selected_bots.extend(picked)
    print(f"  {arch}: picked {len(picked)} (correct={n_correct}, wrong={n_wrong}, from_test={n_nobase})")

print(f"\nTotal sparse bots selected: {len(selected_bots)}")

# --- SPARSE HUMANS: target 6 ---
sparse_humans = [c for c in candidates if c["label"] == "human"]
print(f"\nSparse human candidates: {len(sparse_humans)}")

# Prefer humans WITH personal bios (teach restraint — small accounts with bios are often humans)
# And humans the baseline got wrong (learning signal)
humans_with_bio = [c for c in sparse_humans if len(c["desc"]) > 10]
humans_fp = [c for c in sparse_humans if c["baseline"] == "bot"]
print(f"  With bio (>10 chars): {len(humans_with_bio)}")
print(f"  Baseline wrong (FP): {len(humans_fp)}")

# Pick a balanced set: ~3 FP humans with bio, ~3 correct humans for diversity
fp_with_bio = [c for c in humans_fp if len(c["desc"]) > 10]
correct_humans = [c for c in sparse_humans if c["baseline"] == "human" and len(c["desc"]) > 10]
no_baseline_humans = [c for c in sparse_humans if c["baseline"] == "n/a" and len(c["desc"]) > 10]

selected_humans = []
selected_humans.extend(random.sample(fp_with_bio, min(2, len(fp_with_bio))))
selected_humans.extend(random.sample(correct_humans, min(2, len(correct_humans))))
selected_humans.extend(random.sample(no_baseline_humans, min(2, len(no_baseline_humans))))
# Top up to 6
remaining = [c for c in sparse_humans if c not in selected_humans]
while len(selected_humans) < 6 and remaining:
    selected_humans.append(remaining.pop())

print(f"\nTotal sparse humans selected: {len(selected_humans)}")

# Print selected
print("\n" + "=" * 70)
print("SELECTED SPARSE BOTS")
print("=" * 70)
for c in selected_bots:
    print(f"  [{c['source']}] id={c['id']:<22} baseline={c['baseline']:<6} fol={c['fol']:>3} fing={c['fing']:>5} sc={c['sc']:>4} posts={c['n_posts']:>3} desc=\"{c['desc'][:55]}\"")

print("\n" + "=" * 70)
print("SELECTED SPARSE HUMANS")
print("=" * 70)
for c in selected_humans:
    print(f"  [{c['source']}] id={c['id']:<22} baseline={c['baseline']:<6} fol={c['fol']:>3} fing={c['fing']:>5} sc={c['sc']:>4} posts={c['n_posts']:>3} desc=\"{c['desc'][:55]}\"")

# Save to JSON for later assembly
out = {
    "category": "sparse",
    "bots": [c["data"] for c in selected_bots],
    "humans": [c["data"] for c in selected_humans],
    "meta": {
        "bot_count": len(selected_bots),
        "human_count": len(selected_humans),
        "bot_ids": [c["id"] for c in selected_bots],
        "human_ids": [c["id"] for c in selected_humans],
    },
}

out_path = Path(REPO_ROOT / "data/gepa_trainsets/v4_curation/sparse.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\nSaved to {out_path}")
