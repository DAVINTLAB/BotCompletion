"""Verify the small categories: template, moderate_rt, follow_spam, institutional, support."""

import json
import sys
import datetime
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from twibot.account_classifier import extract_post_features, classify_account
from twibot.context import create_example_for_ablation

REPO_ROOT = Path(__file__).resolve().parents[2]

REF = datetime.datetime(2022, 1, 26, tzinfo=datetime.timezone.utc)
FMT = "%Y-%m-%d %H:%M:%S"

conn = sqlite3.connect(REPO_ROOT / "results/baseline_trainset/baseline_twibot22.db")
baseline_preds = {uid: pred for uid, _, pred in conn.execute("SELECT user_id, gold_label, pred_label FROM predictions")}
conn.close()


def verify(category, expected_type_fragment):
    with open(REPO_ROOT / f"data/gepa_trainsets/v4_curation/{category}.json") as f:
        data = json.load(f)
    bots = data["bots"]
    humans = data["humans"]

    print("=" * 70)
    print(f"CATEGORY: {category.upper()}")
    print("=" * 70)
    print(f"Total: {len(bots)} bots + {len(humans)} humans = {len(bots)+len(humans)}")

    # Check 1: All examples match the expected account_type
    violations = []
    for u in bots + humans:
        v = u.get("verified", False)
        if isinstance(v, str):
            v = v.strip().lower() == "true"
        fol = u.get("followers_count", 0) or 0
        fing = u.get("following_count", 0) or 0
        desc = str(u.get("description", "") or "")
        ex = create_example_for_ablation(u, 10, "centrality_top", REF, FMT)
        features = extract_post_features(str(getattr(ex, "tweets", "") or ""))
        acct = classify_account(
            verified=v, followers=fol, following=fing,
            description=desc, features=features,
        )
        if expected_type_fragment not in acct.lower() and expected_type_fragment != "none":
            violations.append((u.get("id"), acct, u.get("label")))

    if violations:
        print(f"  ✗ VIOLATIONS: {len(violations)}")
        for v in violations[:5]:
            print(f"    {v}")
    else:
        print(f"  ✓ All examples match expected type '{expected_type_fragment}'")

    # Check 2: Duplicates
    ids = [u.get("id") for u in bots + humans]
    dupes = [i for i in set(ids) if ids.count(i) > 1]
    if dupes:
        print(f"  ✗ Duplicates: {dupes}")
    else:
        print(f"  ✓ No duplicates")

    # Check 3: Learning signal
    bot_wrong = sum(1 for u in bots if baseline_preds.get(u.get("id")) == "human")
    human_wrong = sum(1 for u in humans if baseline_preds.get(u.get("id")) == "bot")
    print(f"  Learning signal: {bot_wrong} FN bots, {human_wrong} FP humans")

    # Check 4: Sample data
    print(f"  Sample bots ({min(3, len(bots))}):")
    for u in bots[:3]:
        desc = (u.get("description", "") or "").strip()[:50]
        print(f"    id={u.get('id')} fol={u.get('followers_count',0)} desc=\"{desc}\"")
    print(f"  Sample humans ({min(3, len(humans))}):")
    for u in humans[:3]:
        desc = (u.get("description", "") or "").strip()[:50]
        print(f"    id={u.get('id')} fol={u.get('followers_count',0)} desc=\"{desc}\"")
    print()


verify("template", "template")
verify("moderate_rt", "heavy curator")
verify("follow_spam", "follow-spam")
verify("institutional", "institutional")
verify("support", "support")
