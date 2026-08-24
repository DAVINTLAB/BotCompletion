"""Select the remaining small categories for trainset v4.

Targets:
  - template bot: 6 bot, 7 human = 13
  - moderate-RT: 6 bot, 5 human = 11
  - follow-spam: 8 bot, 0 human = 8
  - institutional: 0 bot, 4 human = 4
  - support: 1 bot, 0 human = 1

Selection principle: full diversity within each small group, prefer baseline-wrong.
"""

import json
import sqlite3
import random
import sys
import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
random.seed(42)

from twibot.account_classifier import extract_post_features, classify_account
from twibot.context import create_example_for_ablation

REF = datetime.datetime(2022, 1, 26, tzinfo=datetime.timezone.utc)
FMT = "%Y-%m-%d %H:%M:%S"

with open(REPO_ROOT / "data/twibot22/train_clustered.json") as f:
    train_pool = json.load(f)

conn = sqlite3.connect(REPO_ROOT / "results/baseline_trainset/baseline_twibot22.db")
baseline_preds = {}
for uid, gold, pred in conn.execute("SELECT user_id, gold_label, pred_label FROM predictions"):
    baseline_preds[uid] = pred
conn.close()


def get_fields(u, source):
    uid = u.get("id", u.get("ID", ""))
    v = u.get("verified", False)
    if isinstance(v, str):
        v = v.strip().lower() == "true"
    fol = u.get("followers_count", 0) or 0
    fing = u.get("following_count", 0) or 0
    desc = str(u.get("description", "") or "")

    ex = create_example_for_ablation(u, 10, "centrality_top", REF, FMT)
    features = extract_post_features(str(getattr(ex, "tweets", "") or ""))
    acct_type = classify_account(
        verified=v, followers=fol, following=fing,
        description=desc, features=features,
    )
    return {
        "id": uid, "source": source, "label": u.get("label", "?"),
        "baseline": baseline_preds.get(uid, "n/a"),
        "fol": fol, "fing": fing,
        "sc": u.get("statuses_count", 0) or 0,
        "n_posts": len(u.get("tweets", []) or []),
        "desc": desc.strip(),
        "acct_type": acct_type,
        "verified": v,
        "features": features,
        "data": u,
    }


print("Scanning pools for small categories...")
all_candidates = []
for u in train_pool:
    all_candidates.append(get_fields(u, "train"))
print(f"Total candidates scanned: {len(all_candidates)}")


def allocate(items, target, prefer_wrong=True):
    if not items or target <= 0:
        return []
    correct = [c for c in items if c["baseline"] == c["label"]]
    wrong = [c for c in items if c["baseline"] != c["label"] and c["baseline"] != "n/a"]
    no_base = [c for c in items if c["baseline"] == "n/a"]

    if prefer_wrong:
        n_wrong = min(max(1, target // 2) if target > 0 else 0, len(wrong))
        n_correct = min(max(0, target - n_wrong), len(correct))
    else:
        n_correct = min(target // 2, len(correct))
        n_wrong = min(max(0, target - n_correct), len(wrong))

    n_fill = target - n_correct - n_wrong
    n_nobase = min(n_fill, len(no_base))

    picked = (
        random.sample(correct, n_correct)
        + random.sample(wrong, n_wrong)
        + random.sample(no_base, n_nobase)
    )
    remaining = target - len(picked)
    if remaining > 0:
        leftover = [c for c in items if c not in picked]
        if leftover:
            picked.extend(random.sample(leftover, min(remaining, len(leftover))))
    return picked


def pick_category(filter_fn, bot_target, human_target, label_name):
    pool = [c for c in all_candidates if filter_fn(c)]
    bots = [c for c in pool if c["label"] == "bot"]
    humans = [c for c in pool if c["label"] == "human"]
    print(f"\n{label_name}: pool has {len(bots)} bots, {len(humans)} humans")

    sel_bots = allocate(bots, bot_target, prefer_wrong=True)
    sel_humans = allocate(humans, human_target, prefer_wrong=True)
    print(f"  Selected: {len(sel_bots)} bots (target {bot_target}), {len(sel_humans)} humans (target {human_target})")
    return sel_bots, sel_humans


# --- TEMPLATE ---
template_bots, template_humans = pick_category(
    lambda c: "template" in c["acct_type"] and not c["verified"],
    bot_target=6, human_target=7, label_name="Template",
)

# --- MODERATE-RT ---
moderate_rt_bots, moderate_rt_humans = pick_category(
    lambda c: "heavy curator" in c["acct_type"] and not c["verified"],
    bot_target=6, human_target=5, label_name="Moderate-RT",
)

# --- FOLLOW-SPAM ---
follow_spam_bots, follow_spam_humans = pick_category(
    lambda c: "follow-spam" in c["acct_type"] and not c["verified"],
    bot_target=8, human_target=0, label_name="Follow-spam",
)

# --- INSTITUTIONAL ---
inst_bots, inst_humans = pick_category(
    lambda c: "institutional" in c["acct_type"] and not c["verified"],
    bot_target=0, human_target=4, label_name="Institutional",
)

# --- SUPPORT ---
support_bots, support_humans = pick_category(
    lambda c: "support" in c["acct_type"] and not c["verified"],
    bot_target=1, human_target=0, label_name="Support",
)


def save_category(name, bots, humans):
    out = {
        "category": name,
        "bots": [c["data"] for c in bots],
        "humans": [c["data"] for c in humans],
        "meta": {
            "bot_count": len(bots),
            "human_count": len(humans),
            "bot_ids": [c["id"] for c in bots],
            "human_ids": [c["id"] for c in humans],
        },
    }
    out_path = Path(REPO_ROOT / f"data/gepa_trainsets/v4_curation/{name}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)


save_category("template", template_bots, template_humans)
save_category("moderate_rt", moderate_rt_bots, moderate_rt_humans)
save_category("follow_spam", follow_spam_bots, follow_spam_humans)
save_category("institutional", inst_bots, inst_humans)
save_category("support", support_bots, support_humans)

print("\n" + "=" * 70)
print("SMALL CATEGORIES COMPLETE")
print("=" * 70)
total_bots = len(template_bots) + len(moderate_rt_bots) + len(follow_spam_bots) + len(inst_bots) + len(support_bots)
total_humans = len(template_humans) + len(moderate_rt_humans) + len(follow_spam_humans) + len(inst_humans) + len(support_humans)
print(f"Small categories combined: {total_bots} bots + {total_humans} humans = {total_bots + total_humans}")
