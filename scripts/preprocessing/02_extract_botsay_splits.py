#!/usr/bin/env python3
"""Step 2 — Extract BotSay's TwiBot-22 splits from the per-user JSONL files.

BotSay (Feng et al., ACL 2024) released a split_new.json for TwiBot-22 with
a 2,694-user training pool and the 340-user balanced evaluation split the
paper calls BotSay-340. This script filters the train.jsonl / test.jsonl
records produced by 01_build_user_jsonl.py down to those user IDs.

split_new.json comes from BotSay's data.zip (see data/README.md) and has:
    {"train": ["u1234567890", ...], "test": ["u9876543210", ...]}

Usage:
    python 02_extract_botsay_splits.py \
        --split-json /path/to/botsay/data/Twibot-22/split_new.json \
        --jsonl-dir data/twibot22 --out-dir data/twibot22

Outputs: train_botsay.json (2,694 users), test_botsay.json (340 users) —
each a JSON array of user records.
"""
import json
import argparse
from pathlib import Path

from tqdm import tqdm


def filter_jsonl(jsonl_path, valid_ids):
    results = []
    with open(jsonl_path, "r", encoding="utf-8") as in_f:
        for line in tqdm(in_f, desc=f"Filtering {jsonl_path}", unit=" lines"):
            obj = json.loads(line.strip())
            if obj.get("id") in valid_ids:
                results.append(obj)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-json", required=True, type=Path,
                    help="Path to BotSay's split_new.json for Twibot-22")
    ap.add_argument("--jsonl-dir", default=Path("data/twibot22"), type=Path,
                    help="Directory with train.jsonl / test.jsonl from step 1")
    ap.add_argument("--out-dir", default=Path("data/twibot22"), type=Path)
    args = ap.parse_args()

    with open(args.split_json, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    train_ids = set(split_data["train"])
    test_ids = set(split_data["test"])
    print(f"BotSay split: {len(train_ids)} train ids, {len(test_ids)} test ids")

    # BotSay's train pool is drawn from TwiBot-22's train split and its
    # 340-user eval set from TwiBot-22's test split; scan both files with the
    # union of ids so records are found regardless of which file holds them.
    unified = train_ids | test_ids
    found = []
    for name in ("train.jsonl", "test.jsonl"):
        path = args.jsonl_dir / name
        if path.exists():
            found.extend(filter_jsonl(path, unified))
        else:
            print(f"  [skip] {path} not found")

    by_id = {u["id"]: u for u in found}
    train_sample = [by_id[i] for i in split_data["train"] if i in by_id]
    test_sample = [by_id[i] for i in split_data["test"] if i in by_id]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_out = args.out_dir / "train_botsay.json"
    test_out = args.out_dir / "test_botsay.json"
    with open(train_out, "w", encoding="utf-8") as f:
        json.dump(train_sample, f, indent=2)
    with open(test_out, "w", encoding="utf-8") as f:
        json.dump(test_sample, f, indent=2)

    print(f"Train pool: {len(train_sample)} users -> {train_out}")
    print(f"Eval split: {len(test_sample)} users -> {test_out}")
    missing = unified - set(by_id)
    if missing:
        print(f"WARNING: {len(missing)} split ids not found in the JSONL files")


if __name__ == "__main__":
    main()
