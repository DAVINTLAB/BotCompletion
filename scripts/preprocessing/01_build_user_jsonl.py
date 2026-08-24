#!/usr/bin/env python3
"""Step 1 — Convert the raw TwiBot-22 dump into per-user JSONL records.

Reads the raw TwiBot-22 release (obtained from its authors — see
data/README.md): label.csv, split.csv, user.json, and tweet_0..8.json.
Joins profile metadata, tweet texts, and labels into one record per user and
writes train.jsonl / val.jsonl / test.jsonl following TwiBot-22's own splits.

Output schema per line:
    {id, name, screen_name, followers_count, following_count,
     favourites_count, statuses_count, created_at, description,
     protected, verified, tweets, label}

Usage:
    python 01_build_user_jsonl.py --raw-dir /path/to/twibot-22 --out-dir data/twibot22

Note: the tweet files total ~100 GB; each is streamed with ijson so peak
memory stays bounded by the retained users' tweets.
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict

import ijson
import pandas as pd
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, type=Path,
                    help="Directory with the raw TwiBot-22 files (label.csv, split.csv, user.json, tweet_0..8.json)")
    ap.add_argument("--out-dir", default=Path("data/twibot22"), type=Path,
                    help="Output directory for train/val/test.jsonl")
    ap.add_argument("--tweet-files", type=int, default=9,
                    help="Number of tweet_X.json files to scan (default 9 = all)")
    args = ap.parse_args()

    raw = args.raw_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading label.csv ...")
    df_label = pd.read_csv(raw / "label.csv")  # columns: [id, label]
    label_map = dict(zip(df_label["id"], df_label["label"]))
    del df_label

    print("Loading split.csv ...")
    df_split = pd.read_csv(raw / "split.csv")  # columns: [id, split]
    split_map = dict(zip(df_split["id"], df_split["split"]))
    del df_split

    print("Streaming user.json ...")
    user_data = {}
    with open(raw / "user.json", "rb") as f:
        for user in tqdm(ijson.items(f, "item"), desc="Reading user.json"):
            user_id_full = user.get("id", "")  # e.g. "u1234567890123456789"
            if not user_id_full:
                continue
            if user_id_full not in label_map or user_id_full not in split_map:
                continue
            # Strip the leading "u" for matching against tweet author_id.
            numeric_id = user_id_full.lstrip("u")
            metrics = user.get("public_metrics") or {}
            user_data[numeric_id] = {
                "id": user_id_full,
                "name": user.get("name", ""),
                "screen_name": user.get("username", ""),
                "followers_count": metrics.get("followers_count", 0),
                "following_count": metrics.get("following_count", 0),
                # TwiBot-22 has no favourites_count; store 0 for schema parity.
                "favourites_count": 0,
                "statuses_count": metrics.get("tweet_count", 0),
                "created_at": user.get("created_at", ""),
                "description": user.get("description", ""),
                "protected": user.get("protected", False),
                "verified": user.get("verified", False),
                "tweets": [],
                "label": "bot" if label_map[user_id_full] == "bot" else "human",
                "split": split_map[user_id_full],
            }
    print(f"Retained {len(user_data):,} labeled users")

    print("Streaming tweet_*.json files ...")
    tweets_by_user = defaultdict(list)
    for i in range(args.tweet_files):
        tfile = raw / f"tweet_{i}.json"
        if not tfile.exists():
            print(f"  [skip] {tfile} not found")
            continue
        print(f"  -> Reading {tfile.name}")
        with open(tfile, "rb") as f_tweet:
            for tweet in tqdm(ijson.items(f_tweet, "item"),
                              desc=f"  tweets in file {i}", leave=False):
                author_id = str(tweet.get("author_id", ""))
                if author_id in user_data:
                    tweets_by_user[author_id].append(tweet.get("text", ""))

    print("Merging tweet texts into user records ...")
    for user_id_num, texts in tweets_by_user.items():
        user_data[user_id_num]["tweets"] = texts
    del tweets_by_user

    print("Writing train.jsonl / val.jsonl / test.jsonl ...")
    files = {
        "train": open(args.out_dir / "train.jsonl", "w", encoding="utf-8"),
        "val": open(args.out_dir / "val.jsonl", "w", encoding="utf-8"),
        "test": open(args.out_dir / "test.jsonl", "w", encoding="utf-8"),
    }
    counts = defaultdict(int)
    for record in tqdm(user_data.values(), desc="Writing splits"):
        split_type = record.pop("split")
        if split_type in files:
            files[split_type].write(json.dumps(record) + "\n")
            counts[split_type] += 1
    for f in files.values():
        f.close()

    print(f"Done: " + ", ".join(f"{k}={v:,}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
