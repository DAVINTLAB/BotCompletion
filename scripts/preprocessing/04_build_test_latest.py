#!/usr/bin/env python3
"""Step 4 — Build data/twibot22/test_latest.json for the Latest selection
strategy: same schema as test_clustered.json but with each user's tweets
sorted newest-first (using created_at from the raw tweet_X.json files).

Streams the 9 raw tweet files (~100 GB) with ijson, filters to the 340
eval-split author_ids, and rewrites test_clustered.json with the per-user
tweets list reordered.

Usage:
    python 04_build_test_latest.py --raw-dir /path/to/twibot-22

Set --tweet-files N for partial smoke tests (default 9 = all files).
"""
import json
import time
import argparse
from pathlib import Path

import ijson

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_TEST = PROJECT_ROOT / "data" / "twibot22" / "test_clustered.json"
DST_TEST = PROJECT_ROOT / "data" / "twibot22" / "test_latest.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, type=Path,
                    help="Directory with the raw TwiBot-22 tweet_0..8.json files")
    ap.add_argument("--tweet-files", type=int, default=9,
                    help="Number of tweet_X.json files to scan (default 9 = all)")
    args = ap.parse_args()
    tweet_files = [args.raw_dir / f"tweet_{i}.json" for i in range(args.tweet_files)]

    # 1. Load test user IDs from existing clustered file
    with open(SRC_TEST) as f:
        test_users = json.load(f)
    # IDs in test_clustered.json look like "u1234567890123456789"; strip 'u' prefix
    # because tweet_X.json uses int author_id.
    user_id_to_record_idx = {}
    for idx, u in enumerate(test_users):
        raw = u["id"]
        clean = raw[1:] if raw.startswith("u") else raw
        user_id_to_record_idx[str(int(clean))] = idx
    print(f"Loaded {len(test_users)} test users (id strip done)")

    # 2. Stream tweet files, accumulate (created_at, text) per author_id
    per_user_tweets = {uid: [] for uid in user_id_to_record_idx}
    total_seen = 0
    total_kept = 0
    t_start = time.time()

    for fi, tfile in enumerate(tweet_files):
        if not tfile.exists():
            print(f"[skip] {tfile} not found")
            continue
        size_gb = tfile.stat().st_size / 1e9
        print(f"\n[{fi+1}/{len(tweet_files)}] {tfile.name}  ({size_gb:.1f} GB)")
        t_file = time.time()
        kept_in_file = 0
        seen_in_file = 0
        with open(tfile, "rb") as f:
            for tweet in ijson.items(f, "item"):
                seen_in_file += 1
                if seen_in_file % 1_000_000 == 0:
                    elapsed = time.time() - t_file
                    rate = seen_in_file / elapsed if elapsed else 0
                    print(f"  ... {seen_in_file:,} tweets ({rate:,.0f}/s, kept {kept_in_file:,})", flush=True)

                aid = tweet.get("author_id")
                if aid is None:
                    continue
                aid_str = str(aid)
                if aid_str not in per_user_tweets:
                    continue

                ts = tweet.get("created_at")
                txt = tweet.get("text")
                if ts is None or txt is None:
                    continue

                per_user_tweets[aid_str].append((ts, txt))
                kept_in_file += 1

        elapsed = time.time() - t_file
        total_seen += seen_in_file
        total_kept += kept_in_file
        print(f"  done in {elapsed:.0f}s — saw {seen_in_file:,}, kept {kept_in_file:,} for our {len(test_users)} users")

    overall = time.time() - t_start
    print(f"\n=== TOTAL: saw {total_seen:,} tweets, kept {total_kept:,} in {overall:.0f}s ===")

    # 3. Sort each user's tweets newest-first, write out
    output = []
    nonzero = 0
    zero_users = []
    for u in test_users:
        raw = u["id"]
        clean = str(int(raw[1:] if raw.startswith("u") else raw))
        u_copy = dict(u)  # shallow copy; we'll replace 'tweets'
        ts_text = per_user_tweets.get(clean, [])
        # Sort by created_at descending (newest first); created_at is a string in
        # ISO-ish format that sorts lexicographically as time.
        ts_text.sort(key=lambda p: p[0], reverse=True)
        u_copy["tweets"] = [t[1] for t in ts_text]
        # Drop cluster-related fields since they no longer apply
        u_copy.pop("cluster_order", None)
        u_copy.pop("cluster_info", None)
        output.append(u_copy)
        if ts_text:
            nonzero += 1
        else:
            zero_users.append(u["screen_name"])

    print(f"Users with tweets after rewrite: {nonzero}/{len(test_users)}")
    if zero_users:
        print(f"Users with 0 tweets: {len(zero_users)} (e.g. {zero_users[:5]})")

    DST_TEST.parent.mkdir(parents=True, exist_ok=True)
    with open(DST_TEST, "w") as f:
        json.dump(output, f)
    print(f"\nWrote {DST_TEST} ({DST_TEST.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
