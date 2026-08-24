"""Build data/gepa_trainsets/gepa_trainset_twibot22_v4_latest.json — same
schema as gepa_trainset_twibot22_v4.json but with tweets sorted newest-first
per user (using created_at from raw tweet_X.json files).

Streams the 9 raw tweet files (~100GB) with ijson, filters to the curated
trainset's author_ids, and rewrites with per-user tweets list reordered.
"""
import json
import time
import os
from pathlib import Path
import ijson

PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DIR = Path(os.environ.get("TWIBOT22_RAW_DIR", "data/raw/twibot-22"))
SRC = PROJECT_ROOT / "data" / "gepa_trainsets" / "gepa_trainset_twibot22_v4.json"
DST = PROJECT_ROOT / "data" / "gepa_trainsets" / "gepa_trainset_twibot22_v4_latest.json"

LIMIT = int(os.environ.get("TWEET_FILES_LIMIT", "9"))
TWEET_FILES = [RAW_DIR / f"tweet_{i}.json" for i in range(LIMIT)]


def main():
    with open(SRC) as f:
        users = json.load(f)
    user_id_to_record_idx = {}
    for idx, u in enumerate(users):
        raw = u["id"]
        clean = raw[1:] if raw.startswith("u") else raw
        user_id_to_record_idx[str(int(clean))] = idx
    print(f"Loaded {len(users)} curated trainset users")
    print(f"Sample stripped ID: {next(iter(user_id_to_record_idx))!r}")

    per_user_tweets = {uid: [] for uid in user_id_to_record_idx}
    total_seen = 0
    total_kept = 0
    t_start = time.time()

    for fi, tfile in enumerate(TWEET_FILES):
        if not tfile.exists():
            print(f"[skip] {tfile} not found")
            continue
        size_gb = tfile.stat().st_size / 1e9
        print(f"\n[{fi+1}/{len(TWEET_FILES)}] {tfile.name}  ({size_gb:.1f} GB)")
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
        print(f"  done in {elapsed:.0f}s — saw {seen_in_file:,}, kept {kept_in_file:,} for our {len(users)} users")

    overall = time.time() - t_start
    print(f"\n=== TOTAL: saw {total_seen:,} tweets, kept {total_kept:,} in {overall:.0f}s ===")

    output = []
    nonzero = 0
    zero_users = []
    for aid_str, idx in user_id_to_record_idx.items():
        record = dict(users[idx])
        tweets = per_user_tweets[aid_str]
        if tweets:
            tweets.sort(key=lambda x: x[0], reverse=True)
            record["tweets"] = [t[1] for t in tweets]
            nonzero += 1
        else:
            record["tweets"] = []
            zero_users.append(record["id"])
        output.append(record)

    print(f"\n{nonzero}/{len(users)} users have tweets ({len(zero_users)} zero-tweet)")
    if zero_users[:5]:
        print(f"Sample zero-tweet IDs: {zero_users[:5]}")

    DST.parent.mkdir(parents=True, exist_ok=True)
    with open(DST, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nWrote {DST}")


if __name__ == "__main__":
    main()
