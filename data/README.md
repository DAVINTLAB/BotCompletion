# Data

This repository ships **no dataset content**. TwiBot-22 contains Twitter/X
user profiles and tweets and is distributed by its authors on request; we do
not redistribute any of it. All experiment inputs are rebuilt locally from
your own copy of the dataset using the scripts in `scripts/preprocessing/`
and `scripts/curation/`.

## 1. Obtain the raw data

1. **TwiBot-22** — request access from the dataset authors via
   [github.com/LuoUndergradXJTU/TwiBot-22](https://github.com/LuoUndergradXJTU/TwiBot-22).
   You need: `label.csv`, `split.csv`, `user.json`, and `tweet_0.json` …
   `tweet_8.json` (the tweet files total roughly 100 GB).
2. **BotSay splits** — the paper evaluates on BotSay-340, the balanced
   340-user TwiBot-22 split introduced by BotSay (Feng et al., ACL 2024).
   Their split file `data/Twibot-22/split_new.json` is included in the
   `data.zip` linked from [github.com/BunsenFeng/botsay](https://github.com/BunsenFeng/botsay)
   (access is conditioned on holding TwiBot-22 access).

## 2. Build the experiment files

From the repository root:

```bash
# Step 1 — join raw TwiBot-22 into per-user JSONL records (~hours; streams 100 GB)
python scripts/preprocessing/01_build_user_jsonl.py \
    --raw-dir /path/to/twibot-22 --out-dir data/twibot22

# Step 2 — extract BotSay's 2,694-user train pool and 340-user eval split
python scripts/preprocessing/02_extract_botsay_splits.py \
    --split-json /path/to/botsay/data/Twibot-22/split_new.json

# Step 3 — Jina-V3 embeddings: centrality ordering + cluster metadata (GPU recommended)
python scripts/preprocessing/03_embed_and_cluster.py \
    -i data/twibot22/train_botsay.json -o data/twibot22/train_clustered.json
python scripts/preprocessing/03_embed_and_cluster.py \
    -i data/twibot22/test_botsay.json -o data/twibot22/test_clustered.json

# Step 4 — newest-first tweet ordering for the Latest strategy
python scripts/preprocessing/04_build_test_latest.py --raw-dir /path/to/twibot-22
```

Resulting layout consumed by the experiment scripts:

```
data/twibot22/
    train_clustered.json    # 2,694-user train pool, centrality-sorted + cluster metadata
    test_clustered.json     # BotSay-340 eval split, same schema
    test_latest.json        # BotSay-340 with tweets newest-first
data/gepa_trainsets/
    gepa_trainset_twibot22_v4.json         # built by scripts/curation/ (347 users)
    gepa_trainset_twibot22_v4_latest.json  # newest-first variant
```

## 3. Build the curated GEPA training set

See the main README ("Training-set curation"): run the baseline over the
train pool, then the `scripts/curation/` selection scripts, then
`assemble_v4.py`.
