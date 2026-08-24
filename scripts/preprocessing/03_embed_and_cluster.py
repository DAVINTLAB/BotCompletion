#!/usr/bin/env python3
"""Step 3 — Embed each user's tweets with Jina Embeddings V3, sort by
semantic centrality, and add cluster-proportional selection metadata
(Sec. 3.2–3.3 of the paper).

Thin wrapper over twibot.cli.embed_tweets. For each user record the output
file has `tweets` re-ordered most-central-first and gains `cluster_order` /
`cluster_info` fields used by the Cluster-Proportional selection strategy.

Usage (run for both splits):
    python 03_embed_and_cluster.py -i data/twibot22/train_botsay.json -o data/twibot22/train_clustered.json
    python 03_embed_and_cluster.py -i data/twibot22/test_botsay.json  -o data/twibot22/test_clustered.json

A GPU is strongly recommended; the script checkpoints and can resume
(--finalize-only finalizes from an existing checkpoint).
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from twibot.cli.embed_tweets import main

if __name__ == "__main__":
    main()
