"""Embedding computation CLI for tweet sorting by centrality and cluster-based selection."""

import os
import json
import html
import signal
import argparse
import math
from typing import List, Dict, Tuple
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ..config import Config

# Global flag for graceful shutdown
STOP_REQUESTED = False


def handle_signal(signum, frame):
    """Signal handler for graceful shutdown."""
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"\nSignal {signum} received. Will stop after current user.", flush=True)


def load_model(model_name: str = "jinaai/jina-embeddings-v3") -> SentenceTransformer:
    """
    Load the sentence transformer model.

    Args:
        model_name: HuggingFace model name

    Returns:
        Loaded SentenceTransformer model
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model {model_name} on {device}...")
    model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
    model.eval()
    return model


def preprocess_tweet(text: str) -> str:
    """
    Preprocess tweet text for embedding.

    - HTML-unescape
    - Replace user mentions with '@user'
    - Replace links with 'http'

    Args:
        text: Raw tweet text

    Returns:
        Preprocessed tweet text
    """
    text = html.unescape(text)
    new_text = []
    for t in text.split():
        if t.startswith('@') and len(t) > 1:
            t = '@user'
        elif t.startswith('http'):
            t = 'http'
        new_text.append(t)
    return " ".join(new_text)


def get_embeddings_batch(
    model: SentenceTransformer,
    text_list: List[str],
    batch_size: int = 64
) -> np.ndarray:
    """
    Compute L2-normalized embeddings for a list of texts.

    Args:
        model: SentenceTransformer model
        text_list: List of texts to embed
        batch_size: Batch size for encoding

    Returns:
        Numpy array of shape [n_texts, dim] with L2-normalized embeddings
    """
    processed = [preprocess_tweet(t) for t in text_list]
    embeddings = model.encode(
        processed,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return embeddings


def sort_tweets_by_centrality(tweets: List[str], embeddings: np.ndarray) -> Tuple[List[str], np.ndarray]:
    """
    Sort tweets by cosine similarity to the centroid.

    Args:
        tweets: List of tweet texts
        embeddings: L2-normalized embeddings for the tweets

    Returns:
        Tuple of (tweets sorted by centrality, sorted indices)
    """
    centroid = embeddings.mean(axis=0)
    norm = np.linalg.norm(centroid) + 1e-12
    centroid /= norm

    # With L2-normalized embeddings, cosine similarity = dot product
    similarities = embeddings @ centroid
    order = np.argsort(-similarities)  # Highest similarity first
    return [tweets[i] for i in order], order


def find_optimal_k(
    embeddings: np.ndarray,
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 42
) -> int:
    """
    Find optimal number of clusters using silhouette score.

    The silhouette score measures how similar samples are to their own cluster
    compared to other clusters. Higher scores indicate better-defined clusters.

    Args:
        embeddings: L2-normalized embeddings (shape: [n, dim])
        k_min: Minimum number of clusters to try
        k_max: Maximum number of clusters to try
        random_state: Random seed for reproducibility

    Returns:
        Optimal number of clusters
    """
    n = len(embeddings)

    # Edge cases
    if n < 3:
        return 1
    k_max = min(k_max, n - 1)  # Can't have more clusters than samples - 1
    if k_max < k_min:
        return max(1, k_max)

    best_k = k_min
    best_score = -1

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # Silhouette score requires at least 2 clusters and 2 samples per cluster
        if len(set(labels)) < 2:
            continue

        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_k = k
            best_score = score

    return best_k


def select_tweets_by_cluster_proportionality(
    tweets: List[str],
    embeddings: np.ndarray,
    k: int,
    k_max: int = 10,
    random_state: int = 42
) -> Tuple[List[str], List[int]]:
    """
    Select k tweets using cluster-based proportional selection.

    This method applies K-Means clustering to the embeddings, allocates a quota
    to each cluster proportional to its size, and selects tweets closest to
    each cluster's centroid.

    The number of clusters K is determined dynamically:
        K = min(k_max, floor(sqrt(n)))
    where n is the number of tweets.

    Args:
        tweets: List of tweet texts
        embeddings: L2-normalized embeddings for the tweets (shape: [n, dim])
        k: Total number of tweets to select (context budget)
        k_max: Maximum number of clusters to prevent over-fragmentation
        random_state: Random seed for K-Means reproducibility

    Returns:
        Tuple of (selected tweets in cluster-proportional order, selected indices)
    """
    n = len(tweets)

    # Edge cases
    if n == 0:
        return [], []
    if k >= n:
        # If budget exceeds available tweets, return all sorted by global centrality
        sorted_tweets, order = sort_tweets_by_centrality(tweets, embeddings)
        return sorted_tweets, order.tolist()
    if k <= 0:
        return [], []

    # Determine number of clusters: K = min(k_max, floor(sqrt(n)))
    K = min(k_max, int(math.floor(math.sqrt(n))))
    K = max(1, K)  # Ensure at least 1 cluster

    # Edge case: if K == 1, just do centrality-based selection
    if K == 1:
        sorted_tweets, order = sort_tweets_by_centrality(tweets, embeddings)
        return sorted_tweets[:k], order[:k].tolist()

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=K, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    cluster_centers = kmeans.cluster_centers_

    # Compute cluster sizes
    cluster_sizes = np.bincount(cluster_labels, minlength=K)

    # Allocate quotas proportional to cluster size
    # k_j = floor(|C_j| / n * k)
    quotas = np.floor(cluster_sizes / n * k).astype(int)

    # Distribute remaining slots to largest clusters
    remaining = k - quotas.sum()
    if remaining > 0:
        # Sort clusters by size (descending) and distribute remaining slots
        sorted_cluster_indices = np.argsort(-cluster_sizes)
        for i in range(int(remaining)):
            cluster_idx = sorted_cluster_indices[i % K]
            quotas[cluster_idx] += 1

    # For each cluster, select quota_j tweets closest to cluster centroid
    selected_indices = []

    for cluster_idx in range(K):
        quota = quotas[cluster_idx]
        if quota == 0:
            continue

        # Get indices of tweets in this cluster
        cluster_member_indices = np.where(cluster_labels == cluster_idx)[0]

        if len(cluster_member_indices) == 0:
            continue

        # Get embeddings for this cluster
        cluster_embeddings = embeddings[cluster_member_indices]

        # Compute distance to cluster centroid
        centroid = cluster_centers[cluster_idx]
        centroid_norm = np.linalg.norm(centroid) + 1e-12
        centroid_normalized = centroid / centroid_norm

        # Cosine similarity (embeddings are already L2-normalized)
        similarities = cluster_embeddings @ centroid_normalized

        # Select top quota tweets closest to centroid
        top_k_in_cluster = min(quota, len(cluster_member_indices))
        top_local_indices = np.argsort(-similarities)[:top_k_in_cluster]

        # Map back to original indices
        selected_from_cluster = cluster_member_indices[top_local_indices]
        selected_indices.extend(selected_from_cluster.tolist())

    # Order selected tweets by their similarity to global centroid for coherence
    if len(selected_indices) > 0:
        selected_embeddings = embeddings[selected_indices]
        global_centroid = selected_embeddings.mean(axis=0)
        global_centroid_norm = np.linalg.norm(global_centroid) + 1e-12
        global_centroid /= global_centroid_norm

        similarities = selected_embeddings @ global_centroid
        reorder = np.argsort(-similarities)
        selected_indices = [selected_indices[i] for i in reorder]

    selected_tweets = [tweets[i] for i in selected_indices]
    return selected_tweets, selected_indices


def compute_both_orderings(
    tweets: List[str],
    embeddings: np.ndarray,
    k_max_clusters: int = 10,
    random_state: int = 42
) -> Dict[str, List[str]]:
    """
    Compute both semantic centrality and cluster-based orderings for tweets.

    This function is used during embedding computation to prepare tweets
    for both selection strategies simultaneously.

    Args:
        tweets: List of tweet texts
        embeddings: L2-normalized embeddings (shape: [n, dim])
        k_max_clusters: Maximum number of clusters for proportional selection
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with:
            - 'centrality': All tweets sorted by semantic centrality
            - 'cluster_order': Indices ordering for cluster-based selection
              (to be used with different k values at inference time)
            - 'cluster_info': Metadata about clustering for later use
    """
    n = len(tweets)

    if n == 0:
        return {
            'centrality': [],
            'cluster_order': [],
            'cluster_info': {'n_clusters': 0, 'cluster_labels': [], 'cluster_sizes': []}
        }

    # 1. Compute centrality ordering (full ordering)
    centrality_sorted, centrality_order = sort_tweets_by_centrality(tweets, embeddings)

    # Build inverse mapping: original_position -> sorted_position
    # centrality_order[sorted_pos] = orig_pos, so we invert it
    orig_to_sorted = {int(orig_pos): sorted_pos for sorted_pos, orig_pos in enumerate(centrality_order)}

    # 2. Compute clustering for proportional selection
    # Use silhouette score to find optimal K (data-driven)
    K = find_optimal_k(embeddings, k_min=2, k_max=k_max_clusters, random_state=random_state)

    if K == 1 or n < 3:
        # With single cluster, indices 0..n-1 in sorted order
        return {
            'centrality': centrality_sorted,
            'cluster_order': list(range(n)),
            'cluster_info': {
                'n_clusters': 1,
                'cluster_labels': [0] * n,
                'cluster_sizes': [n]
            }
        }

    # Apply K-Means with optimal K
    kmeans = KMeans(n_clusters=K, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    cluster_centers = kmeans.cluster_centers_
    cluster_sizes = np.bincount(cluster_labels, minlength=K).tolist()

    # Compute silhouette score for the final clustering
    final_silhouette = silhouette_score(embeddings, cluster_labels) if K > 1 else 0.0

    # Pre-compute ordering within each cluster (by distance to cluster centroid)
    # This allows efficient selection at inference time with any k
    # IMPORTANT: Convert indices from original positions to sorted positions
    cluster_ranked_indices = []

    for cluster_idx in range(K):
        cluster_member_indices = np.where(cluster_labels == cluster_idx)[0]

        if len(cluster_member_indices) == 0:
            continue

        cluster_embeddings = embeddings[cluster_member_indices]
        centroid = cluster_centers[cluster_idx]
        centroid_norm = np.linalg.norm(centroid) + 1e-12
        centroid_normalized = centroid / centroid_norm

        similarities = cluster_embeddings @ centroid_normalized
        order_in_cluster = np.argsort(-similarities)
        orig_ranked_indices = cluster_member_indices[order_in_cluster].tolist()

        # Convert original indices to sorted indices so they work with centrality-sorted tweets
        sorted_ranked_indices = [orig_to_sorted[orig_idx] for orig_idx in orig_ranked_indices]

        cluster_ranked_indices.append({
            'cluster_id': cluster_idx,
            'size': len(cluster_member_indices),
            'ranked_indices': sorted_ranked_indices
        })

    # Sort clusters by size for consistent ordering
    cluster_ranked_indices.sort(key=lambda x: -x['size'])

    return {
        'centrality': centrality_sorted,
        'cluster_order': cluster_ranked_indices,
        'cluster_info': {
            'n_clusters': K,
            'silhouette_score': float(final_silhouette),
            'cluster_labels': cluster_labels.tolist(),
            'cluster_sizes': cluster_sizes
        }
    }


def select_from_precomputed_clusters(
    tweets: List[str],
    cluster_order: List[Dict],
    k: int
) -> List[str]:
    """
    Select k tweets using precomputed cluster ordering.

    This is used at inference time to efficiently select tweets without
    re-running K-Means clustering.

    Args:
        tweets: Original list of tweets
        cluster_order: Precomputed cluster rankings from compute_both_orderings
        k: Number of tweets to select

    Returns:
        List of k selected tweets in cluster-proportional order
    """
    if not cluster_order or k <= 0:
        return []

    n = sum(c['size'] for c in cluster_order)
    if k >= n:
        # Return all tweets, ordered by cluster then by centrality within cluster
        all_indices = []
        for cluster in cluster_order:
            all_indices.extend(cluster['ranked_indices'])
        return [tweets[i] for i in all_indices]

    # Compute quotas
    quotas = []
    for cluster in cluster_order:
        cluster_size = cluster['size']
        quota = int(math.floor(cluster_size / n * k))
        quotas.append(quota)

    # Distribute remaining slots
    remaining = k - sum(quotas)
    for i in range(int(remaining)):
        quotas[i % len(quotas)] += 1

    # Select from each cluster
    selected_indices = []
    for cluster, quota in zip(cluster_order, quotas):
        selected_indices.extend(cluster['ranked_indices'][:quota])

    return [tweets[i] for i in selected_indices]


def checkpoint_paths(out_path: str) -> tuple:
    """Get paths for checkpoint files."""
    base = out_path
    ckpt = base + ".jsonl"
    err = base + ".errors.jsonl"
    meta = base + ".meta.json"
    return ckpt, err, meta


def count_valid_lines_and_repair(ckpt_path: str) -> int:
    """
    Count valid JSONL lines and repair trailing corrupted lines.

    Args:
        ckpt_path: Path to the checkpoint file

    Returns:
        Number of valid lines remaining
    """
    if not os.path.exists(ckpt_path):
        return 0

    valid_count = 0
    tmp_path = ckpt_path + ".tmp"
    repaired = False

    with open(ckpt_path, "r", encoding="utf-8") as fin, \
         open(tmp_path, "w", encoding="utf-8") as fout:
        for line in fin:
            s = line.strip()
            if not s:
                continue
            try:
                json.loads(s)
                fout.write(line)
                valid_count += 1
            except json.JSONDecodeError:
                repaired = True
                break

    if repaired:
        os.replace(tmp_path, ckpt_path)
        print(f"[checkpoint] Repaired trailing partial line; now at {valid_count} users.", flush=True)
    else:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass

    return valid_count


def append_jsonl(path: str, obj: Dict) -> None:
    """Append a single JSON object as one line to a JSONL file."""
    line = json.dumps(obj, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def finalize_from_checkpoint(ckpt_path: str, expected_count: int, out_path: str) -> None:
    """
    Finalize output from checkpoint file.

    Args:
        ckpt_path: Path to checkpoint JSONL file
        expected_count: Expected number of users
        out_path: Path for final output JSON
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    slots = [None] * expected_count
    seen = 0

    with open(ckpt_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except json.JSONDecodeError:
                raise RuntimeError(f"Corrupted checkpoint at line {ln}. Run again to repair.")

            i = rec.get("i")
            user_obj = rec.get("user")
            if i is None or user_obj is None:
                raise RuntimeError(f"Malformed record at line {ln}: missing i/user.")

            if 0 <= i < expected_count and slots[i] is None:
                slots[i] = user_obj
                seen += 1

    missing = [i for i, v in enumerate(slots) if v is None]
    if missing:
        raise RuntimeError(f"Cannot finalize: missing {len(missing)} users (e.g., indices {missing[:5]}...).")

    # Write final JSON
    tmp_out = out_path + ".tmp"
    with open(tmp_out, "w", encoding="utf-8") as f:
        json.dump(slots, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_out, out_path)
    print(f"[finalize] Wrote final dataset to {out_path} ({seen} users).", flush=True)


def run_embedding(
    input_path: str,
    output_path: str,
    model_name: str = "jinaai/jina-embeddings-v3",
    batch_size: int = 64,
    finalize_only: bool = False,
    k_max_clusters: int = 10
) -> None:
    """
    Run embedding computation and sort tweets by both centrality and cluster-based methods.

    Args:
        input_path: Path to input JSON file
        output_path: Path for output JSON file
        model_name: SentenceTransformer model name
        batch_size: Batch size for embedding
        finalize_only: If True, only finalize from existing checkpoint
        k_max_clusters: Maximum number of clusters for proportional selection
    """
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_signal)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, handle_signal)

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    ckpt_path, err_path, meta_path = checkpoint_paths(output_path)

    # Load input data
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    total = len(data)

    if finalize_only:
        count_valid_lines_and_repair(ckpt_path)
        finalize_from_checkpoint(ckpt_path, expected_count=total, out_path=output_path)
        return

    # Resume from checkpoint
    start_idx = count_valid_lines_and_repair(ckpt_path)
    print(f"[checkpoint] Resuming at index {start_idx} of {total}.", flush=True)

    # Save metadata
    meta = {
        "input_path": input_path,
        "output_path": output_path,
        "checkpoint": ckpt_path,
        "errors": err_path,
        "total": total,
        "model_name": model_name,
        "k_max_clusters": k_max_clusters
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # Load model only if work remains
    model = None
    if start_idx < total:
        model = load_model(model_name)

    # Process users
    for i in tqdm(range(start_idx, total), desc="Processing Users", initial=start_idx, total=total):
        if STOP_REQUESTED:
            print(f"\nStop requested. Safely paused at index {i}.", flush=True)
            return

        user = data[i]
        user_copy = dict(user)

        try:
            # Autodetect tweet key
            tweet_key = None
            if isinstance(user_copy.get("tweet"), list):
                tweet_key = "tweet"
            elif isinstance(user_copy.get("tweets"), list):
                tweet_key = "tweets"

            tweets = user_copy.get(tweet_key) if tweet_key else None
            if tweets and len(tweets) > 0:
                embeddings = get_embeddings_batch(model, tweets, batch_size=batch_size)

                # Compute both orderings simultaneously
                orderings = compute_both_orderings(
                    tweets=tweets,
                    embeddings=embeddings,
                    k_max_clusters=k_max_clusters,
                    random_state=42
                )

                # Store centrality-sorted tweets in original key (backwards compatible)
                user_copy[tweet_key] = orderings['centrality']

                # Store cluster ordering info for proportional selection at inference time
                user_copy['cluster_order'] = orderings['cluster_order']
                user_copy['cluster_info'] = orderings['cluster_info']

            rec = {"i": i, "status": "ok", "user": user_copy}

        except Exception as e:
            import traceback
            tb = traceback.format_exc(limit=3)
            err_rec = {"i": i, "error": str(e), "traceback": tb}
            append_jsonl(err_path, err_rec)
            rec = {"i": i, "status": "error", "user": user}

        append_jsonl(ckpt_path, rec)

    # Finalize
    finalize_from_checkpoint(ckpt_path, expected_count=total, out_path=output_path)
    print("Done!", flush=True)


def main():
    """Main entry point for embedding CLI."""
    parser = argparse.ArgumentParser(
        description="Sort tweets by semantic centrality and cluster-based proportionality using Jina v3 embeddings"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file (optional)"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input JSON file path"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="jinaai/jina-embeddings-v3",
        help="SentenceTransformer model name"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size"
    )
    parser.add_argument(
        "--k-max-clusters",
        type=int,
        default=10,
        help="Maximum number of clusters for proportional selection (default: 10)"
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Only finalize from existing checkpoint"
    )

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        config = Config(args.config)
        model_name = args.model if args.model != "jinaai/jina-embeddings-v3" else config.embedding_model
        batch_size = args.batch_size if args.batch_size != 64 else config.embedding_batch_size
    else:
        model_name = args.model
        batch_size = args.batch_size

    run_embedding(
        input_path=args.input,
        output_path=args.output,
        model_name=model_name,
        batch_size=batch_size,
        finalize_only=args.finalize_only,
        k_max_clusters=args.k_max_clusters
    )


if __name__ == "__main__":
    main()
