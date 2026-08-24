"""Helper functions for tweet formatting and cluster-proportional selection."""

import math
from typing import List


def get_tweets_string(tweets_list: List[str], max_tweets: int = 10) -> str:
    """
    Convert a list of tweets to a formatted string.

    Args:
        tweets_list: List of tweet text strings
        max_tweets: Maximum number of tweets to include

    Returns:
        Formatted string with tweets separated by '---' delimiter
    """
    if not tweets_list:
        return "This user has not made any tweets."

    tweets = tweets_list[:max_tweets]
    initial = '---\n\n'
    joined = "\n---\n\n".join(tweets) + "\n---"
    return initial + joined


def get_tweets_by_cluster_selection(
    tweets_list: List[str],
    cluster_order: List,
    max_tweets: int = 10
) -> List[str]:
    """
    Select tweets using cluster-based proportional selection.

    This function uses precomputed cluster orderings to select a representative
    sample of tweets from each cluster proportional to its size.

    Args:
        tweets_list: List of tweets in CENTRALITY-SORTED order (as stored in user['tweets'])
        cluster_order: Precomputed cluster rankings from embed_tweets.compute_both_orderings
                      Either:
                      - List of dicts: {'cluster_id': int, 'size': int, 'ranked_indices': List[int]}
                      - List of ints: indices (when K=1, single cluster = centrality order)
        max_tweets: Number of tweets to select (context budget k)

    Returns:
        List of selected tweets in cluster-proportional order
    """
    if not cluster_order or max_tweets <= 0 or not tweets_list:
        return []

    # Handle K=1 case where cluster_order is just a list of indices
    if isinstance(cluster_order[0], int):
        # Single cluster - just use the indices directly (centrality order)
        indices = cluster_order[:max_tweets]
        return [tweets_list[i] for i in indices if i < len(tweets_list)]

    n = sum(c['size'] for c in cluster_order)
    if max_tweets >= n:
        # Return all tweets, ordered by cluster priority
        all_indices = []
        for cluster in cluster_order:
            all_indices.extend(cluster['ranked_indices'])
        return [tweets_list[i] for i in all_indices if i < len(tweets_list)]

    # Compute quotas proportional to cluster size: k_j = floor(|C_j| / n * k)
    quotas = []
    for cluster in cluster_order:
        cluster_size = cluster['size']
        quota = int(math.floor(cluster_size / n * max_tweets))
        quotas.append(quota)

    # Distribute remaining slots to largest clusters
    remaining = max_tweets - sum(quotas)
    for i in range(int(remaining)):
        quotas[i % len(quotas)] += 1

    # Select from each cluster
    selected_indices = []
    for cluster, quota in zip(cluster_order, quotas):
        selected_indices.extend(cluster['ranked_indices'][:quota])

    return [tweets_list[i] for i in selected_indices if i < len(tweets_list)]
