"""Context construction: post selection and DSPy example creation.

Builds the model input for one user: selects k posts under a given
post-selection mode (Latest / Semantic Centrality / Cluster-Proportional,
Sec. 3.3 of the paper), computes the derived posting-behavior ratios
(Sec. 3.2), and packs everything into a dspy.Example.
"""

from typing import List, Dict, Any, Optional, Literal

import dspy

from .dspy_components.helpers import (
    get_tweets_string,
    get_tweets_by_cluster_selection,
)


# ==============================================================================
# Label Normalization
# ==============================================================================

def normalize_label(raw_label: Any) -> Optional[str]:
    """
    Normalize model output to valid label.

    Handles various model output formats:
    - Direct labels: "bot", "human"
    - Case variations: "Bot", "HUMAN"
    - Verbose outputs: "This is a bot account"

    Returns:
        'bot', 'human', or None if can't be normalized
    """
    if raw_label is None:
        return None

    label = str(raw_label).lower().strip()

    # Direct match
    if label in ['bot', 'human']:
        return label

    # Substring matching for verbose outputs
    # Check 'bot' first since it's more distinctive
    if 'bot' in label and 'human' not in label:
        return 'bot'
    elif 'human' in label and 'bot' not in label:
        return 'human'

    return None


def parse_bool_field(value: Any) -> bool:
    """
    Parse boolean field that may be bool, string, or None.

    Handles:
    - True/False (actual booleans)
    - 'True'/'False' (strings, possibly with whitespace)
    - 'true'/'false' (lowercase strings)
    - None -> False
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == 'true'
    return bool(value)


# ==============================================================================
# Tweet Selection Modes
# ==============================================================================

# 'centrality_top' takes the first k tweets of the stored ordering: the k most
# central tweets for centrality-sorted files, or the k newest for newest-first
# files (the Latest strategy). 'cluster' applies cluster-proportional quotas.
SelectionMode = Literal['centrality_top', 'cluster']


def select_tweets_centrality_top(tweets: List[str], max_tweets: int) -> List[str]:
    """
    Select tweets with most central (representative) tweets first.
    Tweets are already sorted by centrality, so just take from the start.
    """
    return tweets[:max_tweets]


def select_tweets_for_mode(
    user_data: Dict[str, Any],
    max_tweets: int,
    mode: SelectionMode
) -> List[str]:
    """
    Select tweets using the specified mode.

    Args:
        user_data: User data dictionary with 'tweet' or 'tweets' field
        max_tweets: Maximum number of tweets to include
        mode: Selection mode

    Returns:
        List of selected tweet strings
    """
    # Handle both 'tweet' and 'tweets' keys
    tweets = user_data.get('tweet') or user_data.get('tweets', [])
    if not tweets:
        return []

    if mode == 'centrality_top':
        return select_tweets_centrality_top(tweets, max_tweets)
    elif mode == 'cluster':
        cluster_order = user_data.get('cluster_order')
        if cluster_order:
            return get_tweets_by_cluster_selection(tweets, cluster_order, max_tweets)
        # Fallback to centrality if no cluster data
        return select_tweets_centrality_top(tweets, max_tweets)
    else:
        raise ValueError(f"Unknown selection mode: {mode}")


def _compute_input_features(
    all_tweets: List[str], followers: int, following: int
) -> Dict[str, Any]:
    """Compute deterministic per-input features over a user's full visible tweet history.

    These are surfaced as input fields to free the LLM from doing manual
    counting from raw tweet text. They are computed over *all* tweets the
    dataset has for this user (typically dozens to a few hundred), and are
    assumed to generalize to the user's overall posting behavior.
    """
    posts = [p.strip() for p in (all_tweets or []) if p and p.strip()]
    n = len(posts)
    ratio = round(followers / max(1, following), 2)
    if n == 0:
        return {
            "retweet_pct": 0.0,
            "url_pct": 0.0,
            "mention_pct": 0.0,
            "follower_following_ratio": ratio,
        }
    rt_count = sum(1 for p in posts if p.startswith("RT @"))
    url_count = sum(1 for p in posts if "http" in p)
    mention_count = sum(1 for p in posts if p.startswith("@"))
    return {
        "retweet_pct": round(rt_count / n * 100, 1),
        "url_pct": round(url_count / n * 100, 1),
        "mention_pct": round(mention_count / n * 100, 1),
        "follower_following_ratio": ratio,
    }


def create_example_for_ablation(
    user_data: Dict[str, Any],
    max_tweets: int,
    selection_mode: SelectionMode,
    reference_date: Optional[Any] = None,
    date_format: Optional[str] = None
) -> dspy.Example:
    """
    Create a DSPy Example with custom tweet selection mode.
    """
    from .utils.dates import calculate_account_age, format_account_age

    # Select tweets using specified mode
    selected_tweets = select_tweets_for_mode(user_data, max_tweets, selection_mode)
    tweets_str = get_tweets_string(selected_tweets, max_tweets=max_tweets)

    # Handle nested 'profile' structure
    profile = user_data.get('profile', user_data)

    # Get account age
    account_age = user_data.get('account_age')
    if account_age is None and reference_date and date_format:
        created_at = profile.get('created_at')
        if created_at:
            total_days = calculate_account_age(created_at, reference_date, date_format)
            account_age = format_account_age(total_days)

    # Get screen_name
    screen_name = profile.get('screen_name') or profile.get('username', '')

    # Get label - handle both string and int formats
    label = user_data.get('label')
    if isinstance(label, int) or (isinstance(label, str) and label.isdigit()):
        label = 'bot' if str(label) == '1' else 'human'

    followers_count = int(profile.get('followers_count', 0) or 0)
    following_count = int(profile.get('friends_count', 0) or profile.get('following_count', 0) or 0)
    description = str(profile.get('description', ''))
    # Compute features over the user's full visible tweet history, not just the K-post excerpt.
    all_tweets = user_data.get('tweets') or profile.get('tweets') or []
    feats = _compute_input_features(all_tweets, followers_count, following_count)

    example = dspy.Example(
        name=str(profile.get('name', '')),
        username=str(screen_name),
        description=description,
        followers=followers_count,
        following=following_count,
        tweet_count=int(profile.get('statuses_count', 0) or 0),
        account_age=str(account_age) if account_age else 'Unknown',
        protected=parse_bool_field(profile.get('protected')),
        verified=parse_bool_field(profile.get('verified')),
        tweets=tweets_str,
        # Derived ratios computed over the user's full visible tweet history
        retweet_pct=feats["retweet_pct"],
        url_pct=feats["url_pct"],
        mention_pct=feats["mention_pct"],
        follower_following_ratio=feats["follower_following_ratio"],
        label=label
    ).with_inputs(
        "name", "username", "description",
        "followers", "following", "tweet_count",
        "account_age", "protected", "verified", "tweets",
        "retweet_pct", "url_pct", "mention_pct",
        "follower_following_ratio",
    )

    return example
