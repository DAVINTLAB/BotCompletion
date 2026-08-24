"""
Heuristic account-type classifier (Sec. 3.5 of the paper).

Routes each user into one of ten account types from surface signals
(retweet/URL/mention shares, follower asymmetry, bio keywords, verification,
template detection). Used by the training-set curation scripts for
stratification.
"""

from __future__ import annotations

from collections import Counter


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_post_features(tweets_str: str) -> dict:
    """Parse a tweets string (posts joined by '---') and return feature dict."""
    if not tweets_str or not tweets_str.strip():
        return {
            "has_posts": False,
            "count": 0,
            "rt_pct": 0.0,
            "url_pct": 0.0,
            "hashtag_pct": 0.0,
            "mention_pct": 0.0,
            "is_template": False,
            "template_prefix": "",
        }

    posts = [p.strip() for p in tweets_str.split("---") if p.strip()]
    count = len(posts)

    if count == 0:
        return {
            "has_posts": False,
            "count": 0,
            "rt_pct": 0.0,
            "url_pct": 0.0,
            "hashtag_pct": 0.0,
            "mention_pct": 0.0,
            "is_template": False,
            "template_prefix": "",
        }

    rt_count = sum(1 for p in posts if p.startswith("RT @"))
    url_count = sum(1 for p in posts if "http" in p)
    hashtag_count = sum(1 for p in posts if "#" in p)
    mention_count = sum(1 for p in posts if p.startswith("@"))

    # Template detection: 3+ posts sharing the same 15-char prefix
    prefix_counts = Counter(p[:15] for p in posts if len(p) >= 15)
    is_template = False
    template_prefix = ""
    for prefix, freq in prefix_counts.most_common(1):
        if freq >= 3:
            is_template = True
            template_prefix = prefix
            break

    return {
        "has_posts": True,
        "count": count,
        "rt_pct": round(rt_count / count * 100, 1),
        "url_pct": round(url_count / count * 100, 1),
        "hashtag_pct": round(hashtag_count / count * 100, 1),
        "mention_pct": round(mention_count / count * 100, 1),
        "is_template": is_template,
        "template_prefix": template_prefix,
    }


# ---------------------------------------------------------------------------
# Account classification
# ---------------------------------------------------------------------------

_SUPPORT_KEYWORDS = {"support", "help", "customer", "care", "assist", "service"}
_INSTITUTIONAL_KEYWORDS = {
    "official", "news", "media", "university", "institute", "organization",
    "foundation", "company", "inc", "ltd", "corp", "journal",
}


def classify_account(
    verified: bool,
    followers: int,
    following: int,
    description: str,
    features: dict,
) -> str:
    """Return a human-readable account type string."""
    rt_pct = float(features.get("rt_pct", 0.0))
    url_pct = float(features.get("url_pct", 0.0))
    mention_pct = float(features.get("mention_pct", 0.0))
    is_template = features.get("is_template", False)
    has_posts = features.get("has_posts", False)
    count = features.get("count", 0)

    desc_lower = (description or "").lower()
    desc_words = set(desc_lower.replace(",", " ").replace(".", " ").split())

    # 1. Verified → hard rule
    if verified:
        return "verified account"

    # 2. Amplification bot
    if rt_pct > 70:
        return f"amplification bot ({rt_pct}% retweets)"

    # 3. Feed/aggregation bot
    if url_pct > 60:
        return f"feed/aggregation bot ({url_pct}% URLs)"

    # 4. Template bot
    if is_template:
        return "template bot"

    # 5. Support account
    if _SUPPORT_KEYWORDS & desc_words and mention_pct > 30:
        return f"support account ({mention_pct}% mentions)"

    # 6. Institutional/corporate: require strong signal — 2+ institutional keywords,
    # or any single high-signal keyword (not "news"/"media" alone, which can appear in
    # casual descriptions like "news addict").
    inst_matches = _INSTITUTIONAL_KEYWORDS & desc_words
    _WEAK_INSTITUTIONAL = {"news", "media"}
    strong_inst = inst_matches - _WEAK_INSTITUTIONAL
    if len(inst_matches) >= 2 or strong_inst:
        return "institutional/corporate account"

    # 7. Sparse/low-activity
    if not has_posts or (count < 3 and followers < 20):
        return "sparse/low-activity account"

    # 8. Follow-spam
    if followers < 10 and following > 50:
        return "follow-spam account"

    # 9. Heavy curator
    if rt_pct > 50:
        return f"heavy curator ({rt_pct}% retweets)"

    # 10. Default
    return "standard account"
