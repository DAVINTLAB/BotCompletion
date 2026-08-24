import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

LABELS = {"bot", "human"}
_URL_RE = re.compile(r"(?:https?://|www\.|\bt\.co/)", re.I)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", re.I)
_DIGIT_RUN_RE = re.compile(r"\d{4,}")

PROMO_TERMS = (
    "sale", "discount", "coupon", "offer", "free", "giveaway", "gift card",
    "buy", "shop", "store", "product", "services", "marketing", "seo",
    "subscribe", "sign up", "webinar", "customer", "crypto", "bitcoin",
    "forex", "signal", "casino", "bet", "promo",
)
STRONG_PROMO_TERMS = (
    "discount", "coupon", "giveaway", "gift card", "buy", "shop", "subscribe",
    "sign up", "marketing", "seo", "crypto", "bitcoin", "forex", "casino", "bet", "promo",
)
ORG_TERMS = (
    "official", "company", "brand", "business", "startup", "university",
    "institute", "foundation", "agency", "network", "media", "security",
    "software", "team", "club", "organization", "nonprofit", "inc", "llc",
)
NEWS_TERMS = (
    "news", "breaking", "headline", "latest", "update", "weather", "traffic",
    "report", "journal", "newspaper", "radio", "podcast", "magazine", "daily",
)
PERSON_TERMS = (
    "i am", "i'm", "my", "views", "opinions", "dad", "mom", "father",
    "mother", "wife", "husband", "son", "daughter", "student", "teacher",
    "professor", "phd", "doctor", "writer", "author", "artist", "actor",
    "comedian", "journalist", "reporter", "engineer", "founder", "photographer",
)
BOT_TERMS = ("bot", "automated", "auto-post", "autopost", "rss", "feed bot")


def bot_detection_metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> ScoreWithFeedback:
    """GEPA metric for binary bot detection: binary reward plus concise reflective feedback."""
    _ = (trace, pred_trace)  # API-required, intentionally unused.
    component = pred_name or "classify"

    gold_label = _parse_label(_field(gold, "label"))
    pred_raw = _field(pred, "label", "")
    pred_label = _parse_label(pred_raw)
    account = _extract_account(gold)

    if gold_label not in LABELS:
        return ScoreWithFeedback(
            score=0.0,
            feedback=(
                f"Metric setup error for `{component}`: the gold label could not be parsed "
                f"as `bot` or `human` from {repr(_field(gold, 'label'))}."
            ),
        )

    if pred_label not in LABELS:
        support = _evidence_for(account, gold_label, limit=3)
        return ScoreWithFeedback(
            score=0.0,
            feedback=(
                f"Label parse error for `{component}`: the label field began with "
                f"{_safe_first_token(pred_raw)!r}, not exactly `bot` or `human`. "
                f"The correct label is `{gold_label}`. First output word in the label field must be "
                f"only `bot` or `human`; do not prefix it with explanations, JSON, or markers. "
                f"For this account, useful evidence toward `{gold_label}`: {support}."
            ),
        )

    correct = pred_label == gold_label
    return ScoreWithFeedback(
        score=1.0 if correct else 0.0,
        feedback=_make_feedback(component, pred_label, gold_label, account, correct),
    )


def _make_feedback(component: str, pred_label: str, gold_label: str, account: Dict[str, Any], correct: bool) -> str:
    archetype = _archetype(account)
    snapshot = _snapshot(account)
    support = _evidence_for(account, gold_label, limit=3)
    counter_label = "human" if gold_label == "bot" else "bot"
    counter = _evidence_for(account, counter_label, limit=2)
    lesson = _lesson(account, gold_label)

    if correct:
        return (
            f"You correctly classified `{component}` as `{pred_label}`. "
            f"Archetype: {archetype}. {snapshot}. "
            f"Evidence supporting `{gold_label}`: {support}. "
            f"Counter-evidence to keep calibrated: {counter}. "
            f"Instruction lesson: {lesson}"
        )

    return (
        f"You incorrectly classified `{component}` as `{pred_label}`; the correct label is `{gold_label}`. "
        f"Archetype: {archetype}. {snapshot}. "
        f"Evidence that should have pushed toward `{gold_label}`: {support}. "
        f"Likely misleading evidence for `{pred_label}`: {counter}. "
        f"Instruction lesson: {lesson}"
    )


def _parse_label(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    # Take the first word, allowing harmless leading quotes/backticks/brackets.
    m = re.match(r"^[\s`'\"\[\(]*([A-Za-z]+)\b", text)
    if not m:
        return None
    word = m.group(1).lower()
    return word if word in LABELS else None


def _safe_first_token(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text.split(None, 1)[0][:40] if text else ""


def _extract_account(gold) -> Dict[str, Any]:
    name = _str_field(gold, "name")
    username = _str_field(gold, "username", _str_field(gold, "screen_name"))
    description = _str_field(gold, "description")
    tweets = _tweets_list(_field(gold, "tweets", []))

    followers = _int_field(gold, "followers", _int_field(gold, "followers_count", 0))
    following = _int_field(gold, "following", _int_field(gold, "following_count", 0))
    tweet_count = _int_field(gold, "tweet_count", _int_field(gold, "statuses_count", len(tweets)))
    ff_ratio = _float_field(gold, "follower_following_ratio", None)
    if ff_ratio is None:
        ff_ratio = followers / max(1, following)

    retweet_pct = _float_field(gold, "retweet_pct", None)
    url_pct = _float_field(gold, "url_pct", None)
    mention_pct = _float_field(gold, "mention_pct", None)
    if retweet_pct is None:
        retweet_pct = _pct(tweets, lambda t: t.strip().lower().startswith("rt @"))
    if url_pct is None:
        url_pct = _pct(tweets, lambda t: bool(_URL_RE.search(t)))
    if mention_pct is None:
        mention_pct = _pct(tweets, lambda t: t.strip().startswith("@"))

    profile_blob = "\n".join([name, username, description]).lower()
    text_blob = "\n".join([name, username, description] + tweets).lower()
    sample_n = len(tweets)
    dup_pct = _duplicate_pct(tweets)
    avg_len = sum(len(t) for t in tweets) / sample_n if sample_n else 0.0
    hashtag_pct = _pct(tweets, lambda t: "#" in t)

    return {
        "name": name,
        "username": username,
        "description": description,
        "tweets": tweets,
        "sample_n": sample_n,
        "followers": followers,
        "following": following,
        "tweet_count": tweet_count,
        "ff_ratio": ff_ratio,
        "account_age": _str_field(gold, "account_age"),
        "protected": _bool_field(gold, "protected", False),
        "verified": _bool_field(gold, "verified", False),
        "bio_empty": not bool(description.strip()),
        "desc_len": len(description.strip()),
        "retweet_pct": retweet_pct,
        "url_pct": url_pct,
        "mention_pct": mention_pct,
        "hashtag_pct": hashtag_pct,
        "duplicate_pct": dup_pct,
        "avg_tweet_len": avg_len,
        "promo": (
            _contains_terms(profile_blob, PROMO_TERMS)
            or _count_terms(text_blob, STRONG_PROMO_TERMS) >= 2
            or bool(_EMAIL_RE.search(text_blob))
        ),
        "org": _contains_terms(profile_blob + "\n" + text_blob, ORG_TERMS),
        "news": _contains_terms(profile_blob + "\n" + text_blob, NEWS_TERMS),
        "personal": _contains_terms(profile_blob, PERSON_TERMS),
        "explicit_bot": _contains_terms(profile_blob, BOT_TERMS),
        "has_email": bool(_EMAIL_RE.search(text_blob)),
        "digit_handle": bool(_DIGIT_RUN_RE.search(username)),
    }


def _bot_cues(a: Dict[str, Any]) -> List[Tuple[int, str]]:
    cues: List[Tuple[int, str]] = []
    if a["explicit_bot"]:
        cues.append((6, "profile explicitly refers to bot/automation/RSS behavior"))
    if a["protected"] and not a["verified"] and a["sample_n"] == 0:
        cues.append((5, "protected, unverified account with no visible tweets to inspect"))
    if a["sample_n"] == 0 and a["bio_empty"]:
        cues.append((5, "no visible tweets and an empty bio, leaving only sparse metadata"))
    if a["tweet_count"] <= 2 and a["followers"] <= 5 and not a["verified"]:
        cues.append((4, "near-empty account history with very few followers"))
    if a["bio_empty"] and a["ff_ratio"] < 0.15 and a["following"] >= 20:
        cues.append((4, "empty bio plus follow-heavy metadata: following far more accounts than follow it"))
    if a["duplicate_pct"] >= 20:
        cues.append((4, f"many near-duplicate sampled tweets ({a['duplicate_pct']:.0f}% duplicate after URL/mention normalization)"))
    if a["url_pct"] >= 75 and (a["promo"] or a["org"] or a["news"]):
        cues.append((4, f"link-heavy feed ({a['url_pct']:.0f}% URL tweets) tied to promotional/org/news terms"))
    if a["promo"] and (a["url_pct"] >= 40 or a["has_email"]):
        cues.append((4, "commercial/promotional wording, contact info, or sales/service language"))
    if (a["org"] or a["news"]) and a["url_pct"] >= 60 and not a["verified"]:
        cues.append((3, f"unverified organization/news-style account with a high URL share ({a['url_pct']:.0f}%)"))
    if a["avg_tweet_len"] >= 180 and a["url_pct"] >= 50:
        cues.append((3, "long templated-looking posts with many links"))
    if a["retweet_pct"] >= 85 and not a["personal"]:
        cues.append((3, f"mostly retweets ({a['retweet_pct']:.0f}%), suggesting a low-originality feed"))
    if a["mention_pct"] >= 70 and a["promo"]:
        cues.append((3, f"many tweets start as mentions ({a['mention_pct']:.0f}%) in a promotional context"))
    if a["digit_handle"] and (a["bio_empty"] or a["followers"] <= 20):
        cues.append((2, "handle contains a long digit run together with sparse profile signals"))
    if a["followers"] == 0 and a["tweet_count"] <= 10:
        cues.append((2, "almost no social footprint and little account activity"))
    return sorted(cues, reverse=True, key=lambda x: x[0])


def _human_cues(a: Dict[str, Any]) -> List[Tuple[int, str]]:
    cues: List[Tuple[int, str]] = []
    if a["verified"]:
        cues.append((7, "verified=True, which is a very strong human prior for this task"))
    if a["followers"] >= 10000 and a["ff_ratio"] >= 2:
        cues.append((5, "large follower base with followers substantially exceeding following"))
    if a["tweet_count"] >= 10000 and a["followers"] >= 500:
        cues.append((4, "long-lived, high-activity account with a substantial audience"))
    if a["desc_len"] >= 35 and a["personal"]:
        cues.append((4, "rich bio with personal/professional identity markers"))
    if a["sample_n"] >= 10 and a["duplicate_pct"] < 10 and a["url_pct"] < 70:
        cues.append((3, "varied sampled posts rather than repeated templates or pure links"))
    if a["mention_pct"] >= 25 and not a["promo"]:
        cues.append((3, f"many direct replies/mentions ({a['mention_pct']:.0f}%), consistent with conversation"))
    if a["retweet_pct"] >= 20 and a["retweet_pct"] <= 75 and a["personal"]:
        cues.append((2, "mixed retweets plus personal/professional context, like a normal curator"))
    if (a["org"] or a["news"]) and a["followers"] >= 1000 and a["tweet_count"] >= 500:
        cues.append((2, "recognizable organization/news/public-topic profile with nontrivial audience and history"))
    if not a["bio_empty"] and a["followers"] >= 100 and a["tweet_count"] >= 100:
        cues.append((2, "non-empty profile, established history, and at least moderate followers"))
    if a["protected"] and a["verified"]:
        cues.append((2, "protected status is overridden by verification and established profile cues"))
    return sorted(cues, reverse=True, key=lambda x: x[0])


def _evidence_for(a: Dict[str, Any], label: str, limit: int = 3) -> str:
    cues = _bot_cues(a) if label == "bot" else _human_cues(a)
    if not cues:
        if label == "bot":
            return "no single hard bot cue; do not rely only on human-like prose, display name, or conversational tweets when the account is unverified and lacks decisive human priors"
        return "no single hard human cue; avoid a hard bot rule when profile/history/conversation evidence plausibly outweighs sparse, retweet-heavy, or link-like surface patterns"
    return "; ".join(msg for _, msg in cues[:limit])


def _archetype(a: Dict[str, Any]) -> str:
    if a["verified"]:
        return "verified public/high-status account"
    if a["explicit_bot"]:
        return "self-described automation or bot-like feed"
    if a["sample_n"] == 0:
        return "no-visible-tweets/sparse-profile account"
    if a["duplicate_pct"] >= 15 and a["promo"]:
        return "repetitive promotional/service account"
    if a["url_pct"] >= 75 and (a["promo"] or a["org"] or a["news"]):
        return "link-heavy promotional/news/org feed"
    if a["retweet_pct"] >= 80:
        return "retweet-heavy curator/feed"
    if a["mention_pct"] >= 60:
        return "reply/mention-heavy account"
    if a["personal"] and a["desc_len"] >= 35:
        return "personal or professional identity account"
    if a["org"] or a["news"]:
        return "organization/news/public-topic account"
    if a["bio_empty"] and a["tweet_count"] < 100:
        return "sparse low-history profile"
    return "mixed or ambiguous account"


def _lesson(a: Dict[str, Any], gold_label: str) -> str:
    if a["verified"]:
        return "treat verification as a near-hard human prior unless overwhelming automation evidence appears."
    if a["sample_n"] == 0 or (a["bio_empty"] and a["tweet_count"] <= 2):
        return "for sparse accounts, weigh empty bio, no visible posts, tiny history, and follow-heavy metadata before guessing human from the display name."
    if a["duplicate_pct"] >= 15 or a["url_pct"] >= 75 or a["promo"]:
        return "separate real organizations from automated promotion by checking repetition, URL saturation, sales/service language, and targeted mentions."
    if a["retweet_pct"] >= 70 or a["mention_pct"] >= 60:
        return "do not use retweet/reply rate alone; combine it with bio richness, promotional terms, repetition, and audience/history."
    if gold_label == "human":
        return "prefer human when rich identity, conversation, verification, or established audience outweigh isolated bot-like ratios."
    return "prefer bot when sparse metadata, template repetition, link saturation, or explicit automation outweigh human-like names or bios."


def _snapshot(a: Dict[str, Any]) -> str:
    bio = "empty bio" if a["bio_empty"] else f"bio_len={a['desc_len']}"
    age = f", age={a['account_age']}" if a["account_age"] else ""
    return (
        f"Snapshot: verified={a['verified']}, protected={a['protected']}, "
        f"followers/following={a['followers']}/{a['following']} (ratio={a['ff_ratio']:.2g}), "
        f"tweet_count={a['tweet_count']}{age}, {bio}, sample_posts={a['sample_n']}, "
        f"RT={a['retweet_pct']:.0f}%, URL={a['url_pct']:.0f}%, mention-start={a['mention_pct']:.0f}%, "
        f"dup≈{a['duplicate_pct']:.0f}%"
    )


def _tweets_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        parts = [p.strip() for p in re.split(r"\n?\s*---\s*\n?", text) if p.strip()]
        return parts if parts else [text]
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _duplicate_pct(tweets: Sequence[str]) -> float:
    if not tweets:
        return 0.0
    normed = []
    for tweet in tweets:
        s = tweet.lower()
        s = _URL_RE.sub(" URL ", s)
        s = re.sub(r"@\w+", " @USER ", s)
        s = re.sub(r"\s+", " ", s).strip()
        normed.append(s)
    return 100.0 * (1.0 - len(set(normed)) / max(1, len(normed)))


def _pct(tweets: Sequence[str], predicate) -> float:
    if not tweets:
        return 0.0
    return 100.0 * sum(1 for t in tweets if predicate(t)) / len(tweets)


def _contains_terms(text: str, terms: Sequence[str]) -> bool:
    return _count_terms(text, terms) > 0


def _count_terms(text: str, terms: Sequence[str]) -> int:
    count = 0
    for term in terms:
        escaped = re.escape(term).replace(r"\ ", r"\s+")
        count += len(re.findall(r"(?<![a-z0-9_])" + escaped + r"(?![a-z0-9_])", text))
    return count


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    try:
        return obj[name]
    except Exception:
        pass
    return getattr(obj, name, default)


def _str_field(obj: Any, name: str, default: str = "") -> str:
    value = _field(obj, name, default)
    return "" if value is None else str(value)


def _int_field(obj: Any, name: str, default: int = 0) -> int:
    value = _field(obj, name, default)
    try:
        return int(float(value))
    except Exception:
        return default


def _float_field(obj: Any, name: str, default: Optional[float] = 0.0) -> Optional[float]:
    value = _field(obj, name, default)
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _bool_field(obj: Any, name: str, default: bool = False) -> bool:
    value = _field(obj, name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)