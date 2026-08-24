"""DSPy signatures for bot detection."""

import dspy


# Minimal baseline instruction for ablation experiments
# States the task without biasing the model with specific heuristics
BASELINE_INSTRUCTION = """Classify this Twitter account as 'bot' or 'human' based on the profile and tweets provided."""


class BotDetectionSignatureWithInstruction(dspy.Signature):
    """Classify this Twitter account as 'bot' or 'human' based on the profile and tweets provided."""

    name = dspy.InputField(
        desc="The name of the user, as they've defined it on their profile. "
             "Not necessarily a person's name."
    )
    username = dspy.InputField(
        desc="The Twitter screen name, handle, or alias that this user "
             "identifies themselves with."
    )
    description = dspy.InputField(
        desc="The text of this user's profile description (also known as bio), "
             "if the user provided one."
    )
    followers: int = dspy.InputField(
        desc="The number of followers of the user."
    )
    following: int = dspy.InputField(
        desc="The number of users this user is following."
    )
    tweet_count: int = dspy.InputField(
        desc="The total number of tweets this user has made (lifetime, from the user's profile)."
    )
    account_age: str = dspy.InputField(
        desc="Account age in the format 'X years, Y days' for accounts older "
             "than 1 year, or 'Y days' for newer accounts."
    )
    protected: bool = dspy.InputField(
        desc="Indicates if this user has chosen to protect their Tweets "
             "(in other words, if this user's Tweets are private)."
    )
    verified: bool = dspy.InputField(
        desc="Indicates whether or not this Twitter user has a verified account."
    )
    tweets = dspy.InputField(
        desc="A selected sample of the user's tweets, each separated by '---'."
    )
    retweet_pct: float = dspy.InputField(
        desc="Percentage of the user's tweets that are retweets (start with 'RT @')."
    )
    url_pct: float = dspy.InputField(
        desc="Percentage of the user's tweets containing a URL."
    )
    mention_pct: float = dspy.InputField(
        desc="Percentage of the user's tweets that begin with '@' (replies or direct mentions)."
    )
    follower_following_ratio: float = dspy.InputField(
        desc="followers / max(1, following), rounded to 2 decimals."
    )

    label = dspy.OutputField(desc="'bot' or 'human'")


class BotDetectionSignatureGPTOSS(dspy.Signature):
    """Classify this Twitter account as 'bot' or 'human' based on the profile and tweets provided.

Reasoning: medium"""

    name = dspy.InputField(
        desc="The name of the user, as they've defined it on their profile. "
             "Not necessarily a person's name."
    )
    username = dspy.InputField(
        desc="The Twitter screen name, handle, or alias that this user "
             "identifies themselves with."
    )
    description = dspy.InputField(
        desc="The text of this user's profile description (also known as bio), "
             "if the user provided one."
    )
    followers: int = dspy.InputField(
        desc="The number of followers of the user."
    )
    following: int = dspy.InputField(
        desc="The number of users this user is following."
    )
    tweet_count: int = dspy.InputField(
        desc="The total number of tweets this user has made."
    )
    account_age: str = dspy.InputField(
        desc="Account age in the format 'X years, Y days' for accounts older "
             "than 1 year, or 'Y days' for newer accounts."
    )
    protected: bool = dspy.InputField(
        desc="Indicates if this user has chosen to protect their Tweets "
             "(in other words, if this user's Tweets are private)."
    )
    verified: bool = dspy.InputField(
        desc="Indicates whether or not this Twitter user has a verified account. "
             "Highly correlated with the human label if True."
    )
    tweets = dspy.InputField(
        desc="A selected sample of the user's tweets, each separated by '---'."
    )

    label = dspy.OutputField(desc="'bot' or 'human'")
