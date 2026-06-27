"""Concept specs — how to elicit a concept's behaviour from the model (the 'readout' seam).

A ReadoutSpec turns text into a scalar the model reports for the concept (pos-word logit minus
neg-word logit, under a few-shot prompt). Swap this and FeatureScope screens a different concept.
The `probes` are concept-neutral sentences that give steering headroom to move either way.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ReadoutSpec:
    name: str
    few_shot: str           # few-shot examples, each ending in the label word
    template: str           # per-text template with a {text} slot, ending right before the label
    pos_word: str           # label word for the positive pole (leading space matters for tokenization)
    neg_word: str
    probes: list = field(default_factory=list)


SENTIMENT = ReadoutSpec(
    name="sentiment",
    few_shot=("Review: A wonderful, heartwarming film.\nSentiment: positive\n"
              "Review: A boring, pointless waste of time.\nSentiment: negative\n"),
    template="Review: {text}\nSentiment:",
    pos_word=" positive", neg_word=" negative",
    probes=["The movie was decent.", "An average film overall.", "It was okay, I guess.",
            "A film I watched last night.", "The product arrived on time.", "Here is my honest review.",
            "The book was a normal length.", "I went to see it yesterday.", "The service was as expected.",
            "This is a review of the item.", "The show ran for two hours.", "I finished it on the weekend."],
)

FORMALITY = ReadoutSpec(
    name="formality",
    few_shot=("Text: I would be most grateful if you could attend the meeting.\nRegister: formal\n"
              "Text: hey wanna grab food later lol\nRegister: casual\n"),
    template="Text: {text}\nRegister:",
    pos_word=" formal", neg_word=" casual",
    probes=["I sent the message yesterday.", "Here is the note.", "This is the text in question.",
            "They replied to me today.", "The email was fairly short.", "A quick update for you.",
            "We talked about the plan.", "The document is attached.", "I will let you know soon.",
            "That is all for now.", "Please see the details.", "It happened last week."],
)

REGISTRY = {"sentiment": SENTIMENT, "formality": FORMALITY}
