"""Labeled test cases for the classification logic (QA coverage set)."""
from __future__ import annotations

from typing import NamedTuple


class TestCase(NamedTuple):
    """Labeled classification case."""

    __test__ = False  # keep pytest from trying to collect this class

    text: str
    expected: str  # positive_feedback | negative_feedback | query


POSITIVE = "positive_feedback"
NEGATIVE = "negative_feedback"
QUERY = "query"

# 30 realistic banking messages spanning the three classes, including
# mixed-signal edge cases (sentiment + ticket lookup in one message).
TEST_CASES: list[TestCase] = [
    # --- positive feedback (10) ---
    TestCase("I love the new mobile app, it's so fast!", POSITIVE),
    TestCase("Great service today, the agent was very helpful. Thank you!", POSITIVE),
    TestCase("Awesome, my issue is fixed. You guys rock!", POSITIVE),
    TestCase("Really happy with how quickly my transfer went through.", POSITIVE),
    TestCase("Thanks for the quick resolution, much appreciated.", POSITIVE),
    TestCase("Excellent work, the new dashboard is amazing.", POSITIVE),
    TestCase("I'm delighted with the support I received.", POSITIVE),
    TestCase("Good job fixing the billing problem, thanks a lot!", POSITIVE),
    TestCase("The new ATM feature is great!", POSITIVE),
    TestCase("I appreciate the help, you're the best!", POSITIVE),
    # --- negative feedback (10) ---
    TestCase("I'm so frustrated with the hidden fees on my account!", NEGATIVE),
    TestCase("Terrible experience. My card was declined three times.", NEGATIVE),
    TestCase("I'm angry about a duplicate charge I can't explain.", NEGATIVE),
    TestCase("Very disappointed with the slow customer service.", NEGATIVE),
    TestCase("This is unacceptable, my money has been missing for days.", NEGATIVE),
    TestCase("I'm upset that the app keeps crashing.", NEGATIVE),
    TestCase("Horrible support, I got no help at all.", NEGATIVE),
    TestCase("I have a problem with an unauthorized transaction, fix it.", NEGATIVE),
    TestCase("Ugh, the branch was a mess today.", NEGATIVE),
    TestCase("My refund request is taking forever, this is poor service.", NEGATIVE),
    # --- queries (10) ---
    TestCase("What is the status of my ticket 482913?", QUERY),
    TestCase("Can you check ticket 111222 for me?", QUERY),
    TestCase("Where is my ticket 998877 in the queue?", QUERY),
    TestCase("Any update on ticket 555666?", QUERY),
    TestCase("How long until my ticket 123456 is resolved?", QUERY),
    TestCase("Please give me an update on my case number 234567.", QUERY),
    TestCase("I want to track my ticket 345678.", QUERY),
    TestCase("What's happening with ticket 456789?", QUERY),
    TestCase("I'm a bit annoyed but thanks for the update on ticket 678901.", QUERY),
    TestCase("Not happy with the fee, can you check my ticket 789012?", QUERY),
]

# Ticket ids referenced by the query cases — the evaluator seeds a temporary
# database with these so status lookups always succeed during evaluation.
QUERY_TICKET_IDS = [
    "482913", "111222", "998877", "555666", "123456",
    "234567", "345678", "456789", "678901", "789012",
]
