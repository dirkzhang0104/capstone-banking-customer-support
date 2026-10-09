"""LLM abstraction layer.

Uses the OpenAI API when OPENAI_API_KEY is configured; otherwise falls back
to a deterministic heuristic engine (lexicon-based classification plus
templated generation) so the entire multi-agent pipeline, evaluation suite
and UI work fully offline.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Optional

from .. import config

# ---------------------------------------------------------------------------
# Heuristic lexicons and templates
# ---------------------------------------------------------------------------

POSITIVE_LEXICON = {
    "great", "awesome", "excellent", "amazing", "love", "happy", "satisfied",
    "thank", "thanks", "good", "helpful", "fast", "quick", "friendly",
    "impressed", "wonderful", "perfect", "delighted", "appreciate", "superb",
    "nice", "pleased", "best", "rock", "fixed", "quickly", "smooth", "glad",
}

NEGATIVE_LEXICON = {
    "angry", "mad", "terrible", "awful", "horrible", "disappointed",
    "frustrated", "frustrating", "unacceptable", "worst", "poor", "slow",
    "annoying", "annoyed", "upset", "unhappy", "scam", "fraud", "broken",
    "useless", "rude", "hate", "irritated", "complaint", "problem", "issue",
    "error", "charge", "fee", "declined", "missing", "crash", "mess", "ugh",
    "unauthorized",
}

STRONG_QUERY_HINTS = {
    "status", "update", "track", "where", "when", "progress", "how", "what", "why",
}
WEAK_QUERY_HINTS = {"check", "information", "info", "detail", "details", "follow"}

TICKET_RE = re.compile(r"\b(\d{6})\b")

ALLOWED_LABELS = ("positive_feedback", "negative_feedback", "query")

THANKYOU_TEMPLATES = [
    "Thank you so much for the kind words about {topic}! It truly makes our day to hear that we could assist you. We're glad it worked out well, and we're always here if you need anything else.",
    "What a delight to read this — thanks for letting us know about {topic}! Feedback like yours is exactly why we keep striving to do better. Don't hesitate to reach out anytime.",
    "Thank you! Hearing that you had a good experience with {topic} is fantastic news. We're so happy we could help, and we appreciate you taking a moment to share this with us.",
]

EMPATHY_TEMPLATES = [
    "We're truly sorry to hear about {topic}. We completely understand how frustrating that must be, and we take it seriously. Ticket {ticket_id} has been generated, and our team will follow up shortly. Thank you for your patience while we work on a fix.",
    "We apologize for the trouble with {topic} — that's not the experience we want you to have. We understand your frustration. Ticket {ticket_id} has been generated, and our team will follow up shortly with next steps.",
    "Thank you for bringing {topic} to our attention. We're sorry this happened and we understand how upsetting it is. Ticket {ticket_id} has been generated, and our team will follow up shortly. We're on it.",
]


def _normalize(word: str) -> str:
    """Very light stemming so 'fees' matches 'fee', 'crashing' matches 'crash'."""
    for suffix in ("ing", "ies", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            if suffix == "ies":
                return word[: -len(suffix)] + "y"
            return word[: -len(suffix)]
    return word


@dataclass
class Classification:
    label: str
    confidence: float


class HeuristicEngine:
    """Deterministic offline stand-in for an LLM."""

    @staticmethod
    def _hits(words: set, lexicon: set) -> int:
        norm_lex = {_normalize(w) for w in lexicon}
        return sum(1 for w in words if _normalize(w) in norm_lex)

    def classify(self, text: str) -> Classification:
        low = text.lower()
        words = set(re.findall(r"[a-z']+", low))
        pos = self._hits(words, POSITIVE_LEXICON)
        neg = self._hits(words, NEGATIVE_LEXICON)
        has_ticket = bool(TICKET_RE.search(low))
        is_question = "?" in low
        strong = bool(STRONG_QUERY_HINTS & words)
        weak = bool(WEAK_QUERY_HINTS & words)

        # 1) An explicit ticket reference with a lookup intent wins.
        if has_ticket and (is_question or strong or weak):
            return Classification("query", 0.95)
        # 2) Question phrasing with no sentiment signal.
        if (is_question or strong) and pos + neg == 0:
            return Classification("query", 0.8)
        # 3) Sentiment-driven feedback.
        if pos > neg:
            return Classification("positive_feedback", min(0.95, 0.6 + 0.1 * pos))
        if neg > pos:
            return Classification("negative_feedback", min(0.95, 0.6 + 0.1 * neg))
        # 4) No signal at all — default to query (lowest-risk action).
        return Classification("query", 0.5)

    @staticmethod
    def _topic(message: str) -> str:
        words = re.findall(r"[a-zA-Z']+", message.lower())
        stop = {"the", "a", "an", "i", "my", "me", "am", "is", "are", "was",
                "it", "to", "of", "and", "or", "in", "on", "for", "with",
                "that", "this", "so", "very", "really", "today", "keep",
                "keeps", "got", "no", "all", "at", "be", "has", "have", "had",
                "you", "your", "we", "our", "they", "them", "he", "she", "up",
                "out", "about", "can", "could", "will", "would", "not", "but",
                "if", "when", "who", "what", "how", "why"}
        content = [w for w in words if w not in stop][:5]
        return ", ".join(content) if content else "your recent experience"

    def generate_thankyou(self, message: str) -> str:
        return random.choice(THANKYOU_TEMPLATES).format(topic=self._topic(message))

    def generate_empathy(self, message: str, ticket_id: str) -> str:
        return random.choice(EMPATHY_TEMPLATES).format(topic=self._topic(message), ticket_id=ticket_id)


# ---------------------------------------------------------------------------
# Live-LLM prompts
# ---------------------------------------------------------------------------

CLASSIFY_SYSTEM = (
    "You are the Classifier Agent for a banking customer support system. "
    "Classify the customer message into exactly one of: "
    "positive_feedback, negative_feedback, query. "
    'Respond with JSON only: {"label": "<label>", "confidence": <0..1>}'
)

THANKYOU_SYSTEM = (
    "You are the Positive Feedback Handler Agent for a banking customer "
    "support system. Write a warm, personalized thank-you message (2-3 "
    "sentences) that references what the customer praised. Be friendly and "
    "genuine, keep a professional banking tone."
)

EMPATHY_SYSTEM = (
    "You are the Negative Feedback Handler Agent for a banking customer "
    "support system. Write an empathetic apology (2-3 sentences) that "
    "acknowledges the customer's frustration, then end with exactly: "
    "Ticket {ticket_id} has been generated, and our team will follow up shortly."
)


class LLMClient:
    """Facade over the live LLM with automatic heuristic fallback."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 temperature: Optional[float] = None):
        self.api_key = api_key if api_key is not None else config.OPENAI_API_KEY
        self.model = model or config.LLM_MODEL
        self.temperature = (temperature if temperature is not None
                            else config.LLM_TEMPERATURE)
        self._engine = HeuristicEngine()
        self._openai = None
        self.last_prompt_trace: list = []
        self.last_error: Optional[str] = None
        if self.api_key:
            try:
                from openai import OpenAI  # optional dependency
                self._openai = OpenAI(api_key=self.api_key)
            except ImportError:
                self._openai = None
                self.last_error = "openai package not installed; using heuristic engine"

    # -- mode ---------------------------------------------------------------
    @property
    def is_live(self) -> bool:
        return self._openai is not None

    @property
    def mode(self) -> str:
        return "live-llm" if self._openai else "heuristic-fallback"

    # -- primitives ---------------------------------------------------------
    def _chat(self, system: str, user: str, json_mode: bool = False) -> Optional[str]:
        """Call the live LLM; return None on any failure (caller falls back)."""
        if not self._openai:
            return None
        self.last_prompt_trace = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": self.temperature,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = self._openai.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content
            self.last_prompt_trace.append({"role": "assistant", "content": content})
            return content
        except Exception as exc:  # network / auth / rate limits
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.last_prompt_trace.append({"role": "error", "content": self.last_error})
            return None

    # -- high-level operations ----------------------------------------------
    def classify(self, text: str) -> Classification:
        out = self._chat(CLASSIFY_SYSTEM, text, json_mode=True)
        if out:
            try:
                data = json.loads(out)
                label = str(data.get("label", "")).strip()
                if label in ALLOWED_LABELS:
                    conf = float(data.get("confidence", 0.9))
                    return Classification(label, max(0.0, min(1.0, conf)))
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        return self._engine.classify(text)

    def generate_thankyou(self, message: str) -> str:
        out = self._chat(THANKYOU_SYSTEM, message)
        return out if out else self._engine.generate_thankyou(message)

    def generate_empathy(self, message: str, ticket_id: str) -> str:
        out = self._chat(EMPATHY_SYSTEM.format(ticket_id=ticket_id), message)
        if out is None:
            return self._engine.generate_empathy(message, ticket_id)
        if f"Ticket {ticket_id} has been generated" in out:
            return out
        return f"{out.rstrip().rstrip('.')} Ticket {ticket_id} has been generated, and our team will follow up shortly."

    def extract_ticket_number(self, text: str) -> Optional[str]:
        """Ticket numbers are fixed-format (6 digits); regex is exact and free."""
        m = TICKET_RE.search(text)
        return m.group(1) if m else None
