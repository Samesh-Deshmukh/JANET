# src/utils/history.py
"""Short-term conversation memory: the recent back-and-forth, windowed.

The GENERAL handler feeds this to the local LLM so follow-up questions have
context ("what about Germany?" after "capital of France?"). It records EVERY
addressed exchange (any intent), not just LLM ones, so a general question can
also reference that you just set a timer or asked the time.

In-memory only (resets on restart). This is the short-term-window stepping stone
toward a fuller memory system later.
"""
import time
from collections import deque
from dataclasses import dataclass, field

# 10 exchanges = 20 messages. Voice chats rarely need more; bump here to tune.
MAX_EXCHANGES = 10


@dataclass
class Turn:
    """One exchange, plus what JANET did to produce it.

    `facts` is what the action returned and `reasoning` is why JANET replied the
    way it did — neither belongs in the spoken chat log, but both are context the
    LLM benefits from on the next turn.
    """
    user: str
    assistant: str
    facts: str = ""
    reasoning: str = ""
    # When JANET finished this exchange. The addressing fallback uses it to ask
    # "did I speak recently?" — a follow-up seconds after a reply is far more
    # likely to be aimed at JANET than the same words in a silent room.
    at: float = field(default_factory=time.monotonic)


class ConversationHistory:
    def __init__(self, max_exchanges=MAX_EXCHANGES):
        # deque(maxlen=...) drops the oldest exchange automatically when full.
        self._turns = deque(maxlen=max_exchanges)

    def add(self, user_text, assistant_text, facts=None, reasoning=None):
        """Record one completed exchange. `facts`/`reasoning` are optional so the
        older two-argument calls keep working."""
        self._turns.append(Turn(user_text, assistant_text, facts or "", reasoning or ""))

    def messages(self):
        """Flatten to the [{role, content}, ...] list the chat API expects,
        oldest first. Deliberately just the spoken conversation — clean chat."""
        out = []
        for turn in self._turns:
            out.append({"role": "user", "content": turn.user})
            out.append({"role": "assistant", "content": turn.assistant})
        return out

    def context_block(self, limit=3):
        """The recent turns' actions and reasoning, as a short note for the LLM.

        Kept out of `messages()` on purpose: mixing "why I said that" into the
        chat transcript confuses the model about who said what. This goes in as a
        separate system note instead.
        """
        lines = []
        for turn in list(self._turns)[-limit:]:
            if turn.facts:
                lines.append(f"- action returned: {turn.facts}")
            if turn.reasoning:
                lines.append(f"- you reasoned: {turn.reasoning}")
        if not lines:
            return ""
        return "Recently, for context:\n" + "\n".join(lines)

    def seconds_since_last(self):
        """How long since JANET last replied, or None if it hasn't yet.

        monotonic, not wall-clock, so a clock change can't make this negative.
        """
        if not self._turns:
            return None
        return time.monotonic() - self._turns[-1].at
