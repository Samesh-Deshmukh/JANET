# src/utils/history.py
"""Short-term conversation memory: the recent back-and-forth, windowed.

The GENERAL handler feeds this to the local LLM so follow-up questions have
context ("what about Germany?" after "capital of France?"). It records EVERY
addressed exchange (any intent), not just LLM ones, so a general question can
also reference that you just set a timer or asked the time.

In-memory only (resets on restart). This is the short-term-window stepping stone
toward a fuller memory system later.
"""
from collections import deque

# 10 exchanges = 20 messages. Voice chats rarely need more; bump here to tune.
MAX_EXCHANGES = 10


class ConversationHistory:
    def __init__(self, max_exchanges=MAX_EXCHANGES):
        # deque(maxlen=...) drops the oldest exchange automatically when full.
        self._turns = deque(maxlen=max_exchanges)

    def add(self, user_text, assistant_text):
        """Record one completed exchange (what the user said, what JANET replied)."""
        self._turns.append((user_text, assistant_text))

    def messages(self):
        """Flatten to the [{role, content}, ...] list Ollama's chat API expects,
        oldest first."""
        out = []
        for user_text, assistant_text in self._turns:
            out.append({"role": "user", "content": user_text})
            out.append({"role": "assistant", "content": assistant_text})
        return out
