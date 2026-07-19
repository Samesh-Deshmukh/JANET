from dataclasses import dataclass, field
from typing import Callable

from utils.history import ConversationHistory


@dataclass
class Context:
    """Passed to every action handler. `speak` lets async actions produce
    deferred speech; `history` is the short-term conversation memory the LLM
    handler reads. Defaults to an empty history so a bare Context is always
    usable; main() passes the shared instance so memory persists across turns."""
    speak: Callable[[str], None]
    query: str = ""
    history: ConversationHistory = field(default_factory=ConversationHistory)
