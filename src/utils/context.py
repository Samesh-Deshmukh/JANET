from dataclasses import dataclass
from typing import Callable


@dataclass
class Context:
    """Passed to every action handler. Grows over time (e.g. conversation
    history in Block 4). `speak` lets async actions produce deferred speech."""
    speak: Callable[[str], None]
    query: str = ""
