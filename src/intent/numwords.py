# src/intent/numwords.py
"""Convert spoken English number words in a string to digit strings (pure).

Used by the calculator so "twenty times three" becomes "20 times 3". Bounded to
0-999,999 (ones, teens, tens, hundred, thousand); "and" is ignored inside a
number. Hand-rolled (not a dependency) so every line is explainable. Non-number
words pass through.
"""
_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_NUMBER_WORDS = set(_UNITS) | set(_TENS) | {"hundred", "thousand", "and"}


def _run_value(tokens):
    """Numeric value of a run of number-word tokens (accumulator)."""
    total = 0
    current = 0
    for tok in tokens:
        if tok in _UNITS:
            current += _UNITS[tok]
        elif tok in _TENS:
            current += _TENS[tok]
        elif tok == "hundred":
            current = (current or 1) * 100
        elif tok == "thousand":
            total += (current or 1) * 1000
            current = 0
        # "and" contributes nothing
    return total + current


def _flush(run, out):
    if not run:
        return
    if all(t == "and" for t in run):
        out.extend(run)                 # a lone "and" is a word, not a number
    else:
        out.append(str(_run_value(run)))


def words_to_numbers(text):
    """Replace each run of number words with its digit string; keep other words."""
    out = []
    run = []
    for tok in text.split():
        if tok in _NUMBER_WORDS:
            run.append(tok)
        else:
            _flush(run, out)
            run = []
            out.append(tok)
    _flush(run, out)
    return " ".join(out)
