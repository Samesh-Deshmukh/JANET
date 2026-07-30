# src/actions/calc_action.py
"""CALC intent — two paths, and it never guesses.

1. **The fast path** (this file): a small deterministic parser for ordinary
   spoken arithmetic. No model call, so "what's twenty times three" answers
   instantly. It only answers when it can account for EVERY number you said —
   see `_accounted_for`, which is what stops it inventing an answer out of a
   sentence it half-understood.
2. **The solver** (`ai_core.mathsolve`): the LLM translates the sentence into an
   expression and SymPy evaluates it. Slower, but it handles percentages the
   fast path misreads, and algebra, calculus and combinatorics it can't touch
   at all.

The fast path declining is therefore not a failure — it is the trigger for the
better path. That ordering was chosen after a live test where "what's 25% to 52"
(Whisper heard "to", not "of") made the old parser silently drop the 52 and
answer 0.25 for a question whose answer is 13.
"""
import ast
import operator
import re

from ai_core import llm
from ai_core.mathsolve import MathError
from ai_core import mathsolve
from intent import numwords

# Allowed AST operators — anything else is rejected (untrusted-input safe).
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Spoken operators -> symbols. Keys are space-surrounded; the text is space-padded
# first so an end-position word ("five squared") still matches. NOTE: "mod" is
# handled separately, AFTER percents, so the percent regex can't eat its "%".
_WORDS = {
    " plus ": " + ",
    " minus ": " - ",
    " times ": " * ",
    " multiplied by ": " * ",
    " divided by ": " / ",
    " to the power of ": " ** ",
    " squared ": " ** 2 ",
    " cubed ": " ** 3 ",
}

_MAX_EXPONENT = 100     # reject huge exponents so "2 to the power of 99999999" can't hang us

_NUMBER = re.compile(r"\d+(?:\.\d+)?")

CANT_WORK_IT_OUT = "I couldn't work that out."


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left = _eval(node.left)
        right = _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def _to_expression(text):
    """Turn spoken math into a plain arithmetic string."""
    text = numwords.words_to_numbers(text)
    text = f" {text} "                              # pad so boundary words match
    for word, symbol in _WORDS.items():
        text = text.replace(word, symbol)
    text = text.replace("^", " ** ")
    # Percentages while "%" still means percent: "of"/"off" before the bare form.
    text = re.sub(r"(\d+(?:\.\d+)?)\s*(?:percent|%)\s+of\s+(\d+(?:\.\d+)?)", r"(\1/100*\2)", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*(?:percent|%)\s+off\s+(\d+(?:\.\d+)?)", r"(\2-\1/100*\2)", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*(?:percent|%)", r"(\1/100)", text)
    # Now "%" is free to mean modulo (only produced by the words "mod"/"modulo").
    text = text.replace(" modulo ", " % ").replace(" mod ", " % ")
    # Square root of a number.
    text = re.sub(r"square root of\s+(\d+(?:\.\d+)?)", r"(\1**0.5)", text)
    # Longest run of math characters (drops words like "what is").
    candidates = re.findall(r"[-+*/().%\d\s]+", text)
    return max(candidates, key=len).strip() if candidates else ""


def _accounted_for(spoken, expression):
    """Did the expression use every number the person actually said?

    The parser builds its expression by grabbing the longest run of maths
    characters, so a word it doesn't understand doesn't produce an error — it
    quietly truncates the sum. "25% to 52" becomes "(25/100)", which is a
    perfectly valid expression, evaluates to 0.25, and is not remotely the
    answer to the question asked.

    Numbers are the one thing we can check cheaply: if you said 52 and it isn't
    in the expression, the sentence was not understood, whatever the parser
    thinks. (Extra numbers are fine — the percent rule introduces its own 100.)
    """
    said = set(_NUMBER.findall(numwords.words_to_numbers(spoken)))
    used = set(_NUMBER.findall(expression))
    return said and said <= used


def _fast_path(text):
    """Deterministic arithmetic, or None when this parser shouldn't be trusted."""
    math = _to_expression(text)
    if not re.search(r"[-+*/%]", math):
        return None            # no operator -> not arithmetic (also stops "510")
    if not _accounted_for(text, math):
        return None            # we dropped one of their numbers -> don't guess
    try:
        result = _eval(ast.parse(math, mode="eval").body)
    except ZeroDivisionError:
        return "I can't divide by zero."
    except (ValueError, SyntaxError, TypeError, OverflowError, RecursionError):
        return None            # let the solver try instead
    if isinstance(result, float):
        result = int(result) if result.is_integer() else round(result, 2)
    return f"That's {result}."


def handle(slots, ctx):
    # Read the RAW transcript (ctx.query), NOT a normalized slot: normalize strips
    # math symbols and decimals ("25% of 52" -> "25 of 52", "3.5" -> "3 5"), which
    # destroys the expression. Same reason general_action reads ctx.query.
    question = (ctx.query or "").strip()
    quick = _fast_path(question.lower())
    if quick is not None:
        return quick

    # The fast path wasn't sure. Hand the sentence to the LLM to translate, and
    # let SymPy do the actual maths.
    try:
        facts = mathsolve.solve(question)
    except MathError as exc:
        return f"I couldn't work that out — {exc}."
    except llm.LLMUnavailable:
        # No model, and the deterministic parser already declined. Saying so is
        # the honest answer; guessing is what got us here.
        return CANT_WORK_IT_OUT
    return facts if facts else CANT_WORK_IT_OUT
