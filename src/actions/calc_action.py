# src/actions/calc_action.py
import ast
import operator
import re

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


def handle(slots, ctx):
    math = _to_expression((slots.get("expression") or "").lower())
    if not re.search(r"[-+*/%]", math):
        # No operator -> not a calculation (also stops misheard input like "510").
        return "I couldn't work that out."
    try:
        tree = ast.parse(math, mode="eval")
        result = _eval(tree.body)
    except ZeroDivisionError:
        return "I can't divide by zero."
    except (ValueError, SyntaxError, TypeError, OverflowError, RecursionError):
        return "I couldn't work that out."
    if isinstance(result, float):
        result = int(result) if result.is_integer() else round(result, 2)
    return f"That's {result}."
