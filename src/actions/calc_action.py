import ast
import operator
import re

# Only these node/operator types are allowed — anything else is rejected.
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Spoken operators → symbols. Surrounding spaces avoid matching inside words.
_WORDS = {
    " plus ": " + ",
    " minus ": " - ",
    " times ": " * ",
    " multiplied by ": " * ",
    " divided by ": " / ",
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def handle(slots, ctx):
    text = (slots.get("expression") or "").lower()
    for word, symbol in _WORDS.items():
        text = text.replace(word, symbol)
    # Pull out the longest run of math characters, dropping words like "what is".
    candidates = re.findall(r"[-+*/().%\d\s]+", text)
    math = max(candidates, key=len).strip() if candidates else ""
    try:
        tree = ast.parse(math, mode="eval")
        result = _eval(tree.body)
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError, OverflowError, RecursionError):
        return "I couldn't work that out."
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return f"That's {result}."
