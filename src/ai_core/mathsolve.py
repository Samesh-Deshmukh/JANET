# src/ai_core/mathsolve.py
"""Maths by translation: the LLM reads the sentence, SymPy does the arithmetic.

Language models are fluent and bad at calculating. SymPy is the exact opposite —
it cannot read "what's the derivative of x squared" but it will differentiate it
perfectly. So each does only the half it is good at:

    "what's the derivative of x squared plus three x"
        -> LLM  : {operation: "differentiate", expression: "x**2 + 3*x", variable: "x"}
        -> SymPy: 2*x + 3
        -> FACTS: "d/dx(x**2 + 3*x) = 2*x + 3"

This is the same split the rest of JANET already runs on (handlers produce facts,
the responder phrases them) — here the model translates and never computes.

**The model never writes code.** It fills in slots: `operation` is an enum, so
constrained decoding makes it impossible to name an operation that doesn't
exist, and this module decides which SymPy function that maps to. The expression
string is parsed with a restricted namespace (see `_parse`), never eval()'d.
"""
import ast
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _Timeout

import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

from ai_core import llm

# Every name the expression parser will resolve. Anything else in the string
# becomes a plain Symbol (so "x" works), which means a stray "__import__" is an
# unknown symbol rather than a callable. This allowlist IS the sandbox.
_ALLOWED = {
    name: getattr(sympy, name)
    for name in (
        "sqrt", "cbrt", "root", "exp", "log", "Abs", "sign",
        "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
        "sinh", "cosh", "tanh", "floor", "ceiling",
        "factorial", "binomial", "gcd", "lcm", "Rational", "Integer", "Float",
        "pi", "E", "oo", "Eq", "Max", "Min", "simplify", "Symbol",
    )
}
_ALLOWED["ln"] = sympy.log          # people say "natural log"; SymPy calls it log
_ALLOWED["infinity"] = sympy.oo

# Cheap structural rejections before parsing. Dunders and `lambda` have no place
# in a spoken sum, and their only purpose in this string would be an escape.
_FORBIDDEN = re.compile(r"__|\blambda\b|\bimport\b")
_MAX_EXPRESSION_CHARS = 200

# A spoken question can ask for something that takes effectively forever —
# "the 90000th prime", a nasty integral. SymPy has no internal time limit, and
# JANET is single-threaded, so an unbounded call would freeze the microphone.
SOLVE_TIMEOUT_S = 6

# Guards against the cheap ways to make an exact-arithmetic library hang:
# 2**10**9 and factorial(10**6) are both short strings and enormous work.
_MAX_EXPONENT = 1000
_MAX_FACTORIAL = 1000

OPERATIONS = ("evaluate", "simplify", "expand", "factor", "solve",
              "differentiate", "integrate", "limit", "none")

_SCHEMA = {
    "type": "object",
    "properties": {
        "operation": {"type": "string", "enum": list(OPERATIONS)},
        "expression": {"type": "string"},
        "variable": {"type": "string"},
        # Definite integrals and limits only.
        "lower": {"type": "string"},
        "upper": {"type": "string"},
    },
    "required": ["operation", "expression", "variable", "lower", "upper"],
}

PROMPT = (
    "You translate a spoken maths question into a form a computer algebra system "
    "can evaluate. You do NOT do the maths yourself — never calculate, never "
    "give an answer. Translate only.\n"
    "\n"
    "Write `expression` in Python/SymPy syntax: ** for powers, * for every "
    "multiplication (write 3*x, not 3x), sqrt(), log(), sin(), pi, factorial(), "
    "binomial() for 'n choose k'. Percentages become arithmetic: '25% of 52' is "
    "'25/100*52', '15% off 200' is '200 - 15/100*200'.\n"
    "\n"
    "Pick the operation:\n"
    "  evaluate      - work out a value ('what's 25% of 52', 'sqrt of 144')\n"
    "  solve         - find the unknown ('solve x squared equals 4'); write the "
    "equation as Eq(left, right) and set variable\n"
    "  differentiate - a derivative; set variable\n"
    "  integrate     - an integral; set variable, and lower/upper for a definite one\n"
    "  limit         - a limit; set variable and lower (the point approached)\n"
    "  simplify / expand / factor - rearrange an expression\n"
    "  none          - not a maths question at all, or too vague to translate\n"
    "\n"
    "The speech-to-text is imperfect: 'x squared' may arrive as 'x squid', 'plus' "
    "as 'blues', and a spoken 'of' can come through as 'to' or 'off'. Read for "
    "the obvious intended sum. But if you genuinely cannot tell what was asked, "
    "use operation 'none' rather than guessing at numbers.\n"
    "\n"
    "Leave variable, lower and upper as empty strings when they don't apply."
)


class MathError(Exception):
    """The question was maths, but we could not answer it."""


# The only syntax a spoken sum needs. Anything outside this list — an attribute
# lookup, a subscript, a string, a comprehension — is rejected before SymPy sees
# the text, so there is no expression we merely HOPE is harmless.
_OK_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Call,
    ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
    ast.Pow, ast.USub, ast.UAdd,
)


def _check_syntax(text):
    """Vet the expression as SYNTAX, before anything can evaluate it.

    This has to happen first because SymPy computes while it parses. By the time
    a normal `parse_expr("2**10**9")` returns, you are already holding a
    300-million-digit integer, and `factorial(1000000)` burns three seconds —
    inspecting the result afterwards is far too late, and `solve`'s timeout
    can't help either, because the damage is done during parsing.

    Python's own `ast.parse` reads the string to a tree WITHOUT running it, so
    the size limits below are checked against syntax nobody has evaluated yet.
    It also lets the node allowlist be a real allowlist rather than a regex.
    """
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise MathError(f"could not parse {text!r}") from exc

    for node in ast.walk(tree):
        if not isinstance(node, _OK_NODES):
            raise MathError(f"{type(node).__name__} is not allowed in a sum")

        # Only numbers are literals; a string has no place in an arithmetic
        # expression and is how most escapes would start.
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise MathError("only numbers are allowed")

        # Calls must name a function from the allowlist directly — no
        # expressions in the function position.
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED:
                raise MathError("unknown function in expression")
            # factorial(1000000) is a short string and several seconds of work,
            # and factorial(10**6) is the same thing written so the size check
            # can't read it — so a computed argument is refused outright, for
            # the same reason as a power tower.
            if node.func.id in ("factorial", "binomial") and node.args:
                size = _literal(node.args[0])
                if size is None:
                    if _contains_power(node.args[0]):
                        raise MathError("that factorial is too large to work out")
                elif abs(size) > _MAX_FACTORIAL:
                    raise MathError("factorial too large")

        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            exponent = _literal(node.right)
            if exponent is None:
                # Not a plain number: a power tower like 2**10**9, or symbolic.
                # Measuring it would mean building it, so refuse. Nobody asks a
                # voice assistant for a power tower.
                if _contains_power(node.right):
                    raise MathError("nested powers are too large to work out")
            elif abs(exponent) > _MAX_EXPONENT:
                raise MathError("exponent too large")


def _literal(node):
    """The numeric value of a constant node (allowing a leading minus), else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _literal(node.operand)
        return None if inner is None else -inner if isinstance(node.op, ast.USub) else inner
    return None


def _contains_power(node):
    return any(isinstance(child, ast.BinOp) and isinstance(child.op, ast.Pow)
               for child in ast.walk(node))


def _parse(text):
    """Expression string -> SymPy object, with a restricted namespace."""
    text = (text or "").strip()
    if not text:
        raise MathError("empty expression")
    if len(text) > _MAX_EXPRESSION_CHARS:
        raise MathError("expression too long")
    if _FORBIDDEN.search(text):
        raise MathError("forbidden name in expression")
    _check_syntax(text)                 # nothing has evaluated yet — see above
    try:
        # global_dict is the allowlist; auto_symbol (in the standard
        # transformations) turns every OTHER name into a plain Symbol, which is
        # a second line of defence: Symbol('open') isn't callable.
        return parse_expr(text, global_dict=dict(_ALLOWED),
                          transformations=standard_transformations)
    except Exception as exc:                       # any parse failure at all
        raise MathError(f"could not parse {text!r}") from exc


def _symbol(name, expr):
    """The variable to work with: the one named, else the only one present."""
    name = (name or "").strip()
    if name:
        return sympy.Symbol(name)
    free = sorted(expr.free_symbols, key=str)
    if len(free) == 1:
        return free[0]
    raise MathError("which variable?")


def _speak_number(value):
    """A SymPy result -> a short string worth reading out.

    Exact forms are kept ('1/3' stays a third), but anything irrational also gets
    a decimal, because "the square root of two" is not an answer people want.
    """
    if not value.is_number:
        return str(value)
    if value.is_Integer:
        return str(value)
    try:
        decimal = float(value)
    except (TypeError, ValueError):
        return str(value)
    rounded = round(decimal, 4)
    tidy = int(rounded) if float(rounded).is_integer() else rounded
    # An exact fraction is worth saying alongside its decimal; sqrt(2) is not
    # worth saying at all once you have 1.4142.
    if value.is_Rational:
        return f"{value} ({tidy})"
    return str(tidy)


def _run(operation, expr, variable, lower, upper, shown):
    """Dispatch to SymPy. Returns the FACTS string.

    `shown` is the expression as the model wrote it, used for the left-hand side
    of the answer. It matters because SymPy evaluates while parsing: by the time
    `expr` exists, "25/100*52" is already the integer 13, and reporting
    "13 = 13" tells the responder nothing about what was asked.
    """
    if operation == "evaluate":
        value = sympy.simplify(expr)
        return f"{shown} = {_speak_number(value)}"

    if operation == "simplify":
        return f"{shown} simplifies to {sympy.simplify(expr)}"

    if operation == "expand":
        return f"{shown} expands to {sympy.expand(expr)}"

    if operation == "factor":
        return f"{shown} factors to {sympy.factor(expr)}"

    if operation == "solve":
        var = _symbol(variable, expr)
        roots = sympy.solve(expr, var)
        if not roots:
            return f"{shown} has no solution"
        answers = ", ".join(_speak_number(r) for r in roots)
        return f"solving {shown} gives {var} = {answers}"

    if operation == "differentiate":
        var = _symbol(variable, expr)
        return f"d/d{var}({shown}) = {sympy.diff(expr, var)}"

    if operation == "integrate":
        var = _symbol(variable, expr)
        if lower and upper:
            lo, hi = _parse(lower), _parse(upper)
            value = sympy.integrate(expr, (var, lo, hi))
            return f"integral of {shown} d{var} from {lo} to {hi} = {_speak_number(value)}"
        return f"integral of {shown} d{var} = {sympy.integrate(expr, var)} + C"

    if operation == "limit":
        var = _symbol(variable, expr)
        point = _parse(lower) if lower else sympy.Integer(0)
        value = sympy.limit(expr, var, point)
        return f"limit of {shown} as {var} -> {point} = {_speak_number(value)}"

    raise MathError(f"unsupported operation {operation!r}")


def translate(question):
    """Ask the LLM for the slots. Raises llm.LLMUnavailable if it can't be reached."""
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": f"Translate this: {question}"},
    ]
    return llm.chat(messages, schema=_SCHEMA, max_tokens=250)


def solve(question):
    """Spoken maths question -> a FACTS string, or None if it isn't maths.

    Raises `llm.LLMUnavailable` when the model can't be reached, so the caller
    can fall back to the offline parser rather than pretending it failed.
    """
    data = translate(question)
    operation = (data.get("operation") or "none").strip()
    if operation == "none" or operation not in OPERATIONS:
        return None

    expression = data.get("expression") or ""
    variable = (data.get("variable") or "").strip()
    lower = (data.get("lower") or "").strip()
    upper = (data.get("upper") or "").strip()

    expr = _parse(expression)
    print(f"🧮 {operation}: {expr}" + (f"  d{variable}" if variable else ""))

    # SymPy has no time limit of its own and JANET is single-threaded, so an
    # unlucky integral would deafen the microphone.
    #
    # NOTE: deliberately not `with ThreadPoolExecutor(...)`. The context manager
    # calls shutdown(wait=True) on the way out, which blocks until the worker
    # finishes — i.e. it would wait for the exact runaway computation the
    # timeout exists to escape. We abandon the thread instead: Python cannot
    # kill one, so it keeps burning a core until it finishes, but JANET stays
    # responsive, which is the thing that matters.
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(_run, operation, expr, variable, lower, upper,
                         expression.strip())
    try:
        return future.result(timeout=SOLVE_TIMEOUT_S)
    except _Timeout:
        raise MathError("that one took too long to work out")
    finally:
        pool.shutdown(wait=False)
