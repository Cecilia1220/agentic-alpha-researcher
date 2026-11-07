"""Parse the tiny expression grammar. Python is never executed."""
import ast
from collections import Counter
from statistics import fmean

from .data import EXECUTION_FIELDS, FEATURES


class InvalidSignal(ValueError):
    pass


def compile_signal(expression: str) -> ast.expr:
    if not isinstance(expression, str) or not 1 <= len(expression) <= 160:
        raise InvalidSignal("expression must be a string of 1–160 characters")
    try:
        tree = ast.parse(expression.strip(), mode="eval").body
    except SyntaxError as exc:
        raise InvalidSignal("invalid expression syntax") from exc
    if sum(1 for _ in ast.walk(tree)) > 40:
        raise InvalidSignal("expression is too complex")
    if not isinstance(tree, ast.Call) or not isinstance(tree.func, ast.Name) or tree.func.id != "rank":
        raise InvalidSignal("expression must be rank(term)")

    def feature(node):
        if not isinstance(node, ast.Name):
            raise InvalidSignal("expected a feature name")
        if node.id in EXECUTION_FIELDS:
            raise InvalidSignal(f"{node.id} is reserved for outcomes/execution, which would leak future information")
        if node.id not in FEATURES:
            raise InvalidSignal(f"unregistered feature {node.id}")

    def term(node):
        if isinstance(node, ast.Name):
            feature(node)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            term(node.operand)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            term(node.left)
            term(node.right)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"lag", "mean"}:
            if len(node.args) != 2 or node.keywords:
                raise InvalidSignal("use lag(feature, n) or mean(feature, n)")
            feature(node.args[0])
            count = node.args[1]
            minimum = 1 if node.func.id == "lag" else 2
            if not isinstance(count, ast.Constant) or type(count.value) is not int or not minimum <= count.value <= 10:
                raise InvalidSignal(f"{node.func.id} requires a positive integer from {minimum} to 10")
        else:
            raise InvalidSignal("allowed terms are features, negation, addition, subtraction, lag, and mean")

    if len(tree.args) != 1 or tree.keywords:
        raise InvalidSignal("rank takes exactly one positional term")
    term(tree.args[0])
    return tree


def ranks(values: dict) -> dict:
    """Midranks, normalized to [0, 1], with identical values sharing ranks."""
    counts = Counter(values.values())
    if not counts:
        return {}
    scale = max(1, len(values) - 1)
    locations = {}
    offset = 0
    for value in sorted(counts):
        locations[value] = (offset + (counts[value] - 1) / 2) / scale
        offset += counts[value]
    return {key: locations[value] for key, value in values.items()}


def score(panel, tree: ast.expr, session: int) -> dict:
    def value(node, symbol):
        if isinstance(node, ast.Name):
            return panel.feature(node.id, session, session, symbol)
        if isinstance(node, ast.UnaryOp):
            operand = value(node.operand, symbol)
            return None if operand is None else -operand
        if isinstance(node, ast.BinOp):
            left, right = value(node.left, symbol), value(node.right, symbol)
            if left is None or right is None:
                return None
            return left + right if isinstance(node.op, ast.Add) else left - right
        name, count = node.args[0].id, node.args[1].value
        if node.func.id == "lag":
            return panel.feature(name, session - count, session, symbol)
        history = [
            panel.feature(name, past, session, symbol)
            for past in range(session - count + 1, session + 1)
        ]
        return None if None in history else fmean(history)

    observations = {symbol: value(tree.args[0], symbol) for symbol in panel.symbols}
    return ranks({symbol: x for symbol, x in observations.items() if x is not None})
