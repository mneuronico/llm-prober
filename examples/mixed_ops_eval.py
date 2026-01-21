import random
from typing import Dict, List, Optional


EVAL_INSTRUCTION = (
    "Solve the arithmetic expression. Show brief working, then end with the final line "
    "\"ANSWER: <number>\" using only digits for the number."
)


def _rand_n_digit(rng: random.Random, digits: int, exclude_trailing_zero: bool) -> int:
    if digits < 1:
        raise ValueError("digits must be >= 1.")
    low = 10 ** (digits - 1)
    high = 10**digits - 1
    while True:
        value = rng.randint(low, high)
        if not exclude_trailing_zero or value % 10 != 0:
            return value


def _apply_op(left: int, op: str, right: int) -> int:
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    raise ValueError(f"Unsupported operator: {op}")


def generate_mixed_ops_problems(
    count: int,
    *,
    seed: Optional[int] = None,
    num_terms: int = 3,
    digits: int = 2,
    operators: Optional[List[str]] = None,
    exclude_trailing_zero: bool = False,
) -> List[Dict[str, object]]:
    if count <= 0:
        return []
    if num_terms < 2:
        raise ValueError("num_terms must be >= 2.")
    if operators is None:
        operators = ["+", "-", "*"]

    rng = random.Random(seed)
    problems: List[Dict[str, object]] = []
    for i in range(count):
        values = [_rand_n_digit(rng, digits, exclude_trailing_zero) for _ in range(num_terms)]
        ops = [rng.choice(operators) for _ in range(num_terms - 1)]

        expr = str(values[0])
        total = values[0]
        for op, value in zip(ops, values[1:]):
            expr = f"({expr} {op} {value})"
            total = _apply_op(total, op, value)

        problems.append(
            {
                "id": i,
                "values": values,
                "ops": ops,
                "expression": expr,
                "question": f"{expr} = ?",
                "answer": total,
            }
        )
    return problems


def build_mixed_ops_prompts(problems: List[Dict[str, object]]) -> List[str]:
    return [f"{EVAL_INSTRUCTION}\nProblem: {p['expression']}" for p in problems]


__all__ = ["generate_mixed_ops_problems", "build_mixed_ops_prompts", "EVAL_INSTRUCTION"]
