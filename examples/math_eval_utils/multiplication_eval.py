import random
from typing import Dict, List, Optional


EVAL_INSTRUCTION = (
    "Solve the multiplication problem. Show brief working, then end with the final line "
    "\"ANSWER: <number>\" using only digits for the number."
)


def _rand_four_digit_no_zero(rng: random.Random) -> int:
    while True:
        value = rng.randint(1000, 9999)
        if value % 10 != 0:
            return value


def _rand_no_trailing_zero(rng: random.Random, min_value: int, max_value: int) -> int:
    while True:
        value = rng.randint(min_value, max_value)
        if value % 10 != 0:
            return value


def generate_multiplication_problems(
    count: int,
    *,
    seed: Optional[int] = None,
    min_value: int = 1000,
    max_value: int = 9999,
    digits_a: Optional[int] = None,
    digits_b: Optional[int] = None,
    exclude_trailing_zero: bool = True,
) -> List[Dict[str, object]]:
    if count <= 0:
        return []
    if digits_a is not None or digits_b is not None:
        if digits_a is None or digits_b is None:
            raise ValueError("digits_a and digits_b must be provided together.")
        if digits_a < 1 or digits_b < 1:
            raise ValueError("digits_a and digits_b must be >= 1.")
        min_value = 10 ** (digits_a - 1)
        max_value = 10**digits_a - 1
        min_value_b = 10 ** (digits_b - 1)
        max_value_b = 10**digits_b - 1
    else:
        if min_value < 1 or max_value < 1 or min_value > max_value:
            raise ValueError("min_value/max_value must define a valid positive range.")
        min_value_b = min_value
        max_value_b = max_value

    rng = random.Random(seed)
    problems: List[Dict[str, object]] = []
    for i in range(count):
        if exclude_trailing_zero:
            if digits_a is None:
                a = _rand_four_digit_no_zero(rng)
            else:
                a = _rand_no_trailing_zero(rng, min_value, max_value)
            if digits_b is None:
                b = _rand_four_digit_no_zero(rng)
            else:
                b = _rand_no_trailing_zero(rng, min_value_b, max_value_b)
        else:
            a = rng.randint(min_value, max_value)
            b = rng.randint(min_value_b, max_value_b)
        expr = f"{a} * {b}"
        problems.append(
            {
                "id": i,
                "a": a,
                "b": b,
                "expression": expr,
                "question": f"{expr} = ?",
                "answer": a * b,
            }
        )
    return problems


def build_multiplication_prompts(problems: List[Dict[str, object]]) -> List[str]:
    return [f"{EVAL_INSTRUCTION}\nProblem: {p['expression']}" for p in problems]


__all__ = ["generate_multiplication_problems", "build_multiplication_prompts", "EVAL_INSTRUCTION"]
