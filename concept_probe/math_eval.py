import random
import re
from typing import Dict, List, Optional, Tuple, Union

Number = Union[int, float]


def _format_sum(values: List[int]) -> str:
    if not values:
        return "0"
    parts = [str(values[0])]
    for v in values[1:]:
        if v < 0:
            parts.append(f"- {abs(v)}")
        else:
            parts.append(f"+ {v}")
    return " ".join(parts)


def generate_addition_problems(
    count: int,
    *,
    seed: Optional[int] = None,
    min_terms: int = 2,
    max_terms: int = 4,
    min_value: int = 1,
    max_value: int = 99,
    allow_negative: bool = False,
) -> List[Dict[str, object]]:
    if count <= 0:
        return []
    if min_terms < 2 or max_terms < min_terms:
        raise ValueError("min_terms must be >= 2 and max_terms must be >= min_terms.")
    if min_value > max_value:
        raise ValueError("min_value must be <= max_value.")

    rng = random.Random(seed)
    problems: List[Dict[str, object]] = []
    for i in range(count):
        num_terms = rng.randint(min_terms, max_terms)
        values: List[int] = []
        for _ in range(num_terms):
            v = rng.randint(min_value, max_value)
            if allow_negative and rng.random() < 0.3:
                v = -v
            values.append(v)
        expr = _format_sum(values)
        problems.append(
            {
                "id": i,
                "values": values,
                "expression": expr,
                "question": f"{expr} = ?",
                "answer": sum(values),
            }
        )
    return problems


def _answer_regex(marker: str) -> re.Pattern:
    marker = marker.strip()
    if not marker:
        marker = "ANSWER:"
    return re.compile(rf"{re.escape(marker)}\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)


def _coerce_number(text: str) -> Number:
    if "." in text:
        return float(text)
    return int(text)


def extract_answer(
    text: str,
    *,
    marker: str = "ANSWER:",
    allow_fallback: bool = True,
) -> Tuple[Optional[Number], bool]:
    if not text:
        return None, False

    match = _answer_regex(marker).search(text)
    if match:
        return _coerce_number(match.group(1)), True

    if not allow_fallback:
        return None, False

    nums = re.findall(r"[+-]?\d+(?:\.\d+)?", text)
    if not nums:
        return None, False
    return _coerce_number(nums[-1]), False


def evaluate_answer(
    text: str,
    expected: Number,
    *,
    marker: str = "ANSWER:",
    require_marker: bool = True,
    atol: float = 1e-6,
) -> Dict[str, object]:
    value, used_marker = extract_answer(text, marker=marker, allow_fallback=not require_marker)
    if value is None:
        return {"correct": False, "parsed": None, "used_marker": used_marker}

    if require_marker and not used_marker:
        return {"correct": False, "parsed": value, "used_marker": used_marker}

    correct = abs(float(value) - float(expected)) <= float(atol)
    return {"correct": bool(correct), "parsed": value, "used_marker": used_marker}
