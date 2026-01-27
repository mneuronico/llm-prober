import os
import random
import re
from typing import Dict, Iterable, List, Optional, Tuple, Union

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None
try:
    from huggingface_hub import HfApi
except Exception:
    HfApi = None


EVAL_INSTRUCTION = (
    "You will be given a short social context, a question, and three answer options.\n"
    "Choose the best option. Give brief reasoning.\n"
    "Then write 'ANSWER: <letter>' on its own line, where <letter> is A, B, or C."
)

LABEL_TO_LETTER = {"1": "A", "2": "B", "3": "C"}
LETTER_TO_ANSWER_KEY = {"A": "answerA", "B": "answerB", "C": "answerC"}

_ANSWER_REGEX = re.compile(r"\bANSWER\s*:\s*[\(\[]?\s*([ABC])\b", re.IGNORECASE)
_DATASET_CACHE: Dict[str, object] = {}


def _parquet_split_url(dataset_name: str, split: str, filename: str = "0000.parquet") -> str:
    safe_name = dataset_name.strip()
    if not safe_name:
        raise ValueError("dataset_name must be non-empty.")
    return (
        f"https://huggingface.co/datasets/{safe_name}/resolve/"
        f"refs%2Fconvert%2Fparquet/default/{split}/{filename}"
    )


def _list_parquet_urls(dataset_name: str, split: str) -> List[str]:
    if HfApi is None:
        return [_parquet_split_url(dataset_name, split)]

    api = HfApi()
    prefix = f"refs/convert/parquet/default/{split}/"
    files = api.list_repo_files(repo_id=dataset_name, repo_type="dataset")
    parquet_files = [f for f in files if f.startswith(prefix) and f.endswith(".parquet")]
    if not parquet_files:
        return [_parquet_split_url(dataset_name, split)]
    return [
        _parquet_split_url(dataset_name, split, filename=f.split("/")[-1])
        for f in parquet_files
    ]


def _ensure_hf_timeouts(download_timeout: int = 60, etag_timeout: int = 60) -> None:
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(download_timeout))
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(etag_timeout))


def load_social_iqa_split(
    split: str = "validation",
    *,
    dataset_name: str = "allenai/social_i_qa",
    data_files: Optional[Union[str, List[str], Dict[str, Union[str, List[str]]]]] = None,
    download_timeout: int = 60,
) -> object:
    if load_dataset is None:
        raise ImportError("Missing dependency 'datasets'. Install with: pip install datasets")

    _ensure_hf_timeouts(download_timeout=download_timeout)

    cache_key = f"{dataset_name}::{split}"
    if cache_key in _DATASET_CACHE:
        return _DATASET_CACHE[cache_key]

    try:
        if data_files is not None:
            dataset = load_dataset("parquet", data_files=data_files, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
    except Exception:
        if data_files is not None:
            raise
        parquet_urls = _list_parquet_urls(dataset_name, split)
        dataset = load_dataset("parquet", data_files={split: parquet_urls}, split=split)

    _DATASET_CACHE[cache_key] = dataset
    return dataset


def normalize_social_iqa_row(row: Dict[str, object]) -> Dict[str, object]:
    label = str(row.get("label", "")).strip()
    expected = LABEL_TO_LETTER.get(label)
    if expected is None:
        raise ValueError(f"Unrecognized SocialIQA label: {row.get('label')}")

    expected_key = LETTER_TO_ANSWER_KEY[expected]
    context = str(row.get("context", "")).strip()
    question = str(row.get("question", "")).strip()
    answer_a = str(row.get("answerA", "")).strip()
    answer_b = str(row.get("answerB", "")).strip()
    answer_c = str(row.get("answerC", "")).strip()
    expected_text = str(row.get(expected_key, "")).strip()

    return {
        "context": context,
        "question": question,
        "answerA": answer_a,
        "answerB": answer_b,
        "answerC": answer_c,
        "label": label,
        "expected": expected,
        "expected_text": expected_text,
    }


def build_social_iqa_prompt(item: Dict[str, object]) -> str:
    return (
        f"{EVAL_INSTRUCTION}\n\n"
        f"Context: {item.get('context', '')}\n"
        f"Question: {item.get('question', '')}\n"
        "Options:\n"
        f"A) {item.get('answerA', '')}\n"
        f"B) {item.get('answerB', '')}\n"
        f"C) {item.get('answerC', '')}\n"
        "Reasoning:"
    )


def extract_answer_letter(text: str) -> Tuple[Optional[str], bool]:
    if not text:
        return None, False

    match = _ANSWER_REGEX.search(text)
    if match:
        return match.group(1).upper(), True

    fallback = re.findall(r"\b([ABC])\b", text.strip())
    if not fallback:
        return None, False
    return fallback[-1].upper(), False


def evaluate_answer_letter(
    text: str,
    expected: str,
    *,
    require_marker: bool = True,
) -> Dict[str, object]:
    parsed, used_marker = extract_answer_letter(text)
    if parsed is None:
        return {"correct": False, "parsed": None, "used_marker": used_marker}

    if require_marker and not used_marker:
        return {"correct": False, "parsed": parsed, "used_marker": used_marker}

    correct = parsed.strip().upper() == str(expected).strip().upper()
    return {"correct": bool(correct), "parsed": parsed, "used_marker": used_marker}


def social_iqa_evaluator(completion: str, item: Dict[str, object]) -> Dict[str, object]:
    expected = str(item.get("expected", "")).strip()
    result = evaluate_answer_letter(completion, expected, require_marker=True)
    result["expected"] = expected
    return result


def generate_social_iqa_item(
    *,
    split: str = "validation",
    rng: Optional[random.Random] = None,
    dataset_name: str = "allenai/social_i_qa",
    data_files: Optional[Union[str, List[str], Dict[str, Union[str, List[str]]]]] = None,
    download_timeout: int = 60,
) -> Dict[str, object]:
    dataset = load_social_iqa_split(
        split,
        dataset_name=dataset_name,
        data_files=data_files,
        download_timeout=download_timeout,
    )
    rng = rng or random.Random()
    idx = rng.randrange(len(dataset))
    row = dataset[int(idx)]
    return normalize_social_iqa_row(row)


__all__ = [
    "EVAL_INSTRUCTION",
    "build_social_iqa_prompt",
    "evaluate_answer_letter",
    "extract_answer_letter",
    "generate_social_iqa_item",
    "load_social_iqa_split",
    "normalize_social_iqa_row",
    "social_iqa_evaluator",
]
