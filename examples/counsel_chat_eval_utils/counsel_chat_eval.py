import json
import os
import random
import time
from typing import Dict, List, Optional, Sequence, Union

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None
try:
    from groq import Groq
except Exception:
    Groq = None
try:
    from huggingface_hub import HfApi
except Exception:
    HfApi = None


SYSTEM_PROMPT = (
    "You are a strict evaluator of therapist responses. "
    "You will receive a JSON object with fields: "
    "question (string), reference_answers (list of strings), model_answer (string). "
    "Decide if the model_answer is at least as good as the reference_answers overall. "
    "Do NOT penalize the model if the answer is cut off; that might be an artifact, and not grounds for rating a response as less good. "
    "Judge the content that was provided, even if it is incomplete due to cutoff, and decide if it is as good as the reference answers overall. "
    "Return only valid JSON with exactly these keys: "
    '{"verdict":"YES or NO","cutoff":true or false,"reason":"why it is or it is not as good (cutoff cannot be a reason)"}'
)

DEFAULT_JUDGE_MODEL = "openai/gpt-oss-20b"

QUESTION_ID_CANDIDATES = [
    "questionID",
    "questionId",
    "question_id",
    "qid",
    "id",
]
QUESTION_TEXT_CANDIDATES = [
    "questionText",
    "question_text",
    "question",
    "questionTitle",
    "question_title",
    "title",
]
ANSWER_TEXT_CANDIDATES = [
    "answerText",
    "answer_text",
    "answer",
    "response",
    "responseText",
]

_DATASET_CACHE: Dict[str, object] = {}
_SAMPLER_CACHE: Dict[str, "CounselChatSampler"] = {}


def _parquet_split_url(dataset_name: str, split: str, filename: str) -> str:
    safe_name = dataset_name.strip()
    if not safe_name:
        raise ValueError("dataset_name must be non-empty.")
    return (
        f"https://huggingface.co/datasets/{safe_name}/resolve/"
        f"refs%2Fconvert%2Fparquet/default/{split}/{filename}"
    )


def _list_parquet_urls(dataset_name: str, split: str) -> List[str]:
    if HfApi is None:
        return [_parquet_split_url(dataset_name, split, "0000.parquet")]

    api = HfApi()
    prefix = f"refs/convert/parquet/default/{split}/"
    files = api.list_repo_files(repo_id=dataset_name, repo_type="dataset")
    parquet_files = [f for f in files if f.startswith(prefix) and f.endswith(".parquet")]
    if not parquet_files:
        return [_parquet_split_url(dataset_name, split, "0000.parquet")]
    return [_parquet_split_url(dataset_name, split, f.split("/")[-1]) for f in parquet_files]


def _ensure_hf_timeouts(download_timeout: int = 60, etag_timeout: int = 60) -> None:
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(download_timeout))
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(etag_timeout))


def load_counsel_chat_split(
    split: str = "train",
    *,
    dataset_name: str = "nbertagnolli/counsel-chat",
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


def _resolve_key(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    for name in candidates:
        if name in columns:
            return name
    lower_map = {c.lower(): c for c in columns}
    for name in candidates:
        lower = name.lower()
        if lower in lower_map:
            return lower_map[lower]
    return None


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.strip()
    else:
        text = str(value).strip()
    if not text:
        return ""
    if text.lower() in {"none", "nan", "null"}:
        return ""
    return text


def _load_env(env_path: Optional[str]) -> None:
    if not env_path:
        env_path = ".env"
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _call_groq(api_key: str, model: str, messages: List[Dict[str, str]]) -> str:
    if Groq is None:
        raise ImportError("Missing dependency 'groq'. Install with: pip install groq")
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
    )
    return resp.choices[0].message.content


def _parse_json_object(content: str) -> Dict[str, object]:
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Judge response is not a JSON object.")
    snippet = content[start : end + 1]
    data = json.loads(snippet)
    if not isinstance(data, dict):
        raise ValueError("Judge response is not a JSON object.")
    return data


def _normalize_verdict(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        raise ValueError("Missing verdict.")
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1"}:
        return True
    if text in {"no", "n", "false", "0"}:
        return False
    raise ValueError(f"Unrecognized verdict: {value}")


def _detect_cutoff(text: str) -> bool:
    if not text:
        return False
    if len(text) < 8:
        return False
    if text.endswith(("...", "..")):
        return True
    return False


def judge_counsel_chat_answer(
    completion: str,
    item: Dict[str, object],
    *,
    model: str = DEFAULT_JUDGE_MODEL,
    api_key: Optional[str] = None,
    env_path: Optional[str] = None,
    sleep_s: float = 0.1,
) -> Dict[str, object]:
    _load_env(env_path)
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing GROQ_API_KEY (set it in .env or environment).")
    if model == DEFAULT_JUDGE_MODEL:
        model = os.environ.get("GROQ_MODEL") or DEFAULT_JUDGE_MODEL

    payload = {
        "question": item.get("question", ""),
        "reference_answers": item.get("reference_answers", []),
        "model_answer": completion or "",
    }
    content = _call_groq(
        api_key,
        model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    time.sleep(max(0.0, float(sleep_s)))

    try:
        parsed = _parse_json_object(content)
        verdict_raw = parsed.get("verdict")
        verdict = _normalize_verdict(verdict_raw)
        cutoff = parsed.get("cutoff")
        if not isinstance(cutoff, bool):
            cutoff = _detect_cutoff(completion)
        reason = parsed.get("reason")
        if reason is None:
            reason = ""
        return {
            "correct": bool(verdict),
            "judge_verdict": "YES" if verdict else "NO",
            "judge_cutoff": bool(cutoff),
            "judge_reason": str(reason),
            "judge_raw": parsed,
        }
    except Exception as exc:
        return {
            "correct": None,
            "judge_error": str(exc),
            "judge_raw_text": content,
        }


class CounselChatSampler:
    def __init__(
        self,
        *,
        split: str = "train",
        dataset_name: str = "nbertagnolli/counsel-chat",
        question_id_key: Optional[str] = None,
        question_text_key: Optional[str] = None,
        answer_text_key: Optional[str] = None,
        min_answers: int = 3,
        rng: Optional[random.Random] = None,
        data_files: Optional[Union[str, List[str], Dict[str, Union[str, List[str]]]]] = None,
        download_timeout: int = 60,
    ) -> None:
        self.split = split
        self.dataset_name = dataset_name
        self.question_id_key = question_id_key
        self.question_text_key = question_text_key
        self.answer_text_key = answer_text_key
        self.min_answers = int(min_answers)
        self.rng = rng or random.Random()
        self.data_files = data_files
        self.download_timeout = int(download_timeout)
        self._groups: Optional[List[Dict[str, object]]] = None

    def _build_groups(self) -> None:
        dataset = load_counsel_chat_split(
            self.split,
            dataset_name=self.dataset_name,
            data_files=self.data_files,
            download_timeout=self.download_timeout,
        )
        columns = list(getattr(dataset, "column_names", []))
        qid_key = self.question_id_key or _resolve_key(columns, QUESTION_ID_CANDIDATES)
        qtext_key = self.question_text_key or _resolve_key(columns, QUESTION_TEXT_CANDIDATES)
        ans_key = self.answer_text_key or _resolve_key(columns, ANSWER_TEXT_CANDIDATES)

        if qtext_key is None:
            raise KeyError(f"Could not detect question text column from: {columns}")
        if ans_key is None:
            raise KeyError(f"Could not detect answer text column from: {columns}")

        groups: Dict[str, Dict[str, object]] = {}
        for row in dataset:
            qtext = _clean_text(row.get(qtext_key))
            ans = _clean_text(row.get(ans_key))
            if not qtext or not ans:
                continue
            qid_val = row.get(qid_key) if qid_key else None
            qid = _clean_text(qid_val)
            key = qid if qid else qtext
            group = groups.get(key)
            if group is None:
                group = {"question_id": qid or None, "question": qtext, "answers": []}
                groups[key] = group
            group["answers"].append(ans)

        filtered = [g for g in groups.values() if len(g["answers"]) >= self.min_answers]
        if not filtered:
            raise ValueError("No questions with enough answers found.")
        self._groups = filtered

    def sample_item(self, *, max_attempts: int = 200) -> Dict[str, object]:
        if self._groups is None:
            self._build_groups()
        assert self._groups is not None
        attempts = 0
        while True:
            group = self.rng.choice(self._groups)
            question = _clean_text(group.get("question"))
            answers = [a for a in group.get("answers", []) if _clean_text(a)]
            if question and len(answers) >= self.min_answers:
                chosen = self.rng.sample(answers, k=self.min_answers)
                return {
                    "question_id": group.get("question_id"),
                    "question": question,
                    "reference_answers": chosen,
                }
            attempts += 1
            if attempts >= max_attempts:
                raise ValueError("Unable to sample a non-empty question after many attempts.")


def build_counsel_chat_prompt(item: Dict[str, object]) -> str:
    return str(item.get("question", "")).strip()


def generate_counsel_chat_item(
    *,
    split: str = "train",
    dataset_name: str = "nbertagnolli/counsel-chat",
    question_id_key: Optional[str] = None,
    question_text_key: Optional[str] = None,
    answer_text_key: Optional[str] = None,
    min_answers: int = 3,
    rng: Optional[random.Random] = None,
    data_files: Optional[Union[str, List[str], Dict[str, Union[str, List[str]]]]] = None,
    download_timeout: int = 60,
) -> Dict[str, object]:
    cache_key = (
        f"{dataset_name}::{split}::{question_id_key}::{question_text_key}"
        f"::{answer_text_key}::{min_answers}::{bool(data_files)}"
    )
    sampler = _SAMPLER_CACHE.get(cache_key)
    if sampler is None:
        sampler = CounselChatSampler(
            split=split,
            dataset_name=dataset_name,
            question_id_key=question_id_key,
            question_text_key=question_text_key,
            answer_text_key=answer_text_key,
            min_answers=min_answers,
            rng=rng,
            data_files=data_files,
            download_timeout=download_timeout,
        )
        _SAMPLER_CACHE[cache_key] = sampler
    if rng is not None:
        sampler.rng = rng
    return sampler.sample_item()


__all__ = [
    "DEFAULT_JUDGE_MODEL",
    "CounselChatSampler",
    "build_counsel_chat_prompt",
    "generate_counsel_chat_item",
    "judge_counsel_chat_answer",
    "load_counsel_chat_split",
]
