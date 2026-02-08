import argparse
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import requests

from generate_wellbeing_conversations import (
    ASSISTANT_SYSTEM_PROMPT,
    TOPICS,
    USER_SYSTEM_PROMPT_TEMPLATE,
)

DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_USER_MODEL = "google/gemini-2.5-flash"
DEFAULT_ASSISTANT_MODEL = "meta-llama/llama-3.2-3b-instruct"

DEFAULT_OUTPUT_TEMPLATE = "data/wellbeing_conversations_openrouter_{timestamp}.json"
DEFAULT_NUM_CONVERSATIONS = 20
DEFAULT_TURNS_PER_CONVERSATION = 10

DEFAULT_USER_TEMPERATURE = 0.7
DEFAULT_ASSISTANT_TEMPERATURE = 0.8
DEFAULT_USER_TOP_P = 0.95
DEFAULT_ASSISTANT_TOP_P = 0.9
DEFAULT_USER_MAX_TOKENS = 256
DEFAULT_ASSISTANT_MAX_TOKENS = 256
DEFAULT_TIMEOUT_S = 60
DEFAULT_RETRIES = 3
DEFAULT_RETRY_SLEEP = 1.5

TIMESTAMP_RE = re.compile(r"\d{8}_\d{6}$")


def _load_env(env_path: Optional[Path] = None) -> None:
    path = env_path or Path(".env")
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _needs_timestamp(path: Path) -> bool:
    if path.suffix != ".json":
        return False
    if "data" not in path.parts:
        return False
    return not bool(TIMESTAMP_RE.search(path.stem))


def _resolve_output_path(script_dir: Path, raw_output: str) -> Path:
    timestamp = _now_tag()
    if "{timestamp}" in raw_output:
        resolved = raw_output.format(timestamp=timestamp)
        return (script_dir / resolved).resolve()

    candidate = Path(raw_output)
    if candidate.suffix == "":
        resolved = candidate / f"wellbeing_conversations_openrouter_{timestamp}.json"
        return (script_dir / resolved).resolve()

    if _needs_timestamp(candidate):
        stamped = candidate.with_name(f"{candidate.stem}_{timestamp}{candidate.suffix}")
        return (script_dir / stamped).resolve()

    return (script_dir / candidate).resolve()


def _build_headers(api_key: str, http_referer: str, x_title: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": http_referer,
        "X-Title": x_title,
    }


def _normalize_content(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        chunks: List[str] = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks).strip()
    return ""


def _call_openrouter(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    *,
    openrouter_url: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_s: int,
    retries: int,
    retry_sleep: float,
    http_referer: str,
    x_title: str,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    headers = _build_headers(api_key, http_referer=http_referer, x_title=x_title)

    attempt = 0
    while True:
        try:
            response = requests.post(
                openrouter_url,
                headers=headers,
                json=payload,
                timeout=timeout_s,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return _normalize_content(content)
        except Exception as exc:
            attempt += 1
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code in (401, 403):
                raise RuntimeError(
                    "OpenRouter authentication failed (401/403). "
                    "Check OPENROUTER_API_KEY and model access."
                ) from exc
            if attempt >= retries:
                raise RuntimeError(
                    f"OpenRouter call failed after {retries} attempts: {exc}"
                ) from exc
            time.sleep(retry_sleep)


def _invert_roles(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    inverted: List[Dict[str, str]] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            inverted.append({"role": "assistant", "content": content})
        elif role == "assistant":
            inverted.append({"role": "user", "content": content})
    return inverted


def _make_user_system_prompt(topic: str) -> str:
    return USER_SYSTEM_PROMPT_TEMPLATE.format(topic=topic)


def _ensure_nonempty(text: str, fallback: str) -> str:
    cleaned = (text or "").strip()
    return cleaned if cleaned else fallback


def _generate_user_message(
    api_key: str,
    model: str,
    conversation: List[Dict[str, str]],
    *,
    topic: str,
    openrouter_url: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_s: int,
    retries: int,
    retry_sleep: float,
    http_referer: str,
    x_title: str,
) -> str:
    messages = [{"role": "system", "content": _make_user_system_prompt(topic)}]
    messages.extend(_invert_roles(conversation))
    return _call_openrouter(
        api_key,
        model,
        messages,
        openrouter_url=openrouter_url,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
        retries=retries,
        retry_sleep=retry_sleep,
        http_referer=http_referer,
        x_title=x_title,
    )


def _generate_assistant_message(
    api_key: str,
    model: str,
    conversation: List[Dict[str, str]],
    *,
    openrouter_url: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_s: int,
    retries: int,
    retry_sleep: float,
    http_referer: str,
    x_title: str,
) -> str:
    messages = [{"role": "system", "content": ASSISTANT_SYSTEM_PROMPT}]
    messages.extend(conversation)
    return _call_openrouter(
        api_key,
        model,
        messages,
        openrouter_url=openrouter_url,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
        retries=retries,
        retry_sleep=retry_sleep,
        http_referer=http_referer,
        x_title=x_title,
    )


def _generate_conversation(
    api_key: str,
    *,
    topic: str,
    user_model: str,
    assistant_model: str,
    turns_per_conversation: int,
    user_temperature: float,
    assistant_temperature: float,
    user_top_p: float,
    assistant_top_p: float,
    user_max_tokens: int,
    assistant_max_tokens: int,
    timeout_s: int,
    retries: int,
    retry_sleep: float,
    openrouter_url: str,
    http_referer: str,
    x_title: str,
    pause_s: float,
    conversation_label: str,
) -> List[Dict[str, str]]:
    conversation: List[Dict[str, str]] = []
    for turn_index in range(1, turns_per_conversation + 1):
        print(
            f"{conversation_label} turn {turn_index}/{turns_per_conversation}: user",
            flush=True,
        )
        user_message = _generate_user_message(
            api_key,
            user_model,
            conversation,
            topic=topic,
            openrouter_url=openrouter_url,
            temperature=user_temperature,
            top_p=user_top_p,
            max_tokens=user_max_tokens,
            timeout_s=timeout_s,
            retries=retries,
            retry_sleep=retry_sleep,
            http_referer=http_referer,
            x_title=x_title,
        )
        user_message = _ensure_nonempty(user_message, "Can we keep exploring this?")
        conversation.append({"role": "user", "content": user_message})
        time.sleep(pause_s)

        print(
            f"{conversation_label} turn {turn_index}/{turns_per_conversation}: assistant",
            flush=True,
        )
        assistant_message = _generate_assistant_message(
            api_key,
            assistant_model,
            conversation,
            openrouter_url=openrouter_url,
            temperature=assistant_temperature,
            top_p=assistant_top_p,
            max_tokens=assistant_max_tokens,
            timeout_s=timeout_s,
            retries=retries,
            retry_sleep=retry_sleep,
            http_referer=http_referer,
            x_title=x_title,
        )
        assistant_message = _ensure_nonempty(assistant_message, "Tell me more about that.")
        conversation.append({"role": "assistant", "content": assistant_message})
        time.sleep(pause_s)

    return conversation


def _build_metadata(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": "openrouter",
        "openrouter_url": args.openrouter_url,
        "user_model": args.user_model,
        "assistant_model": args.assistant_model,
        "num_conversations": args.num_conversations,
        "turns_per_conversation": args.turns_per_conversation,
        "user_temperature": args.user_temperature,
        "assistant_temperature": args.assistant_temperature,
        "user_top_p": args.user_top_p,
        "assistant_top_p": args.assistant_top_p,
        "user_max_tokens": args.user_max_tokens,
        "assistant_max_tokens": args.assistant_max_tokens,
        "timeout_s": args.timeout_s,
        "retries": args.retries,
        "retry_sleep": args.retry_sleep,
        "pause_s": args.pause_s,
        "system_prompt_user_template": USER_SYSTEM_PROMPT_TEMPLATE,
        "system_prompt_assistant": ASSISTANT_SYSTEM_PROMPT,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate wellbeing conversations using two OpenRouter models."
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT_TEMPLATE)
    parser.add_argument("--env-path", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--shuffle-topics", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--openrouter-url", default=DEFAULT_OPENROUTER_URL)
    parser.add_argument("--http-referer", default="http://localhost")
    parser.add_argument("--x-title", default="wellbeing-openrouter-chat")

    parser.add_argument("--user-model", default=DEFAULT_USER_MODEL)
    parser.add_argument("--assistant-model", default=DEFAULT_ASSISTANT_MODEL)
    parser.add_argument(
        "--num-conversations",
        "--max-conversations",
        dest="num_conversations",
        type=int,
        default=DEFAULT_NUM_CONVERSATIONS,
    )
    parser.add_argument(
        "--turns-per-conversation",
        "--max-assistant-turns",
        dest="turns_per_conversation",
        type=int,
        default=DEFAULT_TURNS_PER_CONVERSATION,
    )

    parser.add_argument("--user-temperature", type=float, default=DEFAULT_USER_TEMPERATURE)
    parser.add_argument(
        "--assistant-temperature", type=float, default=DEFAULT_ASSISTANT_TEMPERATURE
    )
    parser.add_argument("--user-top-p", type=float, default=DEFAULT_USER_TOP_P)
    parser.add_argument("--assistant-top-p", type=float, default=DEFAULT_ASSISTANT_TOP_P)
    parser.add_argument("--user-max-tokens", type=int, default=DEFAULT_USER_MAX_TOKENS)
    parser.add_argument("--assistant-max-tokens", type=int, default=DEFAULT_ASSISTANT_MAX_TOKENS)
    parser.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--retry-sleep", type=float, default=DEFAULT_RETRY_SLEEP)
    parser.add_argument("--pause-s", type=float, default=0.4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _load_env(Path(args.env_path) if args.env_path else None)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Missing OPENROUTER_API_KEY (set it in .env or environment)."
        )
    if args.num_conversations <= 0:
        raise ValueError("--num-conversations must be positive.")
    if args.turns_per_conversation <= 0:
        raise ValueError("--turns-per-conversation must be positive.")
    if args.num_conversations > len(TOPICS):
        raise ValueError(
            f"--num-conversations cannot exceed {len(TOPICS)} (got {args.num_conversations})."
        )
    if args.retries <= 0:
        raise ValueError("--retries must be positive.")
    if args.timeout_s <= 0:
        raise ValueError("--timeout-s must be positive.")

    topics = TOPICS.copy()
    random.seed(args.seed)
    if args.shuffle_topics:
        random.shuffle(topics)
    topics = topics[: args.num_conversations]

    script_dir = Path(__file__).resolve().parent
    output_path = _resolve_output_path(script_dir, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace."
        )

    conversations: List[Dict[str, object]] = []
    for idx, topic in enumerate(topics, start=1):
        conversation_label = f"[conversation {idx}/{len(topics)} | {topic['id']}: {topic['title']}]"
        print(f"{conversation_label} start", flush=True)
        topic_text = topic["prompt"]
        convo = _generate_conversation(
            api_key,
            topic=topic_text,
            user_model=args.user_model,
            assistant_model=args.assistant_model,
            turns_per_conversation=args.turns_per_conversation,
            user_temperature=args.user_temperature,
            assistant_temperature=args.assistant_temperature,
            user_top_p=args.user_top_p,
            assistant_top_p=args.assistant_top_p,
            user_max_tokens=args.user_max_tokens,
            assistant_max_tokens=args.assistant_max_tokens,
            timeout_s=args.timeout_s,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
            openrouter_url=args.openrouter_url,
            http_referer=args.http_referer,
            x_title=args.x_title,
            pause_s=args.pause_s,
            conversation_label=conversation_label,
        )
        conversations.append(
            {
                "topic_id": topic["id"],
                "topic_title": topic["title"],
                "topic_prompt": topic_text,
                "user_system_prompt": _make_user_system_prompt(topic_text),
                "assistant_system_prompt": ASSISTANT_SYSTEM_PROMPT,
                "messages": convo,
            }
        )

    payload = {
        "metadata": _build_metadata(args),
        "topics": topics,
        "conversations": conversations,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote dataset to {output_path}")


if __name__ == "__main__":
    main()
