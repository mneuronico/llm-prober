import argparse
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

try:
    from groq import Groq
except Exception:
    Groq = None
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except Exception:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None

DEFAULT_USER_MODEL = "openai/gpt-oss-120b"
DEFAULT_ASSISTANT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

DEFAULT_ASSISTANT_PROVIDER = "local"
DEFAULT_ASSISTANT_USE_4BIT = True
DEFAULT_ASSISTANT_DTYPE = "bfloat16"
DEFAULT_ASSISTANT_DEVICE_MAP = "auto"
DEFAULT_ASSISTANT_TOP_P = 0.9

DEFAULT_MAX_ASSISTANT_TURNS = 10
DEFAULT_MAX_CONVERSATIONS = 20
DEFAULT_USER_TEMPERATURE = 0.8
DEFAULT_ASSISTANT_TEMPERATURE = 0.8
DEFAULT_USER_MAX_TOKENS = 256
DEFAULT_ASSISTANT_MAX_TOKENS = 256

DEFAULT_OUTPUT_TEMPLATE = "data/wellbeing_conversations_{timestamp}.json"

TIMESTAMP_RE = re.compile(r"\d{8}_\d{6}$")

USER_SYSTEM_PROMPT_TEMPLATE = (
    "You are simulating a human user chatting with an AI assistant. Stay on the exact topic below and keep the conversation going for many turns. Be clear and specific in your questions. Ask follow-up questions, share details, and react naturally. But each message you send should be brief (1-4 sentences), and you should not ask very demanding questions. Be varied and specific (do not just ask the assistant to keep talking or exploring, give it some relevant input). Do not mention that you are an AI or that this is a simulation. Do not ask questions about the assistant. The assistant has no internet or access to recent information. Never end the conversation. Write in English only. Remember, you are roleplaying as the USER, you are NOT the assistant.\n\nTopic: {topic}"
)

ASSISTANT_SYSTEM_PROMPT = (
    "You are a helpful assistant. You give very brief responses to the users's questions (max: 5 sentences)."
)

TOPICS: List[Dict[str, str]] = [
    {
        "id": "topic_01",
        "title": "Minimalist move planning",
        "prompt": "Planning a minimalist move to a new city on a tight budget.",
    },
    {
        "id": "topic_02",
        "title": "Vegetarian meal prep",
        "prompt": "Designing a weekly meal-prep plan for a vegetarian athlete.",
    },
    {
        "id": "topic_03",
        "title": "Jazz piano practice",
        "prompt": "Building a beginner jazz piano practice routine and improvisation plan.",
    },
    {
        "id": "topic_04",
        "title": "Raise negotiation",
        "prompt": "Preparing to negotiate a raise and drafting talking points.",
    },
    {
        "id": "topic_05",
        "title": "Friendship boundaries",
        "prompt": "Managing a long-distance friendship and communication boundaries.",
    },
    {
        "id": "topic_06",
        "title": "Solarpunk worldbuilding",
        "prompt": "Worldbuilding for a solarpunk short story.",
    },
    {
        "id": "topic_07",
        "title": "Sleep hygiene",
        "prompt": "Understanding sleep hygiene and circadian rhythms to feel more rested.",
    },
    {
        "id": "topic_08",
        "title": "First 10K training",
        "prompt": "Creating a training plan for a first 10K run.",
    },
    {
        "id": "topic_09",
        "title": "DIY faucet repair",
        "prompt": "Fixing a dripping faucet and choosing the right tools.",
    },
    {
        "id": "topic_10",
        "title": "Japan trip outline",
        "prompt": "Planning a 5-day trip to Japan with a cultural focus (no internet).",
    },
    {
        "id": "topic_11",
        "title": "Spanish study plan",
        "prompt": "Reaching B1 in Spanish with daily micro-habits.",
    },
    {
        "id": "topic_12",
        "title": "Board game strategy",
        "prompt": "Strategy discussion for a complex board game like Terraforming Mars.",
    },
    {
        "id": "topic_13",
        "title": "AI hiring ethics",
        "prompt": "Debating ethics and bias in AI-assisted hiring.",
    },
    {
        "id": "topic_14",
        "title": "Startup validation",
        "prompt": "Validating a SaaS idea for freelancers.",
    },
    {
        "id": "topic_15",
        "title": "Presentation anxiety",
        "prompt": "Coping with anxiety before a big presentation (therapy-style chat).",
    },
    {
        "id": "topic_16",
        "title": "Headache triggers",
        "prompt": "Discussing common headache triggers and when to seek medical care.",
    },
    {
        "id": "topic_17",
        "title": "Toddler tantrums",
        "prompt": "Handling toddler tantrums with calm, consistent parenting strategies.",
    },
    {
        "id": "topic_18",
        "title": "Murder mystery party",
        "prompt": "Planning a themed murder mystery party for friends.",
    },
    {
        "id": "topic_19",
        "title": "Home backup workflow",
        "prompt": "Setting up a home backup system and file organization workflow.",
    },
    {
        "id": "topic_20",
        "title": "Free will debate",
        "prompt": "Philosophical discussion about free will vs determinism in daily life.",
    },
    {
        "id": "topic_21",
        "title": "Cheap family dinners",
        "prompt": "Planning budget-friendly weeknight dinners for a family with picky eaters.",
    },
    {
        "id": "topic_22",
        "title": "Birthday surprise planning",
        "prompt": "Organizing a meaningful surprise birthday celebration on a small budget.",
    },
    {
        "id": "topic_23",
        "title": "Noisy neighbor stress",
        "prompt": "Handling stress from noisy neighbors while keeping things polite and practical.",
    },
    {
        "id": "topic_24",
        "title": "Burnout recovery habits",
        "prompt": "Recovering from burnout and rebuilding a sustainable work-life routine.",
    },
    {
        "id": "topic_25",
        "title": "First-time dog adoption",
        "prompt": "Preparing for first-time dog adoption, including schedule, costs, and training basics.",
    },
    {
        "id": "topic_26",
        "title": "Used car decision",
        "prompt": "Choosing a reliable used car and avoiding common buying mistakes.",
    },
    {
        "id": "topic_27",
        "title": "Wedding guest budget",
        "prompt": "Getting ready for multiple weddings without overspending on outfits and travel.",
    },
    {
        "id": "topic_28",
        "title": "Morning routine reset",
        "prompt": "Building a realistic morning routine to feel less rushed before work.",
    },
    {
        "id": "topic_29",
        "title": "Sentimental decluttering",
        "prompt": "Decluttering sentimental items without feeling guilty or losing meaningful memories.",
    },
    {
        "id": "topic_30",
        "title": "Job offer comparison",
        "prompt": "Comparing two job offers with different salary, commute, and growth potential.",
    },
    {
        "id": "topic_31",
        "title": "Post-breakup routine",
        "prompt": "Creating healthy routines and boundaries after a difficult breakup.",
    },
    {
        "id": "topic_32",
        "title": "Aging parent support",
        "prompt": "Supporting an aging parent from another city while balancing your own responsibilities.",
    },
    {
        "id": "topic_33",
        "title": "Kids screen-time plan",
        "prompt": "Setting fair screen-time rules for kids without constant arguments.",
    },
    {
        "id": "topic_34",
        "title": "Balcony herb garden",
        "prompt": "Starting a small balcony herb garden for cooking in a rented apartment.",
    },
    {
        "id": "topic_35",
        "title": "First camping weekend",
        "prompt": "Planning a first weekend camping trip with simple gear and low stress.",
    },
    {
        "id": "topic_36",
        "title": "Networking anxiety",
        "prompt": "Managing social anxiety before professional networking events.",
    },
    {
        "id": "topic_37",
        "title": "Photo organization",
        "prompt": "Organizing years of digital photos into a simple system you can maintain.",
    },
    {
        "id": "topic_38",
        "title": "Lower grocery costs",
        "prompt": "Cutting grocery spending while still eating balanced meals each week.",
    },
    {
        "id": "topic_39",
        "title": "Making local friends",
        "prompt": "Making new friends after moving to a new city as an adult.",
    },
    {
        "id": "topic_40",
        "title": "Rainy weekend ideas",
        "prompt": "Planning a cozy rainy weekend with your partner without spending much money.",
    },
]


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


def _torch_dtype_from_str(value: str):
    v = (value or "").lower()
    if v in ("bf16", "bfloat16"):
        return torch.bfloat16
    if v in ("fp16", "float16", "half"):
        return torch.float16
    if v in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(f"Unsupported dtype string: {value}")


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
        resolved = candidate / f"wellbeing_conversations_{timestamp}.json"
        return (script_dir / resolved).resolve()

    if _needs_timestamp(candidate):
        stamped = candidate.with_name(f"{candidate.stem}_{timestamp}{candidate.suffix}")
        return (script_dir / stamped).resolve()

    return (script_dir / candidate).resolve()


def _call_groq(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    retries: int = 3,
    retry_sleep: float = 1.5,
) -> str:
    if Groq is None:
        raise ImportError("Missing dependency 'groq'. Install with: pip install groq")
    client = Groq(api_key=api_key)
    attempt = 0
    while True:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content
            return content.strip() if isinstance(content, str) else ""
        except Exception as exc:
            attempt += 1
            msg = str(exc)
            if "401" in msg or "403" in msg:
                raise RuntimeError(
                    f"Groq API error: {msg} (check GROQ_API_KEY and model access)"
                ) from exc
            if attempt >= retries:
                raise RuntimeError(f"Groq API error after {retries} attempts: {msg}") from exc
            time.sleep(retry_sleep)


def _load_local_assistant(
    model_id: str,
    *,
    use_4bit: bool,
    dtype: str,
    device_map: str,
    hf_token_env: str,
):
    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        raise ImportError(
            "Missing dependency 'transformers' or 'torch'. "
            "Install with: pip install torch transformers bitsandbytes"
        )
    if use_4bit and BitsAndBytesConfig is None:
        raise ImportError("Missing dependency 'bitsandbytes' for 4-bit loading.")
    hf_token = os.environ.get(hf_token_env, None)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True) if use_4bit else None
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        device_map=device_map,
        torch_dtype=_torch_dtype_from_str(dtype),
        quantization_config=quant if use_4bit else None,
    )
    model.eval()
    return tokenizer, model


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


def _generate_user_message(
    api_key: str,
    model: str,
    conversation: List[Dict[str, str]],
    *,
    topic: str,
    temperature: float,
    max_tokens: int,
) -> str:
    system_prompt = _make_user_system_prompt(topic)
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(_invert_roles(conversation))
    return _call_groq(
        api_key,
        model,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _generate_assistant_message(
    api_key: str,
    model: str,
    conversation: List[Dict[str, str]],
    *,
    provider: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    local_assistant: Optional[Dict[str, object]],
) -> str:
    messages = [{"role": "system", "content": ASSISTANT_SYSTEM_PROMPT}]
    messages.extend(conversation)
    if provider == "groq":
        return _call_groq(
            api_key,
            model,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if local_assistant is None:
        raise RuntimeError("Local assistant model is not initialized.")
    tokenizer = local_assistant["tokenizer"]
    model_obj = local_assistant["model"]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    input_ids = input_ids.to(model_obj.device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    do_sample = temperature > 0
    with torch.inference_mode():
        gen_ids = model_obj.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = gen_ids[0][input_ids.shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _ensure_nonempty(text: str, fallback: str) -> str:
    cleaned = (text or "").strip()
    return cleaned if cleaned else fallback


def _generate_conversation(
    api_key: str,
    *,
    topic: str,
    user_model: str,
    assistant_model: str,
    assistant_provider: str,
    conversation_label: str,
    max_assistant_turns: int,
    user_temperature: float,
    assistant_temperature: float,
    user_max_tokens: int,
    assistant_max_tokens: int,
    assistant_top_p: float,
    local_assistant: Optional[Dict[str, object]],
    pause_s: float,
) -> List[Dict[str, str]]:
    conversation: List[Dict[str, str]] = []
    assistant_turns = 0
    while assistant_turns < max_assistant_turns:
        print(
            f"{conversation_label} turn {assistant_turns + 1}/{max_assistant_turns}: user",
            flush=True,
        )
        user_message = _generate_user_message(
            api_key,
            user_model,
            conversation,
            topic=topic,
            temperature=user_temperature,
            max_tokens=user_max_tokens,
        )
        user_message = _ensure_nonempty(user_message, "Can we keep exploring this?")
        conversation.append({"role": "user", "content": user_message})
        time.sleep(pause_s)

        print(
            f"{conversation_label} turn {assistant_turns + 1}/{max_assistant_turns}: assistant",
            flush=True,
        )
        assistant_message = _generate_assistant_message(
            api_key,
            assistant_model,
            conversation,
            provider=assistant_provider,
            temperature=assistant_temperature,
            max_tokens=assistant_max_tokens,
            top_p=assistant_top_p,
            local_assistant=local_assistant,
        )
        assistant_message = _ensure_nonempty(assistant_message, "Tell me more about that.")
        conversation.append({"role": "assistant", "content": assistant_message})
        assistant_turns += 1
        time.sleep(pause_s)
    return conversation


def _build_metadata(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "user_model": args.user_model,
        "assistant_model": args.assistant_model,
        "assistant_provider": args.assistant_provider,
        "assistant_use_4bit": args.assistant_use_4bit,
        "assistant_dtype": args.assistant_dtype,
        "assistant_device_map": args.assistant_device_map,
        "assistant_top_p": args.assistant_top_p,
        "max_assistant_turns": args.max_assistant_turns,
        "max_conversations": args.max_conversations,
        "user_temperature": args.user_temperature,
        "assistant_temperature": args.assistant_temperature,
        "user_max_tokens": args.user_max_tokens,
        "assistant_max_tokens": args.assistant_max_tokens,
        "pause_s": args.pause_s,
        "system_prompt_user_template": USER_SYSTEM_PROMPT_TEMPLATE,
        "system_prompt_assistant": ASSISTANT_SYSTEM_PROMPT,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate long multi-topic conversations between a user simulator and an assistant."
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT_TEMPLATE)
    parser.add_argument("--env-path", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--shuffle-topics", action="store_true")
    parser.add_argument("--user-model", default=None)
    parser.add_argument("--assistant-model", default=None)
    parser.add_argument(
        "--assistant-provider",
        choices=["local", "groq"],
        default=DEFAULT_ASSISTANT_PROVIDER,
    )
    parser.add_argument(
        "--assistant-use-4bit",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ASSISTANT_USE_4BIT,
    )
    parser.add_argument("--assistant-dtype", default=DEFAULT_ASSISTANT_DTYPE)
    parser.add_argument("--assistant-device-map", default=DEFAULT_ASSISTANT_DEVICE_MAP)
    parser.add_argument("--assistant-top-p", type=float, default=DEFAULT_ASSISTANT_TOP_P)
    parser.add_argument("--max-assistant-turns", type=int, default=DEFAULT_MAX_ASSISTANT_TURNS)
    parser.add_argument("--max-conversations", type=int, default=DEFAULT_MAX_CONVERSATIONS)
    parser.add_argument("--user-temperature", type=float, default=DEFAULT_USER_TEMPERATURE)
    parser.add_argument("--assistant-temperature", type=float, default=DEFAULT_ASSISTANT_TEMPERATURE)
    parser.add_argument("--user-max-tokens", type=int, default=DEFAULT_USER_MAX_TOKENS)
    parser.add_argument("--assistant-max-tokens", type=int, default=DEFAULT_ASSISTANT_MAX_TOKENS)
    parser.add_argument("--pause-s", type=float, default=0.4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _load_env(Path(args.env_path) if args.env_path else None)
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing GROQ_API_KEY (set it in .env or environment).")

    user_model = args.user_model or os.environ.get("GROQ_USER_MODEL") or os.environ.get(
        "GROQ_MODEL"
    ) or DEFAULT_USER_MODEL
    assistant_model = (
        args.assistant_model
        or os.environ.get("GROQ_ASSISTANT_MODEL")
        or os.environ.get("GROQ_MODEL")
        or DEFAULT_ASSISTANT_MODEL
    )
    args.user_model = user_model
    args.assistant_model = assistant_model

    topics = TOPICS.copy()
    random.seed(args.seed)
    if args.shuffle_topics:
        random.shuffle(topics)
    if args.max_conversations <= 0:
        raise ValueError("--max-conversations must be positive.")
    if args.max_conversations > len(topics):
        raise ValueError(
            f"--max-conversations cannot exceed {len(topics)} (got {args.max_conversations})."
        )
    topics = topics[: args.max_conversations]

    local_assistant: Optional[Dict[str, object]] = None
    if args.assistant_provider == "local":
        tokenizer, model_obj = _load_local_assistant(
            assistant_model,
            use_4bit=args.assistant_use_4bit,
            dtype=args.assistant_dtype,
            device_map=args.assistant_device_map,
            hf_token_env="HF_TOKEN",
        )
        local_assistant = {"tokenizer": tokenizer, "model": model_obj}

    script_dir = Path(__file__).resolve().parent
    output_path = _resolve_output_path(script_dir, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace."
        )

    metadata = _build_metadata(args)
    conversations: List[Dict[str, object]] = []
    for idx, topic in enumerate(topics, start=1):
        conversation_label = f"[conversation {idx}/{len(topics)} | {topic['id']}: {topic['title']}]"
        print(f"{conversation_label} start", flush=True)
        topic_text = topic["prompt"]
        convo = _generate_conversation(
            api_key,
            topic=topic_text,
            user_model=user_model,
            assistant_model=assistant_model,
            assistant_provider=args.assistant_provider,
            conversation_label=conversation_label,
            max_assistant_turns=args.max_assistant_turns,
            user_temperature=args.user_temperature,
            assistant_temperature=args.assistant_temperature,
            user_max_tokens=args.user_max_tokens,
            assistant_max_tokens=args.assistant_max_tokens,
            assistant_top_p=args.assistant_top_p,
            local_assistant=local_assistant,
            pause_s=args.pause_s,
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
        "metadata": metadata,
        "topics": topics,
        "conversations": conversations,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote dataset to {output_path}")


if __name__ == "__main__":
    main()
