import argparse
import subprocess
import sys
from pathlib import Path


USER_MODEL = "google/gemini-2.5-flash"
MODEL_PRESETS = {
    "270m": (
        "google/gemma-3-270m-it",
        "local",
        "wellbeing_conversations_openrouter_user_local_assistant_gemma3_270m_it_run_{timestamp}.json",
    ),
    "1b": (
        "google/gemma-3-1b-it",
        "local",
        "wellbeing_conversations_openrouter_user_local_assistant_gemma3_1b_it_run_{timestamp}.json",
    ),
    "4b": (
        "google/gemma-3-4b-it",
        "openrouter",
        "wellbeing_conversations_openrouter_gemma3_4b_it_run_{timestamp}.json",
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 40x10 wellbeing conversation datasets for Gemma 3 instruct models via OpenRouter."
    )
    parser.add_argument(
        "--models",
        nargs="*",
        choices=["270m", "1b", "4b"],
        default=["270m", "1b", "4b"],
        help="Subset of Gemma 3 model sizes to run (default: all).",
    )
    parser.add_argument("--env-path", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--openrouter-url", default=None)
    parser.add_argument("--http-referer", default=None)
    parser.add_argument("--x-title", default=None)
    parser.add_argument("--timeout-s", type=int, default=None)
    parser.add_argument("--retries", type=int, default=None)
    parser.add_argument("--retry-sleep", type=float, default=None)
    parser.add_argument("--pause-s", type=float, default=None)
    parser.add_argument(
        "--assistant-use-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Applies to local assistant runs (270m/1b).",
    )
    parser.add_argument(
        "--assistant-dtype",
        default="bfloat16",
        help="Applies to local assistant runs (270m/1b).",
    )
    parser.add_argument(
        "--assistant-device-map",
        default="auto",
        help="Applies to local assistant runs (270m/1b).",
    )
    parser.add_argument(
        "--assistant-hf-token-env",
        default="HF_TOKEN",
        help="HF token env var for local assistant runs.",
    )
    return parser.parse_args()


def _run_one(
    script_path: Path,
    *,
    model_key: str,
    env_path: str | None,
    seed: int,
    overwrite: bool,
    openrouter_url: str | None,
    http_referer: str | None,
    x_title: str | None,
    timeout_s: int | None,
    retries: int | None,
    retry_sleep: float | None,
    pause_s: float | None,
    assistant_use_4bit: bool,
    assistant_dtype: str,
    assistant_device_map: str,
    assistant_hf_token_env: str,
) -> None:
    assistant_model, assistant_provider, filename_template = MODEL_PRESETS[model_key]
    output_template = f"data/{filename_template}"

    cmd = [
        sys.executable,
        str(script_path),
        "--user-model",
        USER_MODEL,
        "--assistant-model",
        assistant_model,
        "--assistant-provider",
        assistant_provider,
        "--num-conversations",
        "40",
        "--turns-per-conversation",
        "10",
        "--seed",
        str(seed),
        "--output",
        output_template,
    ]
    if env_path:
        cmd += ["--env-path", env_path]
    if overwrite:
        cmd.append("--overwrite")
    if openrouter_url:
        cmd += ["--openrouter-url", openrouter_url]
    if http_referer:
        cmd += ["--http-referer", http_referer]
    if x_title:
        cmd += ["--x-title", x_title]
    if timeout_s is not None:
        cmd += ["--timeout-s", str(timeout_s)]
    if retries is not None:
        cmd += ["--retries", str(retries)]
    if retry_sleep is not None:
        cmd += ["--retry-sleep", str(retry_sleep)]
    if pause_s is not None:
        cmd += ["--pause-s", str(pause_s)]
    if assistant_provider == "local":
        cmd += ["--assistant-dtype", str(assistant_dtype)]
        cmd += ["--assistant-device-map", str(assistant_device_map)]
        cmd += ["--assistant-hf-token-env", str(assistant_hf_token_env)]
        cmd += [
            "--assistant-use-4bit" if bool(assistant_use_4bit) else "--no-assistant-use-4bit"
        ]

    print(f"[run] gemma-3-{model_key}-it")
    print(f"      cmd: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = _parse_args()
    script_path = Path(__file__).resolve().parent / "generate_wellbeing_conversations_openrouter.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Base generator script not found: {script_path}")

    for model_key in args.models:
        _run_one(
            script_path,
            model_key=model_key,
            env_path=args.env_path,
            seed=args.seed,
            overwrite=args.overwrite,
            openrouter_url=args.openrouter_url,
            http_referer=args.http_referer,
            x_title=args.x_title,
            timeout_s=args.timeout_s,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
            pause_s=args.pause_s,
            assistant_use_4bit=bool(args.assistant_use_4bit),
            assistant_dtype=str(args.assistant_dtype),
            assistant_device_map=str(args.assistant_device_map),
            assistant_hf_token_env=str(args.assistant_hf_token_env),
        )


if __name__ == "__main__":
    main()
