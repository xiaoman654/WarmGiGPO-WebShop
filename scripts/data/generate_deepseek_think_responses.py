#!/usr/bin/env python3
"""Call a DeepSeek-compatible chat API to generate WebShop reasoning JSONL."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", flags=re.DOTALL | re.IGNORECASE)
ACTION_RE = re.compile(r"(search|click|choose|buy)\[[^\[\]]+\]", flags=re.IGNORECASE)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def load_done_sample_ids(path: Path, min_think_chars: int, require_chosen_action: bool) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = str(row.get("sample_id", "")).strip()
            if sample_id and is_valid_response(row, min_think_chars, require_chosen_action):
                done.add(sample_id)
    return done


def parse_json_text(text: str) -> dict[str, Any] | None:
    text = text.strip()
    fenced = FENCED_JSON_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    if not text.startswith("{"):
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_action(text: str) -> str:
    match = ACTION_RE.search(text)
    return match.group(0).strip() if match else ""


def normalize_response(sample_id: str, message: dict[str, Any], model: str) -> dict[str, Any]:
    content = str(message.get("content") or "")
    reasoning_content = str(message.get("reasoning_content") or "")
    finish_reason = str(message.get("_finish_reason") or "")
    parsed = parse_json_text(content) or {}
    think = parsed.get("think") or parsed.get("reasoning") or parsed.get("generated_think") or ""
    chosen_action = parsed.get("chosen_action") or parsed.get("action") or parsed.get("final_action") or ""
    if not isinstance(think, str):
        think = str(think)
    if not isinstance(chosen_action, str):
        chosen_action = str(chosen_action)
    if not think.strip() and content.strip() and not content.lstrip().startswith("{"):
        think = content.strip()
    chosen_action = extract_action(chosen_action) or extract_action(content)
    return {
        "sample_id": sample_id,
        "think": think.strip(),
        "chosen_action": chosen_action.strip(),
        "model": model,
        "raw_content": content,
        "raw_reasoning_content": reasoning_content,
        "finish_reason": finish_reason,
    }


def is_valid_response(row: dict[str, Any], min_think_chars: int, require_chosen_action: bool) -> bool:
    think = str(row.get("think") or "").strip()
    chosen_action = str(row.get("chosen_action") or "").strip()
    if len(think) < min_think_chars:
        return False
    if think.lstrip().startswith("{"):
        return False
    if require_chosen_action and not chosen_action:
        return False
    return True


def call_chat_completion(
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    thinking: str,
    reasoning_effort: str,
    response_format: str,
    user_id: str,
    timeout: int,
) -> dict[str, Any]:
    url = api_base.rstrip("/") + "/chat/completions"
    payload = build_payload(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking=thinking,
        reasoning_effort=reasoning_effort,
        response_format=response_format,
        user_id=user_id,
    )
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    parsed = json.loads(body)
    choice = parsed["choices"][0]
    message = choice.get("message", {})
    if not isinstance(message, dict):
        message = {"content": str(message)}
    message["_finish_reason"] = choice.get("finish_reason", "")
    return message


def build_payload(
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    thinking: str,
    reasoning_effort: str,
    response_format: str,
    user_id: str = "",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You generate concise WebShop agent reasoning. "
                    "Return only valid JSON with keys think and chosen_action."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if thinking != "none":
        payload["thinking"] = {"type": thinking}
    if reasoning_effort != "none":
        payload["reasoning_effort"] = reasoning_effort
    if response_format != "none":
        payload["response_format"] = {"type": response_format}
    if user_id:
        payload["user_id"] = user_id
    return payload


def normalize_api_key(value: str | None, env_name: str) -> str:
    api_key = (value or "").strip()
    if not api_key:
        raise SystemExit(f"Missing API key. Set {env_name}=...")
    try:
        api_key.encode("latin-1")
    except UnicodeEncodeError as exc:
        raise SystemExit(
            f"{env_name} contains non-ASCII/non-latin-1 characters and cannot be used in an HTTP Authorization header. "
            "Make sure you exported the real DeepSeek API key, not a placeholder such as '你的key', and remove quotes or extra text."
        ) from exc
    if any(ch.isspace() for ch in api_key):
        raise SystemExit(f"{env_name} contains whitespace. Re-export the key without spaces or newlines.")
    return api_key


def normalize_user_id(value: str | None) -> str:
    user_id = (value or "").strip()
    if not user_id:
        return ""
    if not re.fullmatch(r"[a-zA-Z0-9\-_]{1,512}", user_id):
        raise SystemExit("DEEPSEEK_USER_ID / --user-id must match [a-zA-Z0-9\\-_]+ and be at most 512 chars.")
    return user_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Request JSONL from export_webshop_think_requests.py.")
    parser.add_argument("--output", required=True, help="Generated response JSONL.")
    parser.add_argument("--api-base", default=os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com"))
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--thinking",
        choices=("enabled", "disabled", "none"),
        default=os.environ.get("DEEPSEEK_THINKING", "none"),
        help="DeepSeek thinking mode. Use 'none' to omit the thinking field from the request.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high", "none"),
        default=os.environ.get("DEEPSEEK_REASONING_EFFORT", "none"),
        help="DeepSeek reasoning effort. Use 'none' to omit the field from the request.",
    )
    parser.add_argument(
        "--response-format",
        choices=("json_object", "none"),
        default=os.environ.get("DEEPSEEK_RESPONSE_FORMAT", "json_object"),
        help="Use DeepSeek/OpenAI-compatible JSON mode by default. Set to 'none' if unsupported.",
    )
    parser.add_argument(
        "--user-id",
        default=os.environ.get("DEEPSEEK_USER_ID", ""),
        help="Optional DeepSeek user_id for account-side isolation. Must match [a-zA-Z0-9\\-_]+.",
    )
    parser.add_argument("--min-think-chars", type=int, default=20)
    parser.add_argument(
        "--allow-empty-chosen-action",
        action="store_true",
        help="Accept rows whose response has no parseable chosen_action.",
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("DEEPSEEK_WORKERS", "1")))
    parser.add_argument("--sleep", type=float, default=0.2, help="Seconds to sleep after each successful request.")
    parser.add_argument("--failure-output", default=None, help="Optional JSONL path for samples that failed all retries.")
    parser.add_argument("--invalid-output", default=None, help="Optional JSONL path for invalid API responses before retry.")
    parser.add_argument("--fail-fast", action="store_true", help="Abort when one sample fails all retries.")
    parser.add_argument("--overwrite", action="store_true", help="Ignore existing output and regenerate rows.")
    args = parser.parse_args()

    api_key = normalize_api_key(os.environ.get(args.api_key_env), args.api_key_env)
    user_id = normalize_user_id(args.user_id)

    requests = load_jsonl(Path(args.input))
    if args.start:
        requests = requests[args.start :]
    if args.max_samples is not None:
        requests = requests[: args.max_samples]

    output = Path(args.output)
    require_chosen_action = not args.allow_empty_chosen_action
    done = set() if args.overwrite else load_done_sample_ids(output, args.min_think_chars, require_chosen_action)
    if args.overwrite and output.exists():
        output.unlink()
    failure_output = Path(args.failure_output) if args.failure_output else output.with_name(output.stem + "_failures.jsonl")
    invalid_output = Path(args.invalid_output) if args.invalid_output else output.with_name(output.stem + "_invalid.jsonl")
    if args.overwrite and failure_output.exists():
        failure_output.unlink()
    if args.overwrite and invalid_output.exists():
        invalid_output.unlink()

    total = len(requests)
    skipped = 0
    pending = []
    for idx, row in enumerate(requests, 1):
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            print(f"[warn] row {idx} missing sample_id, skipped", file=sys.stderr)
            continue
        if sample_id in done:
            skipped += 1
            continue
        pending.append((idx, row))

    write_lock = threading.Lock()
    print_lock = threading.Lock()
    done_lock = threading.Lock()

    def safe_append(path: Path, row: dict[str, Any]) -> None:
        with write_lock:
            append_jsonl(path, row)

    def safe_print(message: str, *, error: bool = False) -> None:
        with print_lock:
            print(message, file=sys.stderr if error else sys.stdout, flush=True)

    def process_row(item: tuple[int, dict[str, Any]]) -> str:
        idx, row = item
        sample_id = str(row.get("sample_id", "")).strip()
        prompt = str(row.get("prompt", ""))
        last_error: Exception | None = None
        for attempt in range(1, args.retries + 1):
            try:
                message = call_chat_completion(
                    api_base=args.api_base,
                    api_key=api_key,
                    model=args.model,
                    prompt=prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    thinking=args.thinking,
                    reasoning_effort=args.reasoning_effort,
                    response_format=args.response_format,
                    user_id=user_id,
                    timeout=args.timeout,
                )
                out_row = normalize_response(sample_id, message, args.model)
                if not is_valid_response(out_row, args.min_think_chars, require_chosen_action):
                    safe_append(
                        invalid_output,
                        {
                            "sample_id": sample_id,
                            "attempt": attempt,
                            "error": "invalid_response",
                            "think_chars": len(out_row.get("think") or ""),
                            "chosen_action": out_row.get("chosen_action", ""),
                            "finish_reason": out_row.get("finish_reason", ""),
                            "raw_content_len": len(out_row.get("raw_content") or ""),
                            "raw_content": out_row.get("raw_content", ""),
                            "raw_reasoning_content_len": len(out_row.get("raw_reasoning_content") or ""),
                        },
                    )
                    raise ValueError(
                        "invalid response: "
                        f"think_chars={len(out_row.get('think') or '')}, "
                        f"chosen_action={out_row.get('chosen_action')!r}, "
                        f"finish_reason={out_row.get('finish_reason')!r}, "
                        f"raw_content_len={len(out_row.get('raw_content') or '')}, "
                        f"raw_content_prefix={str(out_row.get('raw_content') or '')[:120]!r}, "
                        f"raw_content_suffix={str(out_row.get('raw_content') or '')[-120:]!r}"
                    )
                safe_append(output, out_row)
                with done_lock:
                    done.add(sample_id)
                safe_print(f"[{idx}/{total}] generated {sample_id}")
                time.sleep(args.sleep)
                return "generated"
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
                last_error = exc
                safe_print(
                    f"[warn] {sample_id} attempt {attempt}/{args.retries} failed: {exc}",
                    error=True,
                )
                if attempt < args.retries:
                    wait_s = min(30.0, 2.0**attempt)
                    safe_print(f"[warn] {sample_id} sleep {wait_s:.1f}s before retry", error=True)
                    time.sleep(wait_s)

        failure_row = {
            "sample_id": sample_id,
            "error": str(last_error),
            "model": args.model,
            "request_index": idx,
        }
        safe_append(failure_output, failure_row)
        safe_print(
            f"[error] failed {sample_id} after {args.retries} attempts; recorded in {failure_output}",
            error=True,
        )
        if args.fail_fast:
            raise RuntimeError(f"failed to generate sample_id={sample_id}: {last_error}") from last_error
        return "failed"

    generated = 0
    failed = 0
    workers = max(1, args.workers)
    if workers == 1:
        for item in pending:
            result = process_row(item)
            generated += result == "generated"
            failed += result == "failed"
    else:
        safe_print(f"running with workers={workers}, pending={len(pending)}, skipped_existing={skipped}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_row, item) for item in pending]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                generated += result == "generated"
                failed += result == "failed"

    print(
        json.dumps(
            {
                "input": args.input,
                "output": args.output,
                "model": args.model,
                "thinking": args.thinking,
                "reasoning_effort": args.reasoning_effort,
                "response_format": args.response_format,
                "user_id": user_id,
                "total_requested": total,
                "pending": len(pending),
                "workers": workers,
                "generated": generated,
                "skipped_existing": skipped,
                "failed": failed,
                "output_rows_seen": len(done),
                "failure_output": str(failure_output),
                "invalid_output": str(invalid_output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
