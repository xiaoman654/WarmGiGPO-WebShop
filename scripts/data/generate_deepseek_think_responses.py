#!/usr/bin/env python3
"""Call a DeepSeek-compatible chat API to generate WebShop reasoning JSONL."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
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
    timeout: int,
) -> dict[str, Any]:
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
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
    message = parsed["choices"][0]["message"]
    return message if isinstance(message, dict) else {"content": str(message)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Request JSONL from export_webshop_think_requests.py.")
    parser.add_argument("--output", required=True, help="Generated response JSONL.")
    parser.add_argument("--api-base", default=os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com"))
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=1024)
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
    parser.add_argument("--min-think-chars", type=int, default=20)
    parser.add_argument(
        "--allow-empty-chosen-action",
        action="store_true",
        help="Accept rows whose response has no parseable chosen_action.",
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.2, help="Seconds to sleep after each successful request.")
    parser.add_argument("--overwrite", action="store_true", help="Ignore existing output and regenerate rows.")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key. Set {args.api_key_env}=...")

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

    total = len(requests)
    generated = 0
    skipped = 0
    for idx, row in enumerate(requests, 1):
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            print(f"[warn] row {idx} missing sample_id, skipped", file=sys.stderr)
            continue
        if sample_id in done:
            skipped += 1
            continue

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
                    timeout=args.timeout,
                )
                out_row = normalize_response(sample_id, message, args.model)
                if not is_valid_response(out_row, args.min_think_chars, require_chosen_action):
                    raise ValueError(
                        "invalid response: "
                        f"think_chars={len(out_row.get('think') or '')}, "
                        f"chosen_action={out_row.get('chosen_action')!r}, "
                        f"raw_content_prefix={str(out_row.get('raw_content') or '')[:80]!r}"
                    )
                append_jsonl(output, out_row)
                done.add(sample_id)
                generated += 1
                print(f"[{idx}/{total}] generated {sample_id}", flush=True)
                time.sleep(args.sleep)
                break
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
                last_error = exc
                wait_s = min(30.0, 2.0**attempt)
                print(
                    f"[warn] {sample_id} attempt {attempt}/{args.retries} failed: {exc}; sleep {wait_s:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(wait_s)
        else:
            raise RuntimeError(f"failed to generate sample_id={sample_id}: {last_error}") from last_error

    print(
        json.dumps(
            {
                "input": args.input,
                "output": args.output,
                "model": args.model,
                "thinking": args.thinking,
                "reasoning_effort": args.reasoning_effort,
                "total_requested": total,
                "generated": generated,
                "skipped_existing": skipped,
                "output_rows_seen": len(done),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
