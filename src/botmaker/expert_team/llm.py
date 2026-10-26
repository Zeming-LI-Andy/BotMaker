from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

_DEFAULT_MODEL_FALLBACK = "gpt-4.1"


def get_default_model() -> str:
    """Return OpenAI model name from OPENAI_MODEL, else gpt-4.1."""
    return os.environ.get("OPENAI_MODEL", _DEFAULT_MODEL_FALLBACK)


def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for expert team generation.")
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("openai package is required for expert team generation") from exc

    base_url = os.environ.get("OPENAI_API_BASE_URL")
    return OpenAI(api_key=api_key, base_url=base_url)


def _extract_json_text(raw: str) -> str:
    text = raw.strip()
    match = _JSON_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text


def parse_json_response(raw: str) -> Any:
    text = _extract_json_text(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response is not valid JSON: {raw}") from exc


def call_llm_json(
    prompt: str,
    *,
    model: str,
    system: str,
    client=None,
    temperature: float = 0.8,
) -> Any:
    if client is None:
        client = get_openai_client()

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        extra_body={"thinking": {"type": "enabled"}},
        reasoning_effort="high",
    )
    raw = resp.choices[0].message.content
    if not raw:
        raise ValueError("Empty response from OpenAI")
    return parse_json_response(raw.strip())


def load_prompt(name: str) -> str:
    from pathlib import Path

    path = Path(__file__).parent / "prompts" / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")
