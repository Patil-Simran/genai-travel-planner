"""Travel planner: LangGraph + Hugging Face Inference."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import TypedDict

from dotenv import dotenv_values, load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError, InferenceTimeoutError
from huggingface_hub.utils import get_token as hf_get_token
from langgraph.graph import END, StateGraph

_APP_DIR = Path(__file__).resolve().parent
load_dotenv(_APP_DIR / ".env", override=True)

log = logging.getLogger(__name__)

MODEL_FALLBACK_CHAIN: tuple[str, ...] = (
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "Qwen/Qwen2.5-7B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
)
DEFAULT_MODEL_ID = MODEL_FALLBACK_CHAIN[0]


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _normalize_hf_token_string(s: str) -> str:
    t = (s or "").strip()
    if t.startswith("\ufeff"):
        t = t[1:].strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "'\"":
        t = t[1:-1].strip()
    return t


def _require_hf_token() -> str:
    token = _normalize_hf_token_string(os.getenv("HF_TOKEN") or "")
    if not token:
        token = _normalize_hf_token_string(hf_get_token() or "")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. Create a free token at "
            "https://huggingface.co/settings/tokens and add it to your environment or .env file."
        )
    return token


def _inference_provider_name(token: str) -> str:
    env_file = _APP_DIR / ".env"
    if env_file.is_file():
        raw = dotenv_values(env_file).get("HF_INFERENCE_PROVIDER")
        if raw is not None and str(raw).strip():
            v = str(raw).strip().lower()
            if v in ("auto", "hf-inference"):
                return v
        return "auto" if token.startswith("hf_") else "hf-inference"
    env = (os.getenv("HF_INFERENCE_PROVIDER") or "").strip().lower()
    if env in ("auto", "hf-inference"):
        return env
    return "auto" if token.startswith("hf_") else "hf-inference"


def _models_to_try() -> list[str]:
    env = (os.getenv("HF_MODEL_ID") or "").strip()
    seen: set[str] = set()
    out: list[str] = []
    if env:
        out.append(env)
        seen.add(env)
    for m in MODEL_FALLBACK_CHAIN:
        if m not in seen:
            out.append(m)
            seen.add(m)
    return out


def _truncate(s: str, max_len: int = 420) -> str:
    s = s.strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _friendly_inference_error(exc: BaseException) -> str:
    if isinstance(exc, InferenceTimeoutError):
        return (
            "The inference request timed out (model may be cold or busy). "
            "Wait a moment and retry, or set HF_MODEL_ID to a smaller/faster model."
        )
    if isinstance(exc, HfHubHTTPError):
        parts: list[str] = [_truncate(str(exc))]
        if exc.response.status_code == 401 and "401" not in parts[0]:
            parts.append("Regenerate HF_TOKEN at huggingface.co/settings/tokens (Inference access).")
        if exc.server_message:
            sm = _truncate(exc.server_message, 280)
            if sm and sm not in parts[0]:
                parts.append(f"Server: {sm}")
        try:
            body = exc.response.text
            if body and len(body) < 800:
                data = json.loads(body)
                if isinstance(data, dict) and "error" in data:
                    err = data["error"]
                    if isinstance(err, str):
                        parts.append(_truncate(err, 200))
                    elif isinstance(err, dict) and "message" in err:
                        parts.append(_truncate(str(err["message"]), 200))
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            pass
        return _truncate(" — ".join(p for p in parts if p), 500)
    if isinstance(exc, ValueError) and "auto-router" in str(exc).lower():
        return (
            f"{_truncate(str(exc))} "
            "Set HF_INFERENCE_PROVIDER=hf-inference in .env if your key is not an hf_ Hub token."
        )
    return _truncate(str(exc))


class TravelState(TypedDict):
    user_input: str
    options: str
    destination: str
    days: int
    transport: str
    response: str
    model_id: str


_PLACE_TAIL = r"(?=\s*[\u2014\u2013\-,;:.!]|\s*$|\s+(?:by|with|for|and)\b)"


def _normalize_place_name(raw: str) -> str:
    s = raw.strip(" \t\n\r.,;:!\"'")
    if not s:
        return ""
    s = re.split(r"\s+by\s+", s, maxsplit=1, flags=re.I)[0].strip()
    s = re.sub(
        r"\s+\b(with|for|and|using|via)\s+.*$",
        "",
        s,
        flags=re.I,
    ).strip()
    return s.title() if s else ""


def extract_info(state: TravelState) -> TravelState:
    original = state["user_input"]
    combined = (original + " " + (state.get("options") or "")).strip()
    text = combined.lower().strip()

    days_match = re.search(r"(\d+)\s*days?", text)
    state["days"] = int(days_match.group(1)) if days_match else 3

    if any(w in text for w in ("flight", "plane", "fly", "flying")):
        state["transport"] = "flight"
    elif "train" in text:
        state["transport"] = "train"
    elif "bus" in text:
        state["transport"] = "bus"
    else:
        state["transport"] = "general"

    destination = ""
    place_patterns = (
        r"(?:to|visit|trip to|travel to|going to|head to|flying to|get to|see)\s+"
        r"([a-z][a-z\s'.-]{1,48}?)" + _PLACE_TAIL,
        r"(?:holiday|vacation|trip|itinerary|weekend)\s+in\s+"
        r"([a-z][a-z\s'.-]{1,48}?)" + _PLACE_TAIL,
        r"\d+\s*days?\s+in\s+([a-z][a-z\s'.-]{1,48}?)" + _PLACE_TAIL,
        r"(?:in|around|near)\s+([a-z][a-z\s'.-]{1,48}?)" + _PLACE_TAIL,
    )
    for pat in place_patterns:
        match = re.search(pat, text)
        if match:
            candidate = _normalize_place_name(match.group(1))
            if candidate and not candidate.isdigit():
                destination = candidate
                break

    if not destination:
        lead = re.match(
            r"^([A-Za-z][A-Za-z\s'.-]{1,48}?)\s+(?:trip|vacation|holiday|itinerary|visit)\b",
            combined.strip(),
        )
        if lead:
            destination = _normalize_place_name(lead.group(1))

    state["destination"] = destination
    return state


def check_missing(state: TravelState) -> str:
    combined = (state["user_input"] + " " + (state.get("options") or "")).strip()
    if not combined:
        return "missing"
    if not state.get("destination"):
        return "missing"
    return "plan_trip"


def handle_missing(state: TravelState) -> TravelState:
    if not (state["user_input"].strip() or (state.get("options") or "").strip()):
        state["response"] = (
            "Please describe your trip (for example: '4 days in Lisbon by train' or 'Plan a trip to Tokyo')."
        )
    else:
        state["response"] = (
            "I couldn't detect a destination. Try including a place name, e.g. "
            "'Plan a 5-day trip to Tokyo by train' or '3 days in Mexico City'."
        )
    return state


def _format_itinerary_output(raw: str) -> str:
    s = raw.strip()
    s = re.sub(r"(?s)^\s*```[\w]*\s*\n", "", s)
    s = re.sub(r"(?s)\n```\s*$", "", s)
    s = re.sub(r"(?m)^#{1,6}\s*", "", s)
    s = re.sub(r"(?m)^#{1,6}$", "", s)
    for _ in range(24):
        t = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
        if t == s:
            break
        s = t
    for _ in range(8):
        t = re.sub(r"__([^_]+)__", r"\1", s)
        if t == s:
            break
        s = t
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"`([^`]*)`", r"\1", s)
    s = re.sub(r"(?m)^[\*\-]\s+", "• ", s)
    s = re.sub(r"\*", "", s)
    s = re.sub(r"_{2,}", "", s)
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def _chat_response_text(response: object) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("Model returned no choices")
    message = getattr(choices[0], "message", None)
    if message is None:
        raise ValueError("Model choice has no message")
    content = getattr(message, "content", None)
    if content is None or not str(content).strip():
        raise ValueError("Model returned empty content")
    return str(content).strip()


def generate_plan(state: TravelState) -> TravelState:
    destination = state["destination"]
    days = state.get("days", 3)
    transport = state.get("transport", "general")
    transport_note = f" (travelling by {transport})" if transport != "general" else ""

    max_tokens = _env_int("HF_MAX_TOKENS", 1200)
    temperature = _env_float("HF_TEMPERATURE", 0.7)

    opts = (state.get("options") or "").strip()
    opts_block = f"\n\nExtra preferences and constraints: {opts}" if opts else ""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful travel planning assistant. "
                "Give detailed, structured itineraries with realistic pacing. "
                "Write in plain text only: no markdown — no # headings, no ** or __, no bullet asterisks, no code fences."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Create a {days}-day travel itinerary for {destination}{transport_note}. "
                "Format each day clearly as: Day 1: Morning: ... Afternoon: ... Evening: ... "
                "Include food tips and local attractions."
                f"{opts_block}"
            ),
        },
    ]

    try:
        token = _require_hf_token()
    except RuntimeError as e:
        state["response"] = str(e)
        state["model_id"] = ""
        return state

    models = _models_to_try()
    last_exc: BaseException | None = None

    for mid in models:
        try:
            client = InferenceClient(
                model=mid,
                token=token,
                provider=_inference_provider_name(token),  # type: ignore[arg-type]
            )
            response = client.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            state["response"] = _format_itinerary_output(_chat_response_text(response))
            state["model_id"] = mid
            return state
        except Exception as e:
            last_exc = e
            log.warning("generate_plan: model %s failed: %s", mid, e)

    detail = _friendly_inference_error(last_exc) if last_exc else "Unknown error"
    tried = ", ".join(models)
    state["model_id"] = ""
    state["response"] = (
        "Could not get a travel plan from the model.\n\n"
        f"Details: {detail}\n\n"
        f"Models tried: {tried}.\n"
        "If this is a 401/403 error, regenerate your token at "
        "https://huggingface.co/settings/tokens and update HF_TOKEN in .env.\n"
        "To use one model only, set HF_MODEL_ID in .env (see .env.example)."
    )
    return state


builder = StateGraph(TravelState)
builder.add_node("extract", extract_info)
builder.add_node("plan_trip", generate_plan)
builder.add_node("missing", handle_missing)
builder.set_entry_point("extract")
builder.add_conditional_edges(
    "extract",
    check_missing,
    {"missing": "missing", "plan_trip": "plan_trip"},
)
builder.add_edge("plan_trip", END)
builder.add_edge("missing", END)

graph = builder.compile()


def plan_trip_query(user_input: str, options: str | None = None) -> dict[str, str | int]:
    opt = (options or "").strip()
    result = graph.invoke(
        {
            "user_input": user_input,
            "options": opt,
            "destination": "",
            "days": 3,
            "transport": "",
            "response": "",
            "model_id": "",
        }
    )
    return {
        "response": result.get("response") or "",
        "destination": (result.get("destination") or ""),
        "days": int(result.get("days") or 3),
        "transport": (result.get("transport") or ""),
        "model_id": (
            (result.get("model_id") or "").strip()
            or (os.getenv("HF_MODEL_ID") or "").strip()
            or "—"
        ),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        _require_hf_token()
    except RuntimeError as err:
        print(err)
        raise SystemExit(1) from err

    env_model = (os.getenv("HF_MODEL_ID") or "").strip()
    if env_model:
        print(f"Using model: {env_model}")
    else:
        print("Using model fallback chain:", " → ".join(MODEL_FALLBACK_CHAIN))
    while True:
        user_input = input("\nEnter your travel query (or 'quit'): ")
        if user_input.strip().lower() == "quit":
            break

        result = graph.invoke(
            {
                "user_input": user_input,
                "options": "",
                "destination": "",
                "days": 3,
                "transport": "",
                "response": "",
                "model_id": "",
            }
        )

        print("\n--- Travel Plan ---")
        print(result["response"])
