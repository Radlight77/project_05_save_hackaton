"""One bounded SDK decision. Persist usage and keep the conversation apart from task state."""

import json
import math
import os
from pathlib import Path

from agents import (
    Agent,
    ModelSettings,
    OpenAIChatCompletionsModel,
    Runner,
    SQLiteSession,
    set_tracing_disabled,
)
from agents.exceptions import AgentsException
from openai import AsyncOpenAI, OpenAIError

from run_store import clean, journal, save_state
from scheduling import RuleError

MAX_OUTPUT = 240


async def remember(purpose, messages):
    session = SQLiteSession(purpose, Path("sessions.db"))
    try:
        await session.add_items(messages)
    finally:
        session.close()


def settings():
    provider = os.getenv("MODEL_PROVIDER", "ollama").lower()
    if provider == "ollama":
        return (
            provider,
            os.getenv("OLLAMA_MODEL", "qwen3-vl:30b-a3b-instruct"),
            os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"),
            "ollama",
        )
    if provider == "openai":
        params = (
            os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
            os.getenv("OPENAI_BASE_URL") or None,
            os.getenv("OPENAI_API_KEY"),
        )
    elif provider == "hf":
        params = (
            os.getenv("HF_MODEL", "openai/gpt-oss-120b"),
            os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1"),
            os.getenv("HF_TOKEN"),
        )
    else:
        raise RuleError("unknown_model_provider")
    if not params[0] or not params[2]:
        raise RuleError("missing_model_config")
    return (provider, *params)


def rates(provider):
    if provider == "ollama":
        return 0.0, 0.0
    # Budget estimates for a paid endpoint; the operator can replace them with its actual tariff.
    try:
        values = [
            float(os.getenv(k, default))
            for k, default in (("INPUT_USD_PER_MILLION", "10"), ("OUTPUT_USD_PER_MILLION", "60"))
        ]
    except ValueError:
        raise RuleError("invalid_model_rate") from None
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise RuleError("invalid_model_rate")
    return values


async def choose(state, config, purpose, instructions, payload, allowed, *, persist=save_state):
    provider, name, base_url, api_key = settings()
    set_tracing_disabled(True)
    prompt = clean(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    instructions = clean(instructions) + "\nОтвет имеет форму JSON с единственным полем id."
    if purpose == "venue":
        choices = payload.get("feasible_ids") or [venue["id"] for venue in payload["venues"]]
    else:
        choices = sorted(allowed)
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "decision",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"id": {"type": "string", "enum": choices}},
                "additionalProperties": False,
                "required": ["id"],
            },
        },
    }
    rate_in, rate_out = rates(provider)
    usage = state["token_usage"]
    reservation = (
        (len((instructions + prompt + json.dumps(response_format)).encode()) + 512) * rate_in
        + MAX_OUTPUT * rate_out
    ) / 1_000_000
    if state["turns"] >= config["max_turns"]:
        raise RuleError("max_turns")
    if usage["cost_usd"] + usage["reserved_usd"] + reservation > config["max_cost_usd"]:
        raise RuleError("max_cost_usd")
    state["turns"] += 1
    usage["calls_started"] += 1
    usage["reserved_usd"] += reservation
    usage["model"], usage["provider"] = name, provider
    persist(state)
    journal("model_start", purpose=purpose, model=name, turn=state["turns"])
    try:
        async with AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=120, max_retries=0) as client:
            model = OpenAIChatCompletionsModel(model=name, openai_client=client)
            # Some paid reasoning models only support max_completion_tokens.
            extra = {"response_format": response_format}
            if provider == "openai":
                extra["max_completion_tokens"] = MAX_OUTPUT
            options = ModelSettings(
                max_tokens=None if provider == "openai" else MAX_OUTPUT,
                temperature=0.0 if provider == "ollama" else None,
                extra_body=extra,
            )
            agent = Agent(
                name="Организатор", instructions=instructions, model=model, model_settings=options
            )
            result = await Runner.run(agent, prompt, max_turns=1)
    except (OpenAIError, AgentsException, TimeoutError, OSError) as exc:
        journal("model_error", purpose=purpose, error=type(exc).__name__)
        raise RuleError("model_unavailable") from None
    measured = result.context_wrapper.usage
    if measured.input_tokens <= 0 or measured.output_tokens <= 0:
        raise RuleError("missing_model_usage")
    cost = (measured.input_tokens * rate_in + measured.output_tokens * rate_out) / 1_000_000
    usage["calls"] += measured.requests
    usage["input_tokens"] += measured.input_tokens
    usage["output_tokens"] += measured.output_tokens
    usage["total_tokens"] += measured.total_tokens
    usage["reserved_usd"] = max(0.0, usage["reserved_usd"] - reservation)
    usage["cost_usd"] = round(usage["cost_usd"] + cost, 9)
    persist(state)
    journal(
        "model_end",
        purpose=purpose,
        model=name,
        input_tokens=measured.input_tokens,
        output_tokens=measured.output_tokens,
        cost_usd=cost,
    )
    if usage["cost_usd"] > config["max_cost_usd"]:
        raise RuleError("max_cost_usd")
    try:
        value = json.loads(str(result.final_output))
        answer = value.get("id") if isinstance(value, dict) else None
    except json.JSONDecodeError:
        answer = None
    if not isinstance(answer, str):
        answer = None
    recorded = (
        json.dumps({"id": answer}, ensure_ascii=False) if answer in choices else "[ответ вне схемы]"
    )
    journal(
        "model_choice",
        purpose=purpose,
        selected=answer if answer in choices else None,
        accepted=answer in allowed,
    )
    await remember(
        purpose,
        [
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": recorded},
        ],
    )
    if answer not in allowed:
        raise RuleError("invalid_model_choice")
    return answer
