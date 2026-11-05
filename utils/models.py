"""The two kinds of model that can choose the next step of a walk. Both get the same question and the same options.

DecisionModel
    Sends the numbered options to OpenRouter's decisions endpoint. It returns a probability for every option.

LLM
    Sends the same options to any OpenRouter chat model as text, and asks for a structured answer: a Pydantic model
    whose only field, `option`, must be one of the offered option numbers. The schema goes to the API as a strict JSON
    schema, and the reply is validated with the same Pydantic model, so no text has to be parsed.

    >>> answer_model(("0", "1", "2")).model_json_schema()
    {'additionalProperties': False, 'properties': {'option': {'enum': ['0', '1', '2'], ...}}, 'required': ['option'], ...}
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from .settings import openrouter_key

OPENROUTER = "https://openrouter.ai/api"
DECISIONS_URL = f"{OPENROUTER}/alpha/decisions"
CHAT_URL = f"{OPENROUTER}/v1/chat/completions"
RETRY_STATUSES = {408, 429, 500, 502, 503, 524, 529}
ANSWER_FORMAT = 'Answer with the number of one option in the field "option".'


@dataclass
class Choice:
    """What a model answered at one step."""
    option: str | None                       # the chosen option number, or None if the call failed
    probabilities: dict[str, float] | None   # decision models: every option; LLMs: none
    cost: float = 0.0                        # US dollars, as billed by OpenRouter
    latency_ms: int = 0
    error: str | None = None
    reply: str | None = None                 # LLMs: the raw JSON reply
    reasoning_tokens: int | None = None
    model_version: str | None = None


@lru_cache(maxsize=4096)
def answer_model(option_keys: tuple[str, ...]) -> type[BaseModel]:
    """The Pydantic model of an LLM's answer: one field, `option`, which must be one of the offered numbers."""
    return create_model("OptionChoice", __config__=ConfigDict(extra="forbid"), option=(Literal[option_keys], ...))


def client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=180, headers={"Authorization": f"Bearer {openrouter_key()}"})


async def post(http: httpx.AsyncClient, url: str, body: dict, attempts: int = 6) -> tuple[int | None, dict, int]:
    """POST with retries on rate limits and server errors. Returns (status, json body, latency in ms)."""
    for attempt in range(attempts):
        started = time.perf_counter()
        try:
            response = await http.post(url, json=body)
            status = response.status_code
            payload = response.json() if status == 200 else {"error": response.text[:500]}
        except (httpx.HTTPError, ValueError) as error:
            status, payload = None, {"error": repr(error)}
        latency = round((time.perf_counter() - started) * 1000)
        if status is not None and status not in RETRY_STATUSES:
            return status, payload, latency
        await asyncio.sleep(min(60, 2 ** attempt))
    return status, payload, latency


def provider_preferences(provider: str | None, require_parameters: bool = False) -> dict:
    prefs = {"require_parameters": True} if require_parameters else {}
    if provider:
        prefs |= {"order": [provider], "allow_fallbacks": False}
    return {"provider": prefs} if prefs else {}


# ---------------------------------------------------------------- decision models
@dataclass
class DecisionModel:
    """A decision model on OpenRouter, for example "liquid/d1-ultra". `provider` optionally pins one provider."""
    slug: str
    provider: str | None = None
    kind: str = field(default="decision model", init=False)

    @property
    def label(self) -> str:
        return self.slug

    async def check(self, http: httpx.AsyncClient) -> dict:
        return {"slug": self.slug, "kind": self.kind, "provider": self.provider}

    async def choose(self, http: httpx.AsyncClient, request: dict) -> Choice:
        body = {"model": self.slug, **request, **provider_preferences(self.provider)}
        status, payload, latency = await post(http, DECISIONS_URL, body)
        if status != 200:
            return Choice(None, None, latency_ms=latency, error=f"http_{status}: {payload.get('error', '')[:200]}")
        answer = payload.get("answers", {}).get("next_step", {})
        choice = Choice(option=answer.get("choice"), probabilities=answer.get("probabilities"),
                        cost=(payload.get("usage") or {}).get("cost") or 0.0, latency_ms=latency,
                        model_version=payload.get("model"))
        if choice.option not in request["questions"]["next_step"]["criteria"]:
            choice.error = "invalid_choice"
        return choice


# ---------------------------------------------------------------- LLMs with a structured (Pydantic) answer
def render_text(request: dict) -> str:
    """The decision request as plain text: the same instructions, state and numbered options."""
    state, step = request["state"], request["questions"]["next_step"]
    path = "\n".join(f"  {i}. {s}" for i, s in enumerate(state["path_so_far"], 1)) or "  (none)"
    options = "\n".join(f"{k}: {text}" for k, text in step["criteria"].items())
    return (f"{step['instructions']}\n\n"
            f"Question: {state['question']}\n"
            f"Current entities: {', '.join(state['current_entities'])}\n"
            f"Path so far:\n{path}\n"
            f"Hop: {state['hop']}\n\n"
            f"Options:\n{options}\n\n{ANSWER_FORMAT}")


@dataclass
class LLM:
    """Any chat model on OpenRouter, for example "openai/gpt-5.6-luna" or "anthropic/claude-sonnet-4.5".

    reasoning: False turns reasoning off where the model allows it; True uses the model's default reasoning; or give an
    effort such as "low" or "high". `provider` optionally pins one provider."""
    slug: str
    reasoning: bool | str = False
    provider: str | None = None
    kind: str = field(default="LLM", init=False)
    supports: set[str] = field(default_factory=set, init=False)
    prices: dict = field(default_factory=dict, init=False)

    @property
    def label(self) -> str:
        return self.slug + ("" if self.reasoning is False else f" (reasoning {self.reasoning})")

    async def check(self, http: httpx.AsyncClient) -> dict:
        """Look up the model on OpenRouter: does it support structured output, and what does it cost?"""
        response = await http.get(f"{OPENROUTER}/v1/models/{self.slug}/endpoints")
        if response.status_code != 200:
            raise ValueError(f"OpenRouter does not know the model {self.slug!r}.")
        endpoints = response.json()["data"]["endpoints"]
        if self.provider:
            endpoints = [e for e in endpoints if e["tag"].split("/")[0] == self.provider]
        structured = [e for e in endpoints if {"structured_outputs", "response_format"} & set(e.get("supported_parameters") or [])]
        if not structured:
            raise ValueError(f"{self.slug} has no endpoint with structured outputs on OpenRouter, so it cannot give a "
                             "Pydantic answer. Choose another model.")
        self.supports = set.intersection(*(set(e.get("supported_parameters") or []) for e in structured))
        cheapest = min(structured, key=lambda e: float(e["pricing"]["prompt"]))
        self.prices = {"input_per_million": float(cheapest["pricing"]["prompt"]) * 1e6,
                       "output_per_million": float(cheapest["pricing"]["completion"]) * 1e6}
        return {"slug": self.slug, "kind": self.kind, "reasoning": self.reasoning, "provider": self.provider,
                "prices_usd": self.prices, "supports": sorted(self.supports)}

    def body(self, request: dict) -> dict:
        keys = tuple(request["questions"]["next_step"]["criteria"])
        body = {
            "model": self.slug,
            "messages": [{"role": "user", "content": render_text(request)}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "option_choice", "strict": True, "schema": answer_model(keys).model_json_schema()}},
            "max_tokens": 64 if self.reasoning is False else 8000,
            "usage": {"include": True},
            **provider_preferences(self.provider, require_parameters=True),
        }
        if "temperature" in self.supports:
            body["temperature"] = 0
        if "reasoning" in self.supports:
            effort = "none" if self.reasoning is False else ("medium" if self.reasoning is True else self.reasoning)
            body["reasoning"] = {"effort": effort}
        return body

    async def choose(self, http: httpx.AsyncClient, request: dict) -> Choice:
        status, payload, latency = await post(http, CHAT_URL, self.body(request))
        if status != 200 or not payload.get("choices"):
            return Choice(None, None, latency_ms=latency, error=f"http_{status}: {str(payload.get('error', ''))[:200]}")
        message = payload["choices"][0].get("message") or {}
        usage = payload.get("usage") or {}
        choice = Choice(option=None, probabilities=None, cost=usage.get("cost") or 0.0, latency_ms=latency,
                        reply=message.get("content"), model_version=payload.get("model"),
                        reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"))
        keys = tuple(request["questions"]["next_step"]["criteria"])
        try:
            choice.option = answer_model(keys).model_validate_json(choice.reply or "").option
        except ValidationError:
            choice.error = "invalid_choice"
        return choice
