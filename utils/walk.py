"""One walk: get the options, ask the model, follow its choice, and repeat until it chooses STOP or reaches the step limit.

On WebQSP and CWQ, STOP is not offered while every entity that the walk stands on is an unnamed Freebase record (shown
by its ID, such as m.0j4jq57). These records link other entities together, so the walk moves on to the named entities
behind them. The options are then numbered from 0 without STOP. If only one connection is left, the walk follows it
without asking the model.

Every step goes into the decision log: where the walk stood, the options it was offered, and what the model answered.
"""
from __future__ import annotations

import math
from dataclasses import asdict

from . import graph
from .data import BENCHMARKS

INSTRUCTIONS = (
    "We are walking a knowledge graph step by step to answer the question in the state. "
    "The state lists the entities we are currently standing on and the path taken so far. "
    "Choose the connection to follow next. Choose STOP only if the current entities are "
    "themselves the answer to the question."
)
MAX_OPTIONS = 255  # the decision models' limit per request; a step with more options is decided by a tournament


def build_request(state: dict, options: list[dict]) -> dict:
    """The request that every model receives: the state, the instructions and the numbered options."""
    return {"state": state,
            "questions": {"next_step": {"type": "choice", "instructions": INSTRUCTIONS,
                                        "criteria": {str(i): o["text"] for i, o in enumerate(options)}}}}


def split_evenly(indexes: list[int], limit: int) -> list[list[int]]:
    n = math.ceil(len(indexes) / limit)
    size, extra = divmod(len(indexes), n)
    groups, start = [], 0
    for i in range(n):
        end = start + size + (i < extra)
        groups.append(indexes[start:end])
        start = end
    return groups


def all_unnamed(entities: list[str]) -> bool:
    """True if every entity is an unnamed Freebase record, shown by its ID (m.0j4jq57, g.11b6...)."""
    return bool(entities) and all(name.startswith(("m.", "g.")) for name in entities)


async def decide(model, http, state: dict, options: list[dict], offer_stop: bool = True) -> tuple[int | None, list[dict]]:
    """Ask the model to choose one option. Returns (index of the chosen option or None, the calls made).

    With more than 255 options, STOP waits for a final round, and the relations are split into even groups. The
    model picks a winner in each group, then chooses between the winners and STOP. Without STOP, the relations are
    numbered from 0. Each call records `option_indexes`: which of the step's options it offered, in order."""
    calls = []

    async def ask(indexes: list[int]) -> int | None:
        choice = await model.choose(http, build_request(state, [options[i] for i in indexes]))
        calls.append(asdict(choice) | {"option_indexes": indexes})
        return None if choice.error else indexes[int(choice.option)]

    stop = [0] if offer_stop else []
    relations = list(range(1, len(options)))
    if len(stop) + len(relations) <= MAX_OPTIONS:
        return await ask(stop + relations), calls
    winners = []
    for group in split_evenly(relations, MAX_OPTIONS):
        winners.append(await ask(group))
        if winners[-1] is None:
            return None, calls
    return await ask(stop + winners), calls


async def walk(benchmark: str, question: dict, model, http, driver, max_steps: int) -> dict:
    """Walk one question. Returns the walk's summary, with its decision log under "decisions"."""
    current, path, steps, decisions = list(question["start"]), [], [], []
    stopped_by, error = "step_limit", None
    for hop in range(max_steps):
        rows = await graph.neighbours(driver, benchmark, question, current)
        options = graph.build_options(rows, current, benchmark, question)
        if len(options) == 1:  # nothing to follow: STOP is the only option
            stopped_by = "nothing_to_follow"
            break
        state = {"question": question["question"], "current_entities": current, "path_so_far": path, "hop": hop}
        offer_stop = not (BENCHMARKS[benchmark].no_stop_on_unnamed and all_unnamed(current))
        if not offer_stop and len(options) == 2:  # without STOP, one connection is left: follow it
            chosen, calls = 1, []
        else:
            chosen, calls = await decide(model, http, state, options, offer_stop)
        decisions.append({"hop": hop, "current": current, "options": [o["text"] for o in options],
                          "stop_offered": offer_stop, "chosen": chosen, "calls": calls})
        error = next((c["error"] for c in calls if c["error"]), None)
        if error:
            stopped_by = "invalid_choice" if error == "invalid_choice" else "api_error"
            break
        option = options[chosen]
        if option["step"] == "STOP":
            stopped_by = "model_stop"
            break
        path.append(graph.describe_step(current, option))
        steps.append(option["step"])
        current = option["targets"]

    # An invalid answer counts as a wrong (empty) answer. An API error has no answer and is left out of the scores.
    final = None if stopped_by == "api_error" else ([] if stopped_by == "invalid_choice" else current)
    calls = [c for d in decisions for c in d["calls"]]
    return {"qid": question["qid"], "question": question["question"], "steps": steps, "final": final,
            "stopped_by": stopped_by, "error": error, "calls": len(calls),
            "cost": sum(c["cost"] or 0 for c in calls), "latency_ms": sum(c["latency_ms"] for c in calls),
            "model_versions": sorted({c["model_version"] for c in calls if c["model_version"]}),
            "decisions": decisions}


async def walk_question(question: str, start: list[str], model, benchmark: str = "metaqa", max_steps: int = 4,
                        qid: str = "my-question") -> dict:
    """Walk one question that you type in, for example on the MetaQA film graph. Returns the walk with its log."""
    from . import models

    q = {"qid": qid, "question": question, "start": list(start)}
    driver = graph.connect_async()
    try:
        async with models.client() as http:
            await model.check(http)
            return await walk(benchmark, q, model, http, driver, max_steps)
    finally:
        await driver.close()
