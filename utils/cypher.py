"""Every walk as a Cypher query.

A walk is a set of start entities and a list of steps, such as "starred_actors<" then "directed_by>". Each step
follows one relation in one direction from every entity the walk stands on to all the entities behind it. So a walk
compiles to one Cypher query with one MATCH per step, and the query returns exactly the entities the walk ended on:

    MATCH (n0:KGEntity) WHERE n0.key IN $start_keys
    MATCH (n0)<-[:KG_EDGE {relation: 'starred_actors'}]-(n1:KGEntity) WHERE NOT n1.name IN $leave_out AND n1 <> n0
    WITH DISTINCT n1
    MATCH (n1)-[:KG_EDGE {relation: 'directed_by'}]->(n2:KGEntity) WHERE NOT n2.name IN $leave_out
    WITH DISTINCT n2
    RETURN n2.name AS name ORDER BY name

WITH DISTINCT after each step keeps the query on sets of entities, as the walk is. One long MATCH pattern would not be
the same: Cypher never uses one relationship twice in a pattern, and a walk can step back along the one it came by.
$leave_out holds the question's own entities on MetaQA, which the walk never steps onto, and is empty elsewhere.
"n1 <> n0" on incoming steps copies how the walk sees a relationship from a node to itself: as outgoing only. Such
loops exist because nodes are merged by name, so the country Djibouti and its capital, the city Djibouti, are one node.

    compile_walk, run_query       a walk's query, and running it read-only
    verify                        does every saved walk's query return the walk's answer? (no API calls)
    template, replay              reuse: a question whose pattern (the question with its entity masked) was walked
                                  before gets the cached query, with no model call
    Text2Cypher, run_text2cypher  the usual alternative: an LLM writes one query from the schema, which runs
                                  read-only and is scored like a walk
    compare                       scores, errors and cost side by side, paired against a baseline

Self-check (needs Neo4j and the saved runs): python -m utils.cypher
"""
from __future__ import annotations

import asyncio
import json
import math
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from statistics import NormalDist

import neo4j
import pandas as pd
from tqdm.auto import tqdm

from . import data, graph, models, scoring
from .runs import Run, model_folder
from .settings import RUNS


# ---------------------------------------------------------------- a walk's query
def literal(text: str) -> str:
    """A Cypher string literal."""
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def compile_steps(steps: list[str]) -> str:
    """The query for a list of steps, from the start keys in $start_keys."""
    lines = ["MATCH (n0:KGEntity) WHERE n0.key IN $start_keys"]
    for i, step in enumerate(steps, 1):
        relation, arrow = step[:-1], step[-1]
        edge = f"[:KG_EDGE {{relation: {literal(relation)}}}]"
        if arrow == ">":
            lines.append(f"MATCH (n{i - 1})-{edge}->(n{i}:KGEntity) WHERE NOT n{i}.name IN $leave_out")
        else:  # the walk counts a relationship from a node to itself as outgoing only
            lines.append(f"MATCH (n{i - 1})<-{edge}-(n{i}:KGEntity) WHERE NOT n{i}.name IN $leave_out "
                         f"AND n{i} <> n{i - 1}")
        lines.append(f"WITH DISTINCT n{i}")
    return "\n".join(lines + [f"RETURN n{len(steps)}.name AS name ORDER BY name"])


def params_for(benchmark: str, question: dict) -> dict:
    """The parameters that anchor a query on one question: its start entities (and on MetaQA, what to leave out)."""
    scope = graph.scope_of(benchmark, question)
    return {"start_keys": [graph.key(benchmark, scope, name) for name in question["start"]],
            "leave_out": list(question["start"]) if benchmark == "metaqa" else []}


def compile_walk(benchmark: str, question: dict, steps: list[str]) -> tuple[str, dict]:
    """The Cypher query of a walk, and its parameters."""
    return compile_steps(steps), params_for(benchmark, question)


def compile_retriever_walk(metadata: dict) -> tuple[str, dict]:
    """The query of a walk by neo4j-graphrag's DecisionWalkRetriever, from its result metadata: the start node ids and
    the steps, which are relationship types with a direction ("ACTED_IN>"). As in the retriever, no step goes back
    to a start node, and a relationship from a node to itself is outgoing only. If the walk was cut to the first
    max_frontier_nodes nodes (metadata "truncated_steps"), the query returns the whole, uncut set."""
    lines = ["MATCH (n0) WHERE elementId(n0) IN $start_ids"]
    for i, step in enumerate(metadata["steps"], 1):
        edge = "[:`" + step[:-1].replace("`", "``") + "`]"
        if step[-1] == ">":
            lines.append(f"MATCH (n{i - 1})-{edge}->(n{i}) WHERE NOT elementId(n{i}) IN $start_ids")
        else:
            lines.append(f"MATCH (n{i - 1})<-{edge}-(n{i}) WHERE NOT elementId(n{i}) IN $start_ids AND n{i} <> n{i - 1}")
        lines.append(f"WITH DISTINCT n{i}")
    last = f"n{len(metadata['steps'])}"
    lines.append(f"RETURN elementId({last}) AS id, coalesce({last}.name, {last}.title) AS name ORDER BY name, id")
    return "\n".join(lines), {"start_ids": metadata["start_node_ids"]}


def to_names(value) -> list[str]:
    """Entity names from one returned value: a name, a node, or a list of them."""
    if isinstance(value, (list, tuple)):
        return [name for v in value for name in to_names(v)]
    if isinstance(value, neo4j.graph.Node):
        return [str(value.get("name"))] if value.get("name") is not None else []
    return [] if value is None else [str(value)]


def run_query(driver, query: str, params: dict, timeout: float = 20.0, max_rows: int = 20_000) -> dict:
    """Run a query in a read transaction, which Neo4j refuses to write in. Returns the names of the first column
    (or of the column called "name"), the time it took, and the error code if it failed."""
    started = time.perf_counter()

    @neo4j.unit_of_work(timeout=timeout)
    def work(tx):
        result = tx.run(query, params)
        column = "name" if "name" in result.keys() else (result.keys() or [None])[0]
        names = []
        for i, record in enumerate(result):
            if i == max_rows:
                break
            names += to_names(record[column]) if column else []
        return names

    try:
        with driver.session(default_access_mode=neo4j.READ_ACCESS) as session:
            names, error = session.execute_read(work), None
    except (neo4j.exceptions.Neo4jError, neo4j.exceptions.DriverError, ValueError, TypeError) as e:
        names, error = None, getattr(e, "code", None) or type(e).__name__
    return {"names": None if names is None else sorted(set(names)), "ms": (time.perf_counter() - started) * 1000,
            "error": error}


# ---------------------------------------------------------------- checking the saved walks
def saved_walks(run_names: list[str] | None = None):
    """(run, benchmark, model folder, question, walk) for every saved walk with a model.json."""
    for path in sorted(RUNS.glob("*/*/*/results.jsonl")):
        run, benchmark, model = path.parts[-4:-1]
        if (run_names and run not in run_names) or not (path.parent / "model.json").exists():
            continue
        questions = {q["qid"]: q for q in json.loads((path.parents[1] / "questions.json").read_text())}
        for line in path.read_text().splitlines():
            record = json.loads(line)
            yield run, benchmark, model, questions[record["qid"]], record


def verify(run_names: list[str], driver=None) -> pd.DataFrame:
    """Run every saved walk's query and compare its entities with the walk's answer. Walks that ended in an API
    error or an invalid choice have no answer to compare, and are left out.

    One kind of difference is expected: a walk whose start entity is missing from the graph has nothing to follow,
    so it answers with the start name from the question. A query can only return entities that are in the graph, so
    such a walk is marked "start not in graph"."""
    own = driver is None
    driver = driver or graph.connect()
    rows = []
    try:
        for run, benchmark, model, question, walk in saved_walks(run_names):
            if walk["stopped_by"] in ("api_error", "invalid_choice"):
                continue
            query, params = compile_walk(benchmark, question, walk["steps"])
            result = run_query(driver, query, params)
            same = result["names"] == sorted(set(walk["final"]))
            missing_start = not same and not walk["steps"] and set(result["names"] or []) < set(walk["final"])
            rows.append({"run": run, "benchmark": benchmark, "model": model, "qid": walk["qid"],
                         "steps": len(walk["steps"]), "answer_size": len(walk["final"]), "same": same,
                         "reason": "start not in graph" if missing_start else None, "error": result["error"],
                         "query_ms": result["ms"], "walk_ms": walk["latency_ms"]})
    finally:
        if own:
            driver.close()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- reuse: walk once, then run the query
def template(benchmark: str, question: dict) -> str | None:
    """The question with its start entities replaced by [E], or None if a start name is not in the question.
    MetaQA marks the entity with brackets; elsewhere the start names are looked up in the lowercased question."""
    if benchmark == "metaqa":
        return re.sub(r"\[.+?\]", "[E]", question["question"])
    text = question["question"].lower()
    for name in sorted(question["start"], key=len, reverse=True):
        if name.lower() not in text:
            return None
        text = text.replace(name.lower(), "[E]")
    return text


def walk_confidence(decisions: list[dict]) -> float:
    """The product of the probabilities that the model gave its own choices (the lowest over the rounds of one
    step). A step that needed no model counts as 1."""
    p = 1.0
    for decision in decisions:
        if decision["calls"]:
            p *= min((c["probabilities"] or {}).get(c["option"], 0.0) for c in decision["calls"])
    return p


def replay(benchmark: str, stream: list[tuple[dict, dict, float]], rule: str = "first", sure: float = 0.9,
           driver=None) -> pd.DataFrame:
    """Answer a stream of (question, its walk, the walk's confidence) in order, as a cache would.

    A question whose template has a cached path gets that path's query, with no model call. If the query finds
    nothing, the path does not fit this entity, and the question is walked. A walk that ended with the model's STOP
    puts its path in the cache under its template:
        "first"  the first such walk of a template
        "sure"   the first such walk whose confidence is at least `sure`
        "agree"  once two such walks of a template took the same path
    Every question keeps its walk's own score too, so the cache can be compared with walking every question."""
    own = driver is None
    driver = driver or graph.connect()
    cache: dict[str, list[str]] = {}
    paths: dict[str, list[tuple]] = {}
    rows = []
    try:
        for question, walk, confidence in stream:
            key = template(benchmark, question)
            row = {"qid": question["qid"], "template": key, "walk_score": walk["score"], "walk_exact": walk["exact"],
                   "walk_cost": walk["cost"], "walk_ms": walk["latency_ms"]}
            if key in cache:
                result = run_query(driver, *compile_walk(benchmark, question, cache[key]))
                if result["names"]:
                    s = scoring.score(benchmark, question, result["names"])
                    rows.append(row | {"served_by": "cache", "score": s["score"], "exact": s["exact"],
                                       "cost": 0.0, "ms": result["ms"]})
                    continue
                row["cache_found_nothing"] = True
            rows.append(row | {"served_by": "walk", "score": walk["score"], "exact": walk["exact"],
                               "cost": walk["cost"], "ms": walk["latency_ms"]})
            if key is None or walk["stopped_by"] != "model_stop" or key in cache:
                continue
            if rule == "first" or (rule == "sure" and confidence >= sure):
                cache[key] = walk["steps"]
            elif rule == "agree":
                paths.setdefault(key, []).append(tuple(walk["steps"]))
                if paths[key].count(tuple(walk["steps"])) == 2:
                    cache[key] = walk["steps"]
    finally:
        if own:
            driver.close()
    return pd.DataFrame(rows)


def saved_stream(benchmark: str, run_names: list[str], model: str = "liquid__d1-ultra") -> list[tuple[dict, dict, float]]:
    """(question, walk, confidence) for the saved walks of one model, run after run, each in its run's question
    order. Walks that ended in an API error are left out."""
    stream = []
    for run in run_names:
        folder = RUNS / run / benchmark
        walks = {r["qid"]: r for r in map(json.loads, (folder / model / "results.jsonl").read_text().splitlines())}
        decisions = {r["qid"]: r["decisions"]
                     for r in map(json.loads, (folder / model / "decisions.jsonl").read_text().splitlines())}
        for question in json.loads((folder / "questions.json").read_text()):
            walk = walks.get(question["qid"])
            if walk and walk["stopped_by"] != "api_error":
                stream.append((question, walk, walk_confidence(decisions[question["qid"]])))
    return stream


def replay_summary(replayed: pd.DataFrame) -> dict:
    """How a replay compares with walking every question: the share the cache served, the score and cost of both,
    and on the questions the cache served, how often its answer scored better or worse than the walk's."""
    hits = replayed[replayed["served_by"] == "cache"]
    return {"questions": len(replayed), "served_by_cache": len(hits) / len(replayed),
            "score": replayed["score"].mean(), "walk_score": replayed["walk_score"].mean(),
            "exact": replayed["exact"].mean(), "walk_exact": replayed["walk_exact"].mean(),
            "cost_share": replayed["cost"].sum() / replayed["walk_cost"].sum(),
            "median_ms": replayed["ms"].median(), "walk_median_ms": replayed["walk_ms"].median(),
            "hits_better": (hits["score"] > hits["walk_score"] + 1e-9).mean() if len(hits) else float("nan"),
            "hits_worse": (hits["score"] < hits["walk_score"] - 1e-9).mean() if len(hits) else float("nan")}


# ---------------------------------------------------------------- Text2Cypher: an LLM writes the query
CYPHER_SCHEMA = {"type": "object", "properties": {"cypher": {"type": "string"}}, "required": ["cypher"],
                 "additionalProperties": False}
EDIT_SCHEMA = {"type": "object", "properties": {"needs_change": {"type": "boolean"}, "cypher": {"type": "string"}},
               "required": ["needs_change", "cypher"], "additionalProperties": False}
START_LINE = "MATCH (s:KGEntity) WHERE s.key IN $start_keys"


@lru_cache(maxsize=1)
def metaqa_examples() -> dict[str, list[str]]:
    """One example fact for each MetaQA relation: the first in sorted order."""
    examples = {}
    for head, relation, tail in data.read_metaqa_kb():
        examples.setdefault(relation, [head, relation, tail])
    return examples


def relations_of(benchmark: str, question: dict) -> set[str]:
    if benchmark == "metaqa":
        return set(metaqa_examples())
    return {t[1] for t in question["graph"]}


RETURN_RULE = ("Return the names of the answer entities, without duplicates, in a column called name. The query must "
               "only read.")


def graph_description(benchmark: str, question: dict) -> str:
    """The schema: how facts are stored, and the relations (on MetaQA with one example fact each)."""
    if benchmark == "metaqa":
        schema = ("The relations, each with one example fact (head, relation, tail):\n" + "\n".join(
            f"- {r}: ({h}, {r}, {t})" for r, (h, _, t) in sorted(metaqa_examples().items())) +
            "\nThe answer never includes the question's own entity.")
    else:
        schema = ("This question has its own part of Freebase, with these relations:\n" +
                  ", ".join(sorted(relations_of(benchmark, question))) +
                  "\nSome facts go through unnamed records, whose names are IDs such as m.0j4jq57. Such a record "
                  "links several entities, so a question may need one step to the record and one step from it.")
    return ("The graph: every entity is a node (:KGEntity {name}), and every fact (head, relation, tail) is a "
            "relationship (head)-[:KG_EDGE {relation: '<relation>'}]->(tail).\n" + schema)


def text2cypher_prompt(benchmark: str, question: dict) -> str:
    """The question, the schema and the rules. The model gets the same start entities as the walk."""
    return ("Write one Cypher query that answers the question from the knowledge graph described below.\n\n" +
            graph_description(benchmark, question) + "\n\n"
            f"The question is about: {', '.join(question['start'])}. Start the query with exactly this line, "
            f"which finds them:\n{START_LINE}\n" + RETURN_RULE + "\n\n"
            f"Question: {question['question']}")


def unknown_relations(query: str, known: set[str]) -> list[str]:
    """Relation names in the query that the graph does not have (a check for invented schema, by pattern)."""
    named = re.findall(r"relation\s*(?::|=|<>)\s*'([^']*)'", query)
    for listed in re.findall(r"relation\s+IN\s*\[([^\]]*)\]", query, flags=re.IGNORECASE):
        named += re.findall(r"'([^']*)'", listed)
    return sorted({r for r in named if r not in known})


def query_outcome(error: str | None, names: list[str] | None) -> str:
    if error is None:
        return "answer" if names else "empty"
    if "SyntaxError" in error:
        return "syntax_error"
    if "TimedOut" in error or "Timeout" in error:
        return "timeout"
    if "AccessMode" in error:
        return "write_refused"
    return "query_error"


def edit_prompt(benchmark: str, question: dict, walk: dict) -> str:
    """Show the model the query of a decision model's walk, and ask it to add what the walk cannot express."""
    query = compile_steps(walk["steps"])
    shown = ", ".join(walk["final"][:20]) + (f" ... (+{len(walk['final']) - 20} more)" if len(walk["final"]) > 20 else "")
    return ("Answer the question with one Cypher query on the knowledge graph described below.\n\n" +
            graph_description(benchmark, question) + "\n\n"
            f"A decision model answered the question by walking the graph one relation at a time, from "
            f"{', '.join(question['start'])}. Its walk "
            f"is this query, which uses the parameters $start_keys and $leave_out:\n{query}\n\n"
            f"It returns {len(walk['final'])} entities: {shown}\n\n"
            "Walking can only follow relations. Does the question also ask for something this query does not do, "
            "such as a condition on a date, a number, a type or a second entity, or the largest or the smallest?\n"
            "- If not, set needs_change to false and cypher to an empty string.\n"
            "- If so, set needs_change to true, and write the changed query in cypher. Keep every line of the walk "
            "before RETURN exactly as it is, then add what the question asks for: WHERE conditions on n1, n2 and so "
            "on, more MATCH lines from those entities, or ORDER BY and LIMIT. End with RETURN ... AS name.\n" +
            RETURN_RULE + "\n\n"
            f"Question: {question['question']}")


def keeps_walk(edited: str, walk_query: str) -> bool:
    """Does the edited query keep every line of the walk before RETURN, in order?"""
    lines, position = [" ".join(line.split()) for line in edited.splitlines()], 0
    for kept in walk_query.splitlines()[:-1]:
        kept = " ".join(kept.split())
        while position < len(lines) and lines[position] != kept:
            position += 1
        if position == len(lines):
            return False
        position += 1
    return True


@dataclass
class Text2Cypher:
    """An OpenRouter chat model that writes one Cypher query for each question, as a structured answer.

    mode="write": from the schema alone (Text2Cypher). mode="edit": from the query of a decision model's walk, which it
    may change to add conditions (see edit_prompt and run_text2cypher)."""
    slug: str
    reasoning: bool | str = False
    mode: str = "write"
    kind: str = field(default="Text2Cypher", init=False)

    def __post_init__(self):
        self.llm = models.LLM(self.slug, self.reasoning)
        self.kind = "Text2Cypher" if self.mode == "write" else "walk + edit"

    @property
    def label(self) -> str:
        return ("text2cypher " if self.mode == "write" else "walk-edit ") + self.llm.label

    async def check(self, http) -> dict:
        return await self.llm.check(http) | {"kind": self.kind}

    async def write(self, http, prompt: str) -> dict:
        schema = CYPHER_SCHEMA if self.mode == "write" else EDIT_SCHEMA
        body = {"model": self.slug, "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "cypher_query", "strict": True, "schema": schema}},
                "max_tokens": 2000 if self.reasoning is False else 16000, "usage": {"include": True},
                **models.provider_preferences(None, require_parameters=True)}
        if "temperature" in self.llm.supports:
            body["temperature"] = 0
        if "reasoning" in self.llm.supports:
            body["reasoning"] = {"effort": "none" if self.reasoning is False else
                                 ("medium" if self.reasoning is True else self.reasoning)}
        status, payload, latency = await models.post(http, models.CHAT_URL, body)
        if status != 200 or not payload.get("choices"):
            return {"cypher": None, "cost": 0.0, "latency_ms": latency, "model_version": None,
                    "error": f"http_{status}: {str(payload.get('error', ''))[:200]}"}
        usage = payload.get("usage") or {}
        reply = {"cost": usage.get("cost") or 0.0, "latency_ms": latency, "model_version": payload.get("model"),
                 "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")}
        try:
            answer = json.loads(payload["choices"][0]["message"]["content"])
            return reply | {"cypher": answer["cypher"], "needs_change": answer.get("needs_change"), "error": None}
        except (KeyError, TypeError, ValueError):
            return reply | {"cypher": None, "error": "no_query"}


async def run_text2cypher(run: Run, model: Text2Cypher, concurrency: int = 8, budget_usd: float | None = None,
                          walks: dict[str, dict] | None = None):
    """Write, run and score a query for every question of the run. Saved like a walk, in <run>/<benchmark>/<model>/
    results.jsonl. An API error has no answer and is left out of the scores, as for walks. A query that fails or
    finds nothing is a wrong (empty) answer.

    In edit mode, `walks` holds each question's saved walk. The walk's answer stands when the model sees no need to
    change it ("unchanged"), and when the edit drops a line of the walk, fails or finds nothing ("fallback_..."). So
    an edit can only narrow or extend the walk, never replace it. The cost includes the walk's."""
    if (model.mode == "edit") != (walks is not None):
        raise ValueError("Pass the saved walks in edit mode, and only then.")
    folder = model_folder(run, model)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "results.jsonl"
    done = {r["qid"] for r in map(json.loads, path.read_text().splitlines())} if path.exists() else set()
    queue: asyncio.Queue = asyncio.Queue()
    for q in run.questions:
        if q["qid"] not in done:
            queue.put_nowait(q)
    driver, spent = graph.connect(), 0.0
    bar = tqdm(total=len(run.questions), initial=len(done), desc=model.label, unit="question")
    async with models.client() as http:
        (folder / "model.json").write_text(json.dumps(await model.check(http), indent=1))

        async def worker():
            nonlocal spent
            while not queue.empty() and (budget_usd is None or spent < budget_usd):
                question = queue.get_nowait()
                walk = walks[question["qid"]] if walks is not None else None
                if walk is not None and walk["stopped_by"] in ("api_error", "invalid_choice"):
                    reply = {"cypher": None, "cost": 0.0, "latency_ms": 0, "model_version": None,
                             "error": f"walk_{walk['stopped_by']}"}
                else:
                    prompt = (edit_prompt(run.benchmark, question, walk) if walk is not None
                              else text2cypher_prompt(run.benchmark, question))
                    reply = await model.write(http, prompt)
                record = {"qid": question["qid"], "question": question["question"], "cypher": reply["cypher"],
                          "cost": reply["cost"], "latency_ms": reply["latency_ms"], "calls": 1,
                          "model_versions": [reply["model_version"]] if reply["model_version"] else [],
                          "reasoning_tokens": reply.get("reasoning_tokens"), "error": reply["error"]}
                if reply["cypher"] is None and walk is not None:
                    # No edit, so the walk's own answer stands: None after a walk API error, [] after an invalid choice.
                    ended = (walk["stopped_by"] if reply["error"].startswith("walk_")
                             else "fallback_" + ("no_query" if reply["error"] == "no_query" else "api_error"))
                    record |= {"stopped_by": ended, "final": walk["final"], "query_ms": None, "unknown_relations": []}
                elif reply["cypher"] is None:
                    api_error = reply["error"] != "no_query"
                    record |= {"stopped_by": "api_error" if api_error else "no_query",
                               "final": None if api_error else [], "query_ms": None, "unknown_relations": []}
                elif walk is not None and not reply["needs_change"]:
                    record |= {"stopped_by": "unchanged", "final": walk["final"], "query_ms": None,
                               "unknown_relations": []}
                elif walk is not None and not keeps_walk(reply["cypher"], compile_steps(walk["steps"])):
                    record |= {"stopped_by": "fallback_rewrote_walk", "final": walk["final"], "query_ms": None,
                               "unknown_relations": unknown_relations(reply["cypher"],
                                                                      relations_of(run.benchmark, question))}
                else:
                    result = await asyncio.to_thread(run_query, driver, reply["cypher"],
                                                     params_for(run.benchmark, question))
                    names = result["names"] or []
                    if run.benchmark == "metaqa":
                        names = [n for n in names if n not in question["start"]]
                    outcome = query_outcome(result["error"], names)
                    if walk is not None and outcome != "answer":
                        outcome, names = "fallback_" + outcome, walk["final"]
                    record |= {"stopped_by": "edited" if walk is not None and outcome == "answer" else outcome,
                               "final": names, "error": result["error"], "query_ms": result["ms"],
                               "unknown_relations": unknown_relations(reply["cypher"],
                                                                      relations_of(run.benchmark, question))}
                if walk is not None and walk["stopped_by"] != "api_error":
                    record |= {"edit_cost": record["cost"], "cost": record["cost"] + walk["cost"],
                               "latency_ms": record["latency_ms"] + walk["latency_ms"],
                               "calls": record["calls"] + walk["calls"]}
                record |= scoring.score(run.benchmark, question, record["final"])
                with open(path, "a") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                spent += record["cost"]
                bar.update(1)
                bar.set_postfix(spent=f"${spent:.4f}")

        try:
            await asyncio.gather(*(worker() for _ in range(concurrency)))
        finally:
            bar.close()
            driver.close()
    if not queue.empty():
        print(f"{model.label}: stopped at the budget of ${budget_usd}; {queue.qsize()} questions left.")


# ---------------------------------------------------------------- side by side
def paired(a: pd.Series, b: pd.Series) -> dict:
    """Mean of a - b over the same questions, with a 95% interval and a two-sided p-value (normal approximation)."""
    d = (a - b).dropna()
    mean, se = d.mean(), d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
    p = 2 * (1 - NormalDist().cdf(abs(mean) / se)) if se and se > 0 else (1.0 if mean == 0 else 0.0)
    return {"difference": mean, "low": mean - 1.96 * se, "high": mean + 1.96 * se, "p_value": p, "questions": len(d)}


def compare(results: dict[str, pd.DataFrame], baseline: str) -> pd.DataFrame:
    """One row per method: score, exactly correct, cost per 1,000 questions, how its answers ended, and the paired
    difference from the baseline on the questions both answered (API errors left out)."""
    base = results[baseline].set_index("qid")
    rows = []
    for name, df in results.items():
        df = df[df["stopped_by"] != "api_error"].set_index("qid")
        ended = df["stopped_by"].value_counts(normalize=True)
        both = df.index.intersection(base.index)
        rows.append({"method": name, "questions": len(df), "score": df["score"].mean(),
                     "exactly_correct": df["exact"].mean(), "usd_per_1000": 1000 * df["cost"].mean(),
                     "median_ms": df["latency_ms"].median()}
                    | {f"ended_{k}": v for k, v in ended.items()}
                    | ({f"vs_{baseline}_{k}": v for k, v in paired(df.loc[both, "score"],
                                                                   base.loc[both, "score"]).items()}
                       if name != baseline else {}))
    return pd.DataFrame(rows).set_index("method")


if __name__ == "__main__":
    query = compile_steps(["starred_actors<", "directed_by>"])
    assert query.splitlines()[1] == ("MATCH (n0)<-[:KG_EDGE {relation: 'starred_actors'}]-(n1:KGEntity) "
                                     "WHERE NOT n1.name IN $leave_out AND n1 <> n0"), query
    assert literal("it's") == "'it\\'s'"
    checked = verify(["run-001"])
    differ = checked[~checked["same"]]
    assert len(checked) > 500 and differ["reason"].eq("start not in graph").all(), differ.head()
    print(f"self-check passed: {len(checked)} saved walks; every query returned the walk's answer, except "
          f"{len(differ)} walk(s) whose start entity is not in the graph")
