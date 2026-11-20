"""Mind the Query: Text2Cypher questions on real Neo4j graphs, answered with and without a walk.

Mind the Query (Chauhan et al., EMNLP 2025 Industry Track) pairs questions with gold Cypher queries on 11 graphs, in
five kinds: simple and complex retrieval, simple and complex aggregation, and "evaluation" lookups. Ten of the graphs
come as Neo4j dumps. Download it with:

    git clone --depth 1 https://github.com/endeavorXx/Mind-the-Query data/downloads/mind_the_query/repo

Each graph is served on its own by a separate Neo4j, so the benchmark's graphs are never touched. For example:

    docker volume create mtq-pole
    docker run --rm -i -v mtq-pole:/data neo4j:2026.09.0-community \
        neo4j-admin database load neo4j --from-stdin --overwrite-destination < <repo>/Datasets/pole-50.dump
    docker run -d --name mtq-neo4j -p 127.0.0.1:7697:7687 -v mtq-pole:/data -e NEO4J_AUTH=neo4j/<password> \
        neo4j:2026.09.0-community

Two ways to answer, with the same LLM:
    Text2Cypher   the LLM writes the query from the graph's schema
    walk + edit   a decision model walks the graph with neo4j-graphrag's DecisionWalkRetriever, which finds the start
                  node with a full-text index. The walk becomes a query (cypher.compile_retriever_walk), and the LLM
                  writes the final query from it. It keeps the walk and adds conditions, aggregation and the
                  properties to return, or, if the walk does not fit, writes its own query

A query is right when it returns the same rows as the gold query. Rows are compared as sets of values, ignoring
column names, column order and row order. "Right with extra columns" also accepts rows that hold more values than
the question asked for.
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from collections import Counter
from pathlib import Path

import neo4j

from . import cypher, models
from .settings import openrouter_key

DUMPS = {"bloom50": "bloom-50.dump", "covid": "contact-tracing-50.dump", "er": "entity-resolution-50.dump",
         "gdsc": "graph-data-science-50.dump", "healthcare": "healthcare-analytics-50.dump",
         "legis_graph": "legis-graph-50.dump", "osm": "openstreetmap-50.dump", "pole": "pole-50.dump",
         "twitter_trolls": "twitter-trolls-50.dump", "wwc": "wwc2019-50.dump"}  # star_wars comes without a dump
CATEGORIES = ["Simple_Retrieval", "Complex_Retrieval", "Simple_Aggregation", "Complex_Aggregation", "Evaluation_query"]
# The properties that name a node, in order. The walk shows each node by the first of them that it has.
NAME_PROPERTIES = ["name", "fullName", "title", "screen_name", "tag", "description", "address", "streetAddress",
                   "manufacturerName", "outcome", "ageGroup", "email_address", "phoneNo", "phone", "ip", "ssn",
                   "accountNumber", "code", "areaCode", "number", "billID", "badge_no", "reg", "type", "text", "id"]
SEARCH_INDEX = "mtq_search"
RULES = "Return only what the question asks for. The query must only read."


# ---------------------------------------------------------------- questions and schemas
def questions(folder: Path, per_category: int = 10, seed: int = 0, datasets: list[str] | None = None) -> list[dict]:
    """`per_category` random test questions of each kind, on each graph that has a dump."""
    picked = []
    for dataset in datasets or DUMPS:
        for category in CATEGORIES:
            path = folder / "Train_Test_Splits" / "Automated" / dataset / "test" / f"{category}_test.json"
            items = json.loads(path.read_text(encoding="utf-8"))
            for item in random.Random(f"{seed}-{dataset}-{category}").sample(items, min(per_category, len(items))):
                picked.append({"qid": item["unique_id"], "dataset": dataset, "category": category,
                               "question": item["NL Question"], "gold": item["Cypher"]})
    return picked


def schema(folder: Path, dataset: str) -> dict:
    for line in (folder / "db_schema.jsonl").read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if dataset in entry:
            return entry[dataset]
    raise KeyError(dataset)


def schema_text(folder: Path, dataset: str) -> str:
    """The schema as the LLM sees it. Labels that start with "_" hold Neo4j Bloom's own settings, and are left out."""
    s = schema(folder, dataset)
    listed = lambda props: ", ".join(f"{p['property']} ({p['datatype']})" for p in props)
    nodes = [f"- {label}: {listed(props)}" for label, props in s["node_props"].items() if not label.startswith("_")]
    rels = [f"- {kind}: {listed(props)}" for kind, props in s.get("rel_props", {}).items()]
    patterns = [f"- (:{r['start']})-[:{r['type']}]->(:{r['end']})" for r in s["relationships"]
                if not (r["start"].startswith("_") or r["end"].startswith("_"))]
    return ("Node labels and their properties:\n" + "\n".join(nodes) +
            ("\nRelationship properties:\n" + "\n".join(rels) if rels else "") +
            "\nRelationships:\n" + "\n".join(patterns))


def prepare(driver, folder: Path, dataset: str) -> list[str]:
    """Create the full-text index that the walk finds its start node with, on every text property of every label.
    Returns the properties that name a node in this graph, in order."""
    s = schema(folder, dataset)
    labels = [label for label in s["node_props"] if not label.startswith("_")]
    props = {p["property"]: p["datatype"] for label in labels for p in s["node_props"][label]}
    texts = sorted(p for p, kind in props.items() if kind == "STRING")
    quote = lambda name: "`" + name.replace("`", "``") + "`"
    driver.execute_query(f"CREATE FULLTEXT INDEX {SEARCH_INDEX} IF NOT EXISTS "
                         f"FOR (n:{'|'.join(map(quote, labels))}) ON EACH [{', '.join('n.' + quote(p) for p in texts)}]")
    driver.execute_query("CALL db.awaitIndexes(600)")
    return [p for p in NAME_PROPERTIES if p in props] + [p for p in texts if p not in NAME_PROPERTIES]


# ---------------------------------------------------------------- running and comparing queries
def norm(value):
    """A value in a form that compares across queries: whole floats as ints, other floats rounded, nodes by id."""
    if isinstance(value, float):
        return int(value) if value.is_integer() else round(value, 4)
    if isinstance(value, (list, tuple)):
        return tuple(norm(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, norm(v)) for k, v in value.items()))
    if isinstance(value, (neo4j.graph.Node, neo4j.graph.Relationship)):
        return value.element_id
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return str(value)  # dates, times, durations and points


def run_rows(driver, query: str, params: dict | None = None, timeout: float = 30.0, max_rows: int = 10_000) -> dict:
    """Run a query in a read transaction. Returns its rows, each a sorted tuple of values, or the error code."""
    started = time.perf_counter()

    @neo4j.unit_of_work(timeout=timeout)
    def work(tx):
        rows = []
        for i, record in enumerate(tx.run(query.strip().rstrip(";"), params or {})):
            if i == max_rows:
                break
            rows.append(tuple(sorted(map(norm, record.values()), key=lambda v: (type(v).__name__, repr(v)))))
        return rows

    try:
        with driver.session(default_access_mode=neo4j.READ_ACCESS) as session:
            rows, error = session.execute_read(work), None
    except (neo4j.exceptions.Neo4jError, neo4j.exceptions.DriverError, ValueError, TypeError) as e:
        rows, error = None, getattr(e, "code", None) or type(e).__name__
    return {"rows": rows, "error": error, "ms": (time.perf_counter() - started) * 1000}


def same(gold: list[tuple], answer: list[tuple] | None) -> bool:
    return answer is not None and set(gold) == set(answer)


def contains(gold: list[tuple], answer: list[tuple] | None) -> bool:
    """Has the answer as many rows as the gold, each holding the values of its own gold row (extra columns allowed)?"""
    if answer is None or len(set(gold)) != len(set(answer)):
        return False
    if len(set(gold)) > 1000:  # ponytail: the matching below is quadratic, so big answers must match exactly
        return same(gold, answer)
    left = [Counter(row) for row in set(answer)]
    for row in map(Counter, set(gold)):
        match = next((i for i, other in enumerate(left) if not row - other), None)
        if match is None:
            return False
        left.pop(match)
    return True


# ---------------------------------------------------------------- the two ways to answer
def text2cypher_prompt(schema: str, question: str) -> str:
    return ("Write one Cypher query that answers the question from the Neo4j graph described below.\n\n"
            f"{schema}\n\n{RULES}\n\nQuestion: {question}")


def edit_prompt(schema: str, question: str, walk: dict, query: str) -> str:
    moves = ", then ".join(f"{s[:-1]} ({'outgoing' if s[-1] == '>' else 'incoming'})" for s in walk["steps"])
    names = ", ".join(map(str, walk["names"][:10])) + (" ..." if walk["total"] > 10 else "")
    return ("Write one Cypher query that answers the question from the Neo4j graph described below.\n\n"
            f"{schema}\n\n"
            f"A decision model walked the graph to find what the question is about. It started at {walk['start']} "
            f"({walk['labels']}), " + (f"followed {moves}, " if moves else "") +
            f"and ended on {walk['total']} node(s): {names}. Its walk is this query, which finds the start node "
            f"with the parameter $start_ids:\n{query}\n\n"
            "If the walk fits the question, keep its lines before RETURN as they are, and add what the question asks "
            "for: conditions, more MATCH lines, aggregation, ordering, and the properties to return. If the walk "
            "does not fit the question, write the query that the question needs without it.\n"
            f"{RULES}\n\nQuestion: {question}")


def scored(prefix: str, reply: dict, result: dict, gold: list[tuple]) -> dict:
    return {f"{prefix}_cypher": reply["cypher"], f"{prefix}_cost": reply["cost"],
            f"{prefix}_model": reply["model_version"], f"{prefix}_error": result["error"],
            f"{prefix}_rows": None if result["rows"] is None else len(set(result["rows"])),
            f"{prefix}_exact": same(gold, result["rows"]), f"{prefix}_lenient": contains(gold, result["rows"])}


async def answer(question: dict, schema: str, driver, retriever, writer: cypher.Text2Cypher, http) -> dict:
    """Run the gold query, then Text2Cypher, then the walk and its edit. A question whose gold query fails or finds
    nothing here is kept with its error, and left out of the scores. Without a walk (it failed, or found no start
    node), walk + edit falls back to the Text2Cypher answer."""
    record = {k: question[k] for k in ("qid", "dataset", "category", "question", "gold")}
    record |= {"t2c_cost": 0.0, "walk_cost": 0.0, "edit_cost": 0.0}
    gold = await asyncio.to_thread(run_rows, driver, question["gold"])
    record |= {"gold_rows": None if gold["rows"] is None else len(set(gold["rows"])), "gold_error": gold["error"]}
    if not gold["rows"]:
        return record

    nothing = {"rows": None, "error": None}
    reply = await writer.write(http, text2cypher_prompt(schema, question["question"]))
    result = await asyncio.to_thread(run_rows, driver, reply["cypher"]) if reply["cypher"] else nothing | {
        "error": reply["error"]}
    record |= scored("t2c", reply, result, gold["rows"])
    fallback = {"edit_exact": record["t2c_exact"], "edit_lenient": record["t2c_lenient"]}

    try:
        walk = await asyncio.to_thread(retriever.get_search_results, query_text=question["question"])
    except Exception as e:  # any failure of the walk leaves the edit nothing to start from
        return record | fallback | {"walk_error": f"{type(e).__name__}: {str(e)[:200]}", "edit_fallback": "walk_error"}
    meta = walk.metadata
    record |= {"walk_start": meta["start_nodes"], "walk_steps": meta["steps"], "walk_stopped_by": meta["stopped_by"],
               "walk_total": meta["result_nodes_total"], "walk_cost": meta["decision_cost"] or 0.0}
    if not meta["start_node_ids"]:
        return record | fallback | {"edit_fallback": "no_start_node"}

    query, params = cypher.compile_retriever_walk(meta)
    labels = (await asyncio.to_thread(driver.execute_query, "MATCH (n) WHERE elementId(n) = $id RETURN labels(n)",
                                      id=meta["start_node_ids"][0], routing_=neo4j.RoutingControl.READ)).records
    shown = {"start": meta["start_nodes"][0], "labels": ", ".join(labels[0][0]) if labels else "",
             "steps": meta["steps"], "names": [r["name"] for r in walk.records], "total": meta["result_nodes_total"]}
    reply = await writer.write(http, edit_prompt(schema, question["question"], shown, query))
    result = await asyncio.to_thread(run_rows, driver, reply["cypher"], params) if reply["cypher"] else nothing | {
        "error": reply["error"]}
    return record | {"walk_cypher": query} | scored("edit", reply, result, gold["rows"]) | {
        "kept_walk": bool(reply["cypher"]) and cypher.keeps_walk(reply["cypher"], query)}


async def run(folder: Path, dataset: str, picked: list[dict], driver, out: Path, llm: str = "openai/gpt-5.6-luna",
              walker: str = "liquid/d1-ultra", concurrency: int = 8, budget_usd: float = 1.0) -> float:
    """Answer the picked questions of one graph, which `driver` serves. One line per question goes to `out`, and
    questions already there are skipped. Stops starting new questions once `budget_usd` is spent."""
    from neo4j_graphrag.decision import OpenRouterDecisionModel
    from neo4j_graphrag.retrievers import DecisionWalkRetriever

    retriever = DecisionWalkRetriever(driver, OpenRouterDecisionModel(walker, api_key=openrouter_key()),
                                      fulltext_index_name=SEARCH_INDEX,
                                      node_name_properties=prepare(driver, folder, dataset))
    writer, text = cypher.Text2Cypher(llm), schema_text(folder, dataset)
    done = {json.loads(line)["qid"] for line in out.read_text().splitlines()} if out.exists() else set()
    queue: asyncio.Queue = asyncio.Queue()
    for question in picked:
        if question["dataset"] == dataset and question["qid"] not in done:
            queue.put_nowait(question)
    spent = 0.0

    async with models.client() as http:
        async def worker():
            nonlocal spent
            while not queue.empty() and spent < budget_usd:
                record = await answer(queue.get_nowait(), text, driver, retriever, writer, http)
                spent += record["t2c_cost"] + record["walk_cost"] + record["edit_cost"]
                with open(out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        await asyncio.gather(*(worker() for _ in range(concurrency)))
    return spent


if __name__ == "__main__":
    gold = [("Sergeant", "42-4529912"), (1, "x")]
    assert same(gold, [(1, "x"), ("Sergeant", "42-4529912")]) and not same(gold, [(1, "x")])
    assert contains(gold, [(1, "x", "extra"), ("42-4529912", "Sergeant", "Waldo")]) and not contains(gold, [(1, "x")])
    assert norm(2.0) == 2 and norm(1 / 3) == 0.3333 and norm([1.0, None]) == (1, None)
    print("self-check passed")
