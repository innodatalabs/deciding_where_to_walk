"""Neo4j: loading the benchmark graphs, and turning "what is connected to these entities?" into numbered options.

Every entity is a node labelled KGEntity with a unique `key`, and every fact is a KG_EDGE relationship with a
`relation` property. WebQSP and CWQ give each question its own small subgraph, so their keys look like
"webqsp|WebQTest-0|Jamaica". MetaQA has one shared film knowledge base, so its keys look like "metaqa|kb|Tom Hanks".
Entities with the same name inside one graph are the same node.
"""
from __future__ import annotations

from neo4j import AsyncGraphDatabase, GraphDatabase

from .settings import neo4j_settings

MAX_NAMES = 10  # how many target names an option shows; the walk still moves to all of them

LOAD = """
UNWIND $rows AS row
MERGE (h:KGEntity {key: row.hk}) ON CREATE SET h.name = row.h, h.benchmark = $benchmark, h.scope = $scope
MERGE (t:KGEntity {key: row.tk}) ON CREATE SET t.name = row.t, t.benchmark = $benchmark, t.scope = $scope
CREATE (h)-[:KG_EDGE {relation: row.r}]->(t)
"""

NEIGHBOURS = """
MATCH (e:KGEntity)-[r:KG_EDGE]-(t:KGEntity)
WHERE e.key IN $keys
RETURN r.relation AS relation,
       CASE WHEN startNode(r) = e THEN 'outgoing' ELSE 'incoming' END AS direction,
       t.name AS target
"""


# ---------------------------------------------------------------- connecting
def connect():
    uri, auth = neo4j_settings()
    driver = GraphDatabase.driver(uri, auth=auth)
    driver.verify_connectivity()
    return driver


def connect_async():
    uri, auth = neo4j_settings()
    return AsyncGraphDatabase.driver(uri, auth=auth)


def scope_of(benchmark: str, question: dict) -> str:
    return "kb" if benchmark == "metaqa" else question["qid"]


def key(benchmark: str, scope: str, name: str) -> str:
    return f"{benchmark}|{scope}|{name}"


# ---------------------------------------------------------------- loading
def setup(driver) -> None:
    driver.execute_query("CREATE CONSTRAINT kg_entity_key IF NOT EXISTS FOR (e:KGEntity) REQUIRE e.key IS UNIQUE")
    driver.execute_query("CREATE INDEX kg_entity_scope IF NOT EXISTS FOR (e:KGEntity) ON (e.benchmark, e.scope)")


def loaded_scopes(driver, benchmark: str) -> set[str]:
    records, _, _ = driver.execute_query(
        "MATCH (e:KGEntity {benchmark: $b}) RETURN DISTINCT e.scope AS scope", b=benchmark)
    return {r["scope"] for r in records}


def load_triples(driver, benchmark: str, scope: str, triples: list[list[str]], batch: int = 5000) -> int:
    """Load one graph (one question's subgraph, or the MetaQA knowledge base). Duplicate triples are loaded once."""
    unique = sorted({tuple(t) for t in triples})
    rows = [{"h": h, "r": r, "t": t, "hk": key(benchmark, scope, h), "tk": key(benchmark, scope, t)} for h, r, t in unique]
    for i in range(0, len(rows), batch):
        driver.execute_query(LOAD, rows=rows[i:i + batch], benchmark=benchmark, scope=scope)
    return len(rows)


def ensure_loaded(driver, benchmark: str, questions: list[dict]) -> dict:
    """Load whatever these questions need and is not in Neo4j yet: the MetaQA film knowledge base, or each WebQSP and
    CWQ question's subgraph. Returns how many entities and facts the questions' graphs hold."""
    from tqdm.auto import tqdm

    from . import data

    setup(driver)
    loaded = loaded_scopes(driver, benchmark)
    if benchmark == "metaqa":
        if "kb" not in loaded:
            print("Loading the film knowledge base into Neo4j (once, about a minute)...")
            load_triples(driver, "metaqa", "kb", data.read_metaqa_kb())
        return count(driver, "metaqa", ["kb"])
    missing = [q for q in questions if q["qid"] not in loaded]
    for q in tqdm(missing, desc="Loading question graphs", disable=not missing):
        load_triples(driver, benchmark, q["qid"], q["graph"])
    return count(driver, benchmark, [q["qid"] for q in questions])


def count(driver, benchmark: str, scopes: list[str] | None = None) -> dict:
    where = "" if scopes is None else "AND e.scope IN $scopes"
    records, _, _ = driver.execute_query(
        f"MATCH (e:KGEntity {{benchmark: $b}}) WHERE true {where} "
        "OPTIONAL MATCH (e)-[r:KG_EDGE]->() RETURN count(DISTINCT e) AS entities, count(r) AS facts",
        b=benchmark, scopes=scopes)
    return dict(records[0])


def clear(driver, benchmark: str) -> None:
    """Delete one benchmark's graph (KGEntity nodes only; nothing else in the database is touched)."""
    while True:
        records, _, _ = driver.execute_query(
            "MATCH (e:KGEntity {benchmark: $b}) WITH e LIMIT 20000 DETACH DELETE e RETURN count(*) AS n", b=benchmark)
        if records[0]["n"] == 0:
            return


# ---------------------------------------------------------------- walking: what is connected, as numbered options
async def neighbours(driver, benchmark: str, question: dict, current: list[str]) -> list[dict]:
    scope = scope_of(benchmark, question)
    result = await driver.execute_query(NEIGHBOURS, keys=[key(benchmark, scope, name) for name in current])
    return [r.data() for r in result.records]


def shown_names(names: list[str]) -> str:
    text = ", ".join(names[:MAX_NAMES])
    if len(names) > MAX_NAMES:
        text += f" ... (+{len(names) - MAX_NAMES} more)"
    return text


def build_options(rows: list[dict], current: list[str], benchmark: str, question: dict) -> list[dict]:
    """Group the connections by (relation, direction). Option 0 is always STOP: "the current entities are the answer".

    MetaQA answers never lead back to the question's own entity, so it is left out of the targets."""
    leave_out = set(question["start"]) if benchmark == "metaqa" else set()
    groups: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        targets = groups.setdefault((row["relation"], row["direction"]), set())
        if row["target"] not in leave_out:
            targets.add(row["target"])
    options = [{"step": "STOP", "targets": current, "text": "STOP: the answer is: " + shown_names(current)}]
    for (relation, direction), targets in sorted(groups.items()):
        if targets:
            options.append({"step": relation + (">" if direction == "outgoing" else "<"), "relation": relation,
                            "direction": direction, "targets": sorted(targets),
                            "text": f"follow {relation} ({direction}) to: {shown_names(sorted(targets))}"})
    return options


def describe_step(current: list[str], option: dict) -> str:
    """One line of the path so far, as the models see it."""
    arrow = "->" if option["direction"] == "outgoing" else "<-"
    return (f"{{{shown_names(current)}}} {option['relation']} ({option['direction']} {arrow}) "
            f"{{{shown_names(option['targets'])}}}")
