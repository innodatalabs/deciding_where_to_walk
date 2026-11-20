"""BioHopR: biomedical questions answered by walking PrimeKG.

PrimeKG (Chandak et al., 2023; CC0) is one biomedical knowledge graph that every question shares, like MetaQA's film
knowledge base. BioHopR (Kim et al., 2025) asks one-step and two-step questions about its drugs, genes and proteins,
diseases and phenotypes, such as "Name all diseases that are related to a effect/phenotype that is associated with
drug Benzyl benzoate."

A question's right answers ("answers") are everything its relations reach in PrimeKG v2.1, through every middle entity
on two-step questions. Each pair of node types has one relation (RELATION). BioHopR's own answer list ("official")
seems to come from an earlier PrimeKG release, so it sometimes differs; benchmark/RESULTS.md says how often.

Self-check: python -m utils.biohopr
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import urllib.request
from collections import defaultdict
from pathlib import Path

from .data import BENCHMARKS
from .settings import DOWNLOADS

FOLDER = DOWNLOADS / "biohopr"
DATAVERSE = "https://dataverse.harvard.edu/api/access/datafile/{}"
# PrimeKG v2.1 on Harvard Dataverse (doi:10.7910/DVN/IXA7BM): file name -> (file id, MD5)
PRIMEKG_FILES = {"edges.csv": (6180616, "5d4d211a22e88544b78fde2735e797bc"),
                 "nodes.tab": (6180617, "4924de04fb3deefa1e0a8dada424538e")}
PRIMEKG_SIZE = {"entities": 129375, "facts": 4050064}  # facts: undirected edges, each stored once
BIOHOPR_FILE = "BioHopR.json"

# The PrimeKG relation behind each pair of node types, as BioHopR's answer lists and wording show: "treat" is
# indication, "side effects" drug_effect, "phenotypes" disease_phenotype_positive.
RELATION = {frozenset(pair): relation for pair, relation in [
    (("drug", "disease"), "indication"), (("drug", "gene/protein"), "drug_protein"),
    (("gene/protein", "disease"), "disease_protein"), (("drug", "effect/phenotype"), "drug_effect"),
    (("disease", "effect/phenotype"), "disease_phenotype_positive")]}


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(folder: Path = FOLDER) -> Path:
    """Download PrimeKG (about 400 MB) and BioHopR into data/downloads/biohopr/ (skipped if already there)."""
    from huggingface_hub import hf_hub_download

    folder.mkdir(parents=True, exist_ok=True)
    for name, (file_id, expected) in PRIMEKG_FILES.items():
        target = folder / name
        if not target.exists():
            print(f"Downloading PrimeKG {name} from Harvard Dataverse...")
            partial = target.with_suffix(".part")
            # Dataverse refuses Python's default User-Agent
            request = urllib.request.Request(DATAVERSE.format(file_id), headers={"User-Agent": "deciding-where-to-walk"})
            with urllib.request.urlopen(request, timeout=600) as response, open(partial, "wb") as f:
                shutil.copyfileobj(response, f, 1 << 20)
            partial.rename(target)
        if md5(target) != expected:
            raise ValueError(f"{target} is not the expected PrimeKG file (MD5 mismatch). Delete it and download again.")
    b = BENCHMARKS["biohopr"]
    hf_hub_download(b.repo, BIOHOPR_FILE, repo_type="dataset", revision=b.revision, local_dir=folder)
    return folder


def read_primekg(folder: Path = FOLDER):
    """PrimeKG's nodes (columns index, type, name, display) and edges (x, y, relation), each edge once: PrimeKG lists
    every edge both ways. `display` is a unique name: a name that several nodes share gets the node's type appended,
    and its index too where that is still not unique (PrimeKG already has names such as "fever (disease)")."""
    import numpy as np
    import pandas as pd

    nodes = pd.read_csv(folder / "nodes.tab", sep="\t", usecols=["node_index", "node_type", "node_name"],
                        keep_default_na=False)  # a gene named NA stays "NA"
    nodes = nodes.rename(columns={"node_index": "index", "node_type": "type", "node_name": "name"})
    nodes["name"] = nodes["name"].str.strip()
    shared = nodes["name"].duplicated(keep=False)
    display = nodes["name"].where(~shared, nodes["name"] + " (" + nodes["type"] + ")")
    clash = shared & display.duplicated(keep=False)
    nodes["display"] = display.where(~clash, display + " [" + nodes["index"].astype(str) + "]")
    assert nodes["display"].is_unique
    edges = pd.read_csv(folder / "edges.csv", usecols=["relation", "x_index", "y_index"],
                        dtype={"relation": "category", "x_index": "int32", "y_index": "int32"})
    x, y = edges["x_index"].to_numpy(), edges["y_index"].to_numpy()
    edges = pd.DataFrame({"x": np.minimum(x, y), "y": np.maximum(x, y), "relation": edges["relation"]})
    return nodes, edges.drop_duplicates(ignore_index=True)


def questions(folder: Path = FOLDER) -> list[dict]:
    """The 2,494 one-step questions (one per middle entity and relation of BioHopR's rows), then the 7,633 two-step
    questions (one per row). Answers are display names, so scoring compares plain sets."""
    nodes, edges = read_primekg(folder)
    node = dict(zip(zip(nodes["name"], nodes["type"]), nodes["index"]))
    display = dict(zip(nodes["index"], nodes["display"]))
    wanted = edges[edges["relation"].isin(set(RELATION.values()))]
    linked = defaultdict(set)
    for x, y, relation in zip(wanted["x"], wanted["y"], wanted["relation"]):
        linked[x, relation].add(y)
        linked[y, relation].add(x)

    def reach(name: str, path: str) -> list[str]:
        kinds = path.split(":")
        current = {node[name.strip(), kinds[0]]}
        for a, b in zip(kinds, kinds[1:]):
            current = {n for c in current for n in linked[c, RELATION[frozenset((a, b))]]}
        return sorted(display[n] for n in current)

    def named(answers: set[str], kind: str) -> list[str]:
        # 43 of BioHopR's names lack the " (disease)" that PrimeKG v2.1 adds, such as "diabetes mellitus"
        return sorted({display[node[a, kind] if (a, kind) in node else node[f"{a} ({kind})", kind]] for a in answers})

    rows = json.loads((folder / BIOHOPR_FILE).read_text(encoding="utf-8"))
    one, two = {}, []
    for i, r in enumerate(rows):
        answers = {a.strip() for a in r["answer"]}
        k = (r["hop1"].strip(), r["relation_hop1"])
        if k in one:
            # Rows agree on a middle entity's answers, except a two-step row through several drugs lists them all
            one[k]["official"] &= answers
        else:
            one[k] = {"question": r["hop1_question_multi"], "official": answers}
        two.append({"qid": f"biohopr-2hop-{i}", "question": r["hop2_question_multi"], "hops": 2,
                    "start": [display[node[r["hop2"].strip(), r["hop2_type"]]]],
                    "answers": reach(r["hop2"], r["relation_hop2"]), "official": named(answers, r["target_type"]),
                    "path": r["relation_hop2"]})
    first = [{"qid": f"biohopr-1hop-{n}", "question": q["question"], "hops": 1,
              "start": [display[node[name, path.split(":")[0]]]], "answers": reach(name, path),
              "official": named(q["official"], path.split(":")[1]), "path": path}
             for n, ((name, path), q) in enumerate(one.items())]
    return first + two


def stratified(questions: list[dict], per_hop: int, seed: int = 0) -> list[dict]:
    """per_hop one-step and per_hop two-step questions, picked at random with the seed and interleaved (one-step,
    two-step, one-step, ...). A smaller pick is the start of a larger one."""
    rng = random.Random(seed)
    groups = [[q for q in questions if q["hops"] == hops] for hops in (1, 2)]
    for group in groups:
        rng.shuffle(group)
    return [q for pair in zip(groups[0][:per_hop], groups[1][:per_hop]) for q in pair]


# ponytail: a copy of paired_test in utils/self_aware.py (branch self-aware-walk); import it from there once merged.
def paired_test(a, b) -> dict:
    """The mean of a - b over the questions that both have, with a 95% interval and a two-sided p-value. A paired
    z-test; on yes/no outcomes it is close to McNemar's test."""
    # ponytail: normal approximation, fine for the hundreds of questions a test needs; too rough for a tiny pilot
    d = (a - b).dropna()
    mean, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
    p = math.erfc(abs(mean / se) / math.sqrt(2)) if se > 0 else float(mean == 0)
    return {"questions": len(d), "difference": mean, "low": mean - 1.96 * se, "high": mean + 1.96 * se, "p_value": p}


# ---------------------------------------------------------------- self-check: python -m utils.biohopr
def _check() -> None:
    """A seven-node PrimeKG and three BioHopR rows, written to a temporary folder and read back."""
    import tempfile

    from . import graph, scoring

    nodes = [("drug", "Aspirin"), ("gene/protein", "PTGS1"), ("gene/protein", "PTGS2"),
             ("disease", "migraine (disease)"), ("disease", "fever"), ("effect/phenotype", "fever"),
             ("disease", "fever (disease)")]
    edges = [("drug_protein", 0, 1), ("drug_protein", 0, 2), ("disease_protein", 1, 3), ("disease_protein", 2, 4),
             ("protein_protein", 1, 2), ("indication", 0, 6), ("disease_phenotype_positive", 6, 5)]

    def row(start, middle, path, answer):
        kinds = path.split(":")
        return {"hop2": start, "hop2_type": kinds[0], "hop1": middle, "hop1_type": kinds[1], "target_type": kinds[2],
                "relation_hop2": path, "relation_hop1": ":".join(kinds[1:]), "answer": answer,
                "hop2_question_multi": f"two-step from {start}", "hop1_question_multi": f"one-step from {middle}"}

    rows = [row("Aspirin", "PTGS1", "drug:gene/protein:disease", ["migraine "]),
            row("PTGS2", "Aspirin", "gene/protein:drug:disease", ["fever (disease)"]),
            row("PTGS1", "Aspirin", "gene/protein:drug:disease", ["fever (disease)", "migraine"])]
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder / "nodes.tab").write_text("node_index\tnode_id\tnode_type\tnode_name\tnode_source\n" + "".join(
            f"{i}\tid{i}\t{t}\t{n}\tsource\n" for i, (t, n) in enumerate(nodes)))
        (folder / "edges.csv").write_text("relation,display_relation,x_index,y_index\n" + "".join(
            f"{r},{r},{x},{y}\n{r},{r},{y},{x}\n" for r, x, y in edges))  # both ways, as PrimeKG stores them
        (folder / BIOHOPR_FILE).write_text(json.dumps(rows))
        read_nodes, read_edges = read_primekg(folder)
        qs = {q["qid"]: q for q in questions(folder)}

    assert read_nodes["display"].tolist() == ["Aspirin", "PTGS1", "PTGS2", "migraine (disease)",
                                              "fever (disease) [4]", "fever (effect/phenotype)", "fever (disease)"]
    assert len(read_edges) == len(edges)
    assert list(qs) == ["biohopr-1hop-0", "biohopr-1hop-1", "biohopr-2hop-0", "biohopr-2hop-1", "biohopr-2hop-2"]
    # two-step: through both PTGS1 and PTGS2, while BioHopR's list covers PTGS1 only (and lacks " (disease)")
    assert qs["biohopr-2hop-0"]["answers"] == ["fever (disease) [4]", "migraine (disease)"]
    assert qs["biohopr-2hop-0"]["official"] == ["migraine (disease)"]
    assert qs["biohopr-2hop-2"]["answers"] == ["fever (disease)"]
    assert qs["biohopr-2hop-2"]["official"] == ["fever (disease)", "migraine (disease)"]
    # one-step: Aspirin's rows disagree, so its official list keeps what both list
    assert qs["biohopr-1hop-1"] == {"qid": "biohopr-1hop-1", "question": "one-step from Aspirin", "hops": 1,
                                    "start": ["Aspirin"], "answers": ["fever (disease)"],
                                    "official": ["fever (disease)"], "path": "drug:disease"}
    assert [q["hops"] for q in stratified(list(qs.values()), 2)] == [1, 2, 1, 2]
    assert stratified(list(qs.values()), 2)[:2] == stratified(list(qs.values()), 1)

    q = qs["biohopr-2hop-0"]
    assert scoring.score("biohopr", q, ["fever (disease) [4]", "migraine (disease)"]) == {
        "score": 1.0, "exact": True, "recall": 1.0, "f1": 1.0, "precision_official": 0.5, "exact_official": False}
    s = scoring.score("biohopr", q, ["migraine (disease)", "PTGS2"])
    assert (s["score"], s["recall"], s["f1"], s["precision_official"], s["exact"]) == (0.5, 0.5, 0.5, 0.5, False)
    assert scoring.score("biohopr", q, [])["score"] == 0.0

    # undirected: one option per relation, whichever way the edge was stored
    neighbour_rows = [{"relation": "drug_protein", "direction": "outgoing", "target": "PTGS1"},
                      {"relation": "drug_protein", "direction": "incoming", "target": "PTGS2"},
                      {"relation": "indication", "direction": "incoming", "target": "fever (disease)"}]
    options = graph.build_options(neighbour_rows, ["Aspirin"], "biohopr", qs["biohopr-1hop-1"])
    assert [o["step"] for o in options] == ["STOP", "drug_protein", "indication"]
    assert options[1]["text"] == "follow drug_protein to: PTGS1, PTGS2"
    assert graph.describe_step(["Aspirin"], options[1]) == "{Aspirin} drug_protein (--) {PTGS1, PTGS2}"
    print("biohopr self-check passed")


if __name__ == "__main__":
    _check()
