"""The three benchmarks: where they come from, how to read them, and the small samples that ship with the repository.

Every question is a plain dict:

    {"qid": "WebQTest-0", "question": "what does jamaican people speak", "start": ["Jamaica"],
     "answers": ["Jamaican English", "Jamaican Creole English Language"]}

WebQSP and CWQ questions also carry "graph", the list of [head, relation, tail] triples of the question's own subgraph
(from the RoG release on Hugging Face), and "gold", the gold answers from the benchmark's official file, which the
scoring uses. MetaQA questions carry "hops", and all of them share one film knowledge base.
"""
from __future__ import annotations

import gzip
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from .settings import DOWNLOADS, SAMPLES


@dataclass(frozen=True)
class Benchmark:
    name: str
    title: str
    repo: str            # Hugging Face dataset
    revision: str        # pinned, so everyone gets the same files
    max_steps: int       # the longest walk allowed
    metric: str          # the benchmark's main score, see scoring.py
    split: str = ""
    # Freebase has unnamed records (shown by their ID, e.g. m.0j4jq57) that link other entities together. While the
    # walk stands only on such records, STOP is not offered, so the walk moves on to the named entities behind them.
    no_stop_on_unnamed: bool = False


BENCHMARKS = {
    "metaqa": Benchmark("metaqa", "MetaQA", "camazlucas/MetaQA", "f83854092893610686ab19f2038c039afa9bc643",
                        max_steps=4, metric="Hits@1"),
    "webqsp": Benchmark("webqsp", "WebQSP", "rmanluo/RoG-webqsp", "c0632533135a06f8c5d536b420deec5fcb5c58f3",
                        max_steps=3, metric="F1", split="test", no_stop_on_unnamed=True),
    "cwq": Benchmark("cwq", "ComplexWebQuestions", "rmanluo/RoG-cwq", "b0f6275586286312c4e99ef4b0adc65463c91f8e",
                     max_steps=4, metric="P@1", split="validation", no_stop_on_unnamed=True),
}

SAMPLE_SIZE = 200  # questions per benchmark in data/samples/

# The official question files, with the gold answers that the official scoring uses.
OFFICIAL = {
    "webqsp": ("https://download.microsoft.com/download/F/5/0/F5012144-A4FB-4084-897F-CFDA99C60BDF/WebQSP.zip",
               "WebQSP/data/WebQSP.test.json"),
    "cwq": ("https://www.dropbox.com/sh/7pkwkrfnwqhsnpo/AADH8beLbOUWxwvY_K38E3ADa/ComplexWebQuestions_dev.json?dl=1",
            "ComplexWebQuestions_dev.json"),
}


# ---------------------------------------------------------------- downloading the full benchmarks
def download(benchmark: str) -> Path:
    """Download a benchmark from Hugging Face into data/downloads/<benchmark>/ (skipped if already there)."""
    from huggingface_hub import snapshot_download

    b = BENCHMARKS[benchmark]
    patterns = (["kb/kb.txt", "*-hop/vanilla/qa_test.txt"] if benchmark == "metaqa" else [f"data/{b.split}-*.parquet"])
    folder = DOWNLOADS / benchmark
    snapshot_download(b.repo, repo_type="dataset", revision=b.revision, allow_patterns=patterns, local_dir=folder)
    if benchmark in OFFICIAL:
        url, inner = OFFICIAL[benchmark]
        target = folder / "official" / inner
        if not target.exists():
            import io
            import urllib.request
            import zipfile

            target.parent.mkdir(parents=True, exist_ok=True)
            payload = urllib.request.urlopen(url, timeout=300).read()
            if url.split("?")[0].endswith(".zip"):
                target.write_bytes(zipfile.ZipFile(io.BytesIO(payload)).read(inner))
            else:
                target.write_bytes(payload)
    return folder


def read_full(benchmark: str, folder: Path | None = None, official: Path | None = None) -> list[dict]:
    """Every test question of a benchmark, from a downloaded folder (and the official question file)."""
    folder = folder or download(benchmark)
    if benchmark == "metaqa":
        return _read_metaqa_questions(folder)
    import pandas as pd

    official = official or folder / "official" / OFFICIAL[benchmark][1]
    files = sorted((folder / "data").glob(f"{BENCHMARKS[benchmark].split}-*.parquet"))
    rog = pd.concat(pd.read_parquet(f, columns=["id", "q_entity", "graph"]) for f in files).set_index("id")
    graphs = {qid: [[str(x) for x in t] for t in g] for qid, g in rog["graph"].items()}
    if benchmark == "webqsp":
        # The official questions and topic entities. Eleven questions have no answer and no subgraph.
        return [{"qid": q["QuestionId"], "question": q["RawQuestion"], "start": [q["Parses"][0]["TopicEntityName"]],
                 "answers": [a["EntityName"] or a["AnswerArgument"] for a in q["Parses"][0]["Answers"]],
                 "gold": {"parses": [[{"id": a["AnswerArgument"], "name": a["EntityName"]} for a in p["Answers"]]
                                     for p in q["Parses"]]},
                 "graph": graphs.get(q["QuestionId"], [])}
                for q in json.loads(official.read_text(encoding="utf-8"))["Questions"]]
    gold = {q["ID"]: q for q in json.loads(official.read_text(encoding="utf-8"))}
    return [{"qid": qid, "question": gold[qid]["question"], "start": [str(e) for e in row.q_entity],
             "answers": [a["answer"] for a in gold[qid]["answers"]],
             "gold": {"answers": [{"answer": a["answer"], "aliases": a["aliases"]} for a in gold[qid]["answers"]]},
             "graph": graphs[qid]}
            for qid, row in rog.iterrows() if qid in gold]


def read_metaqa_kb(folder: Path | None = None) -> list[list[str]]:
    """The MetaQA film knowledge base as [head, relation, tail] triples."""
    if folder is None:
        path = SAMPLES / "metaqa_kb.txt.gz"
        lines = gzip.open(path, "rt", encoding="utf-8").read().splitlines()
    else:
        lines = (folder / "kb" / "kb.txt").read_text(encoding="utf-8").splitlines()
    triples = {tuple(line.split("|")) for line in lines if line.count("|") == 2}
    return [list(t) for t in sorted(triples)]


ENTITY_IN_BRACKETS = re.compile(r"\[(.+?)\]")


def _read_metaqa_questions(folder: Path) -> list[dict]:
    questions = []
    for hops in (1, 2, 3):
        path = folder / f"{hops}-hop" / "vanilla" / "qa_test.txt"
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            text, answers = line.split("\t")
            questions.append({"qid": f"metaqa-{hops}hop-test-{index}", "question": text, "hops": hops,
                              "start": [ENTITY_IN_BRACKETS.search(text).group(1)],
                              "answers": sorted(set(answers.split("|")))})
    return questions


# ---------------------------------------------------------------- the samples that ship with the repository
def make_samples(folders: dict[str, Path], officials: dict[str, Path], size: int = SAMPLE_SIZE, seed: int = 0) -> None:
    """Maintainers only: write data/samples/ from downloaded benchmarks. The order is shuffled once, with a fixed seed,
    so "the first 50 questions" are the same 50 for everyone. MetaQA takes the same number of questions per hop."""
    SAMPLES.mkdir(parents=True, exist_ok=True)
    for benchmark, folder in folders.items():
        questions = read_full(benchmark, folder, officials.get(benchmark))
        rng = random.Random(seed)
        if benchmark == "metaqa":
            by_hop = [[q for q in questions if q["hops"] == h] for h in (1, 2, 3)]
            picked = [rng.sample(group, size // 3 + (h < size % 3)) for h, group in enumerate(by_hop)]
            sample = [q for trio in zip(*picked) for q in trio]  # interleaved: 1-hop, 2-hop, 3-hop, 1-hop, ...
            sample += [q for group in picked for q in group[len(picked[-1]):]]
            kb = read_metaqa_kb(folder)
            with gzip.open(SAMPLES / "metaqa_kb.txt.gz", "wt", encoding="utf-8") as f:
                f.write("\n".join("|".join(t) for t in kb))
        else:
            sample = rng.sample(questions, size)
        with gzip.open(SAMPLES / f"{benchmark}.jsonl.gz", "wt", encoding="utf-8") as f:
            for q in sample:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")


def read_sample(benchmark: str) -> list[dict]:
    path = SAMPLES / f"{benchmark}.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def questions_for(benchmark: str, size: int | None = 50, source: str = "sample", seed: int = 0) -> list[dict]:
    """The questions for a run.

    source="sample": the first `size` questions of the sample in data/samples/ (at most 200), no download needed.
    source="huggingface": the full benchmark, downloaded; `size` questions picked at random with `seed`, or all of them.
    """
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark {benchmark!r}. Choose one of {list(BENCHMARKS)}.")
    if source == "sample":
        sample = read_sample(benchmark)
        if size is not None and size > len(sample):
            raise ValueError(f"The sample has {len(sample)} questions. Use source='huggingface' for more.")
        return sample[:size] if size else sample
    if source == "huggingface":
        questions = read_full(benchmark)
        return questions if size is None else random.Random(seed).sample(questions, size)
    raise ValueError("source must be 'sample' or 'huggingface'")
