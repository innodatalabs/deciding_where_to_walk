"""Benchmark runs: one folder per run, one subfolder per benchmark and model.

    benchmark/runs/
    └── run-001/                         a new run gets the next number, or any name you choose
        └── webqsp/
            ├── run_config.json          the benchmark, the questions and the settings of this run
            ├── questions.json           the questions themselves
            └── liquid__d1-ultra/
                ├── model.json           the model and its settings
                ├── results.jsonl        one line per question: the answer, its score and its cost
                └── decisions.jsonl      the decision log: every step, its options, and the model's answer

Running the same run name again continues where it stopped. A run name that already holds other questions or other
settings is refused, so two setups never mix. New models can be added to an existing run.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from . import graph, models, scoring, walk
from .data import BENCHMARKS
from .settings import RUNS


@dataclass
class Run:
    name: str
    benchmark: str
    questions: list[dict]

    @property
    def folder(self) -> Path:
        return RUNS / self.name / self.benchmark


def next_run_name() -> str:
    numbers = [int(m.group(1)) for p in RUNS.glob("run-*") if (m := re.fullmatch(r"run-(\d+)", p.name))]
    return f"run-{max(numbers, default=0) + 1:03d}"


def open_run(benchmark: str, questions: list[dict], name: str | None = None) -> Run:
    """Create a run, or reopen it if it already exists with the same questions and settings."""
    run = Run(name or next_run_name(), benchmark, questions)
    config = {"benchmark": benchmark, "max_steps": BENCHMARKS[benchmark].max_steps,
              "no_stop_on_unnamed": BENCHMARKS[benchmark].no_stop_on_unnamed,
              "question_ids": [q["qid"] for q in questions]}
    path = run.folder / "run_config.json"
    if path.exists():
        stored = json.loads(path.read_text())
        stored.pop("created_at", None)
        if stored != config:
            raise ValueError(f"Run {run.name!r} already has other questions or settings for {benchmark}. "
                             "Choose a new run name (or leave it empty for the next number).")
    else:
        run.folder.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config | {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=1))
        (run.folder / "questions.json").write_text(json.dumps(questions, ensure_ascii=False))
    return run


def model_folder(run: Run, model) -> Path:
    return run.folder / re.sub(r"[^\w.\-]+", "__", model.label).strip("_")


def done_questions(folder: Path) -> set[str]:
    path = folder / "results.jsonl"
    if not path.exists():
        return set()
    return {r["qid"] for r in map(json.loads, path.read_text().splitlines()) if r["stopped_by"] != "api_error"}


async def run_models(run: Run, model_list: list, concurrency: int = 8, budget_usd: float | None = None) -> None:
    """Walk every question of the run with every model. Questions already done are skipped.

    budget_usd stops a model's run once it has spent that much (the questions in flight still finish)."""
    max_steps = BENCHMARKS[run.benchmark].max_steps
    driver = graph.connect_async()
    async with models.client() as http:
        try:
            for model in model_list:
                info = await model.check(http)
                folder = model_folder(run, model)
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "model.json").write_text(json.dumps(info, indent=1))
                done = done_questions(folder)
                todo = [q for q in run.questions if q["qid"] not in done]
                queue: asyncio.Queue = asyncio.Queue()
                for q in todo:
                    queue.put_nowait(q)
                spent = 0.0
                bar = tqdm(total=len(run.questions), initial=len(done), desc=model.label, unit="question")

                async def worker():
                    nonlocal spent
                    while not queue.empty() and (budget_usd is None or spent < budget_usd):
                        question = queue.get_nowait()
                        record = await walk.walk(run.benchmark, question, model, http, driver, max_steps)
                        record |= scoring.score(run.benchmark, question, record["final"])
                        decisions = {"qid": record["qid"], "decisions": record.pop("decisions")}
                        with open(folder / "results.jsonl", "a") as f:
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        with open(folder / "decisions.jsonl", "a") as f:
                            f.write(json.dumps(decisions, ensure_ascii=False) + "\n")
                        spent += record["cost"]
                        bar.update(1)
                        bar.set_postfix(spent=f"${spent:.4f}")

                await asyncio.gather(*(worker() for _ in range(concurrency)))
                bar.close()
                if not queue.empty():
                    print(f"{model.label}: stopped at the budget of ${budget_usd}; {queue.qsize()} questions left. "
                          "Run again with a higher budget to continue.")
        finally:
            await driver.close()


# ---------------------------------------------------------------- reading results back
def load_results(benchmark: str | None = None, runs: list[str] | None = None, root: Path = RUNS) -> pd.DataFrame:
    """Every walk of every run (or the chosen runs), one row per question and model. `root` can point to another
    folder with the same layout, such as benchmark/published_results/."""
    rows = []
    for path in sorted(root.glob("*/*/*/results.jsonl")):
        run, bench, _ = path.parts[-4:-1]
        if (benchmark and bench != benchmark) or (runs and run not in runs):
            continue
        info = json.loads((path.parent / "model.json").read_text())
        label = info["slug"] + ("" if info.get("reasoning", False) is False else f" (reasoning {info['reasoning']})")
        for line in path.read_text().splitlines():
            record = json.loads(line)
            rows.append({"run": run, "benchmark": bench, "model": label, "kind": info["kind"]} | record)
    return pd.DataFrame(rows)


def load_decisions(run: str, benchmark: str, model_label: str, root: Path = RUNS) -> dict[str, list[dict]]:
    """The decision log of one model in one run, by question."""
    folder = root / run / benchmark / re.sub(r"[^\w.\-]+", "__", model_label).strip("_")
    return {r["qid"]: r["decisions"] for r in map(json.loads, (folder / "decisions.jsonl").read_text().splitlines())}
