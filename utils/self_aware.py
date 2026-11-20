"""Testing a self-aware walk: a walk that uses the decision model's probabilities to notice how sure it is.

A self-aware walk can:

- abstain: give no answer when the whole walk's confidence is low;
- pick: when the walk ends on several entities, ask which one answers the question, and keep it alone if sure;
- escalate: send an unsure step to a stronger model, and follow that model's choice instead.

Escalating more always costs more, abstaining more always leaves fewer, more accurate answers, and cutting an answer
always changes its score, so better numbers alone prove nothing. The probabilities make a difference only if the
self-aware walk beats doing the same thing at random: abstaining on the same share of questions, cutting the same
answers, or escalating the same share of steps. Every test here makes that comparison.

Checks on a run you already made (benchmark/2_run_benchmark.ipynb), with no new calls:
    walk_signals, resolution, signal_table    is there anything to put a threshold on, and do the probabilities
                                              know when the walk is wrong?
    risk_coverage, abstention, calibration    abstaining by confidence, against random and an oracle

Picking the answer, with one more call per walk that ended on several entities (no graph needed):
    pick_answers, pick_outcomes    ask the decision model which entity answers, keep it alone if it is sure, and
                                   score that against the walk's answer, cutting at random, and an oracle

Escalation, on disagreement trees:
    grow_trees, load_trees    at every step both models choose, and the walk follows each different choice, so a
                              tree holds every walk that any escalation policy could make on its question
    outcomes, sweep, cross_fit    score any policy on the trees, without new calls and with no run-to-run noise
                                  between policies, because they all reuse the same answers
    headroom                  the most that any router between the two models could gain

Statistics: bootstrap, paired_test, holm, questions_needed.

Self-check, on a small fake graph with fake models (no API key or Neo4j needed): uv run python -m utils.self_aware
"""
from __future__ import annotations

import asyncio
import json
import math
import random
import re
from dataclasses import asdict
from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from . import graph, models, scoring, walk
from .data import BENCHMARKS
from .runs import Run, folder_name
from .settings import RUNS


# ---------------------------------------------------------------- how sure the decision model was
def confidence(calls: list[dict]) -> float:
    """How sure a decision model was of one step: the probability it gave its own choice (the lowest over the rounds
    of a step with more than 255 options). 0 if a call failed, and 1 for a step that needed no model."""
    if not calls:
        return 1.0
    if any(c["error"] or not c["probabilities"] for c in calls):
        return 0.0
    return min(c["probabilities"].get(c["option"], 0.0) for c in calls)


def walk_confidence(decisions: list[dict], how: str = "min") -> float:
    """How sure the decision model was of a whole walk: its lowest step confidence, or ("product") all of them
    multiplied."""
    steps = [confidence(d["calls"]) for d in decisions if d["calls"]]
    if not steps:
        return 1.0
    return min(steps) if how == "min" else math.prod(steps)


# ---------------------------------------------------------------- checks on a run you already made
def walk_signals(results: pd.DataFrame, decisions: dict[str, list[dict]]) -> pd.DataFrame:
    """One row per walk of one model (`results` from load_results, `decisions` from load_decisions): its score,
    whether it was exactly right, its confidence, and signals that every walk gives for free. Walks lost to an API
    error are left out."""
    scored = results[results["score"].notna()]
    return pd.DataFrame({
        "score": scored["score"].to_numpy(float),
        "exact": scored["exact"].astype(float).to_numpy(),
        "confidence": [walk_confidence(decisions[q]) for q in scored["qid"]],
        "product": [walk_confidence(decisions[q], "product") for q in scored["qid"]],
        "steps": scored["steps"].map(len).to_numpy(),
        "answer size": scored["final"].map(len).to_numpy(),
        "hit step limit": (scored["stopped_by"] == "step_limit").to_numpy(float),
    }, index=pd.Index(scored["qid"], name="qid"))


def resolution(decisions: dict[str, list[dict]]) -> pd.Series:
    """How the decision model's confidence is spread over all its steps. A threshold can only act on the steps below
    1.00, so if most steps sit at 1.00 there is little for it to work with."""
    p = pd.Series([confidence(d["calls"]) for steps in decisions.values() for d in steps if d["calls"]])
    return pd.Series({"model steps": len(p), "at 1.00": (p >= 1).mean(),
                      "0.90 to 0.99": p.between(0.9, 1, inclusive="left").mean(),
                      "0.50 to 0.89": p.between(0.5, 0.9, inclusive="left").mean(), "below 0.50": (p < 0.5).mean()})


def auroc(signal, outcome) -> float:
    """How well a signal ranks right answers (outcome 1) above wrong ones (outcome 0): 0.5 is chance, 1 is perfect.
    Ties count half."""
    ranks = pd.Series(np.asarray(signal, float)).rank().to_numpy()
    right = np.asarray(outcome, float) == 1
    n_right, n_wrong = right.sum(), (~right).sum()
    if not n_right or not n_wrong:
        return float("nan")
    return float((ranks[right].sum() - n_right * (n_right + 1) / 2) / (n_right * n_wrong))


FREE_SIGNALS = {  # signals that every walk gives anyway, each pointing the same way: higher means "more likely right"
    "fewer steps": lambda f: -f["steps"],
    "smaller answer": lambda f: -f["answer size"],
    "did not hit the step limit": lambda f: -f["hit step limit"],
}


def signal_table(signals: pd.DataFrame) -> pd.DataFrame:
    """AUROC for predicting "exactly correct", of the confidence and of the free signals, with 95% intervals, and how
    much better the confidence (lowest step) is than each. If it is not better, the probabilities add nothing."""
    candidates = {"confidence (lowest step)": lambda f: f["confidence"],
                  "confidence (product of steps)": lambda f: f["product"], **FREE_SIGNALS}
    rows = []
    for name, get in candidates.items():
        auc = bootstrap(signals, lambda f: auroc(get(f), f["exact"]))
        gain = bootstrap(signals, lambda f: auroc(f["confidence"], f["exact"]) - auroc(get(f), f["exact"]))
        rows.append({"signal": name, "AUROC": auc["estimate"], "low": auc["low"], "high": auc["high"],
                     "confidence is better by": gain["estimate"], "by, low": gain["low"], "by, high": gain["high"]})
    return pd.DataFrame(rows).set_index("signal").round(3)


def risk_coverage(signal, score) -> pd.Series:
    """The error of the answers kept (1 - score), for every share of answers kept, most confident first. Walks with
    the same signal are kept together, so each counts as their average and their order does not matter."""
    f = pd.DataFrame({"signal": np.asarray(signal, float), "score": np.asarray(score, float)})
    f["score"] = f.groupby("signal")["score"].transform("mean")
    error = 1 - f.sort_values("signal", ascending=False)["score"].to_numpy()
    kept = np.arange(1, len(error) + 1)
    return pd.Series(np.cumsum(error) / kept, index=kept / len(error), name="error of the answers kept")


def abstention(signals: pd.DataFrame, signal: str = "confidence", outcome: str = "score") -> pd.Series:
    """Abstaining by confidence, against abstaining at random and an oracle that drops exactly the wrong answers. Each
    is the area under its risk-coverage curve (lower is better). "gap closed" is how far the confidence gets from
    random towards the oracle: 0 is no better than random, 1 is as good as the oracle."""
    def areas(f):
        return (risk_coverage(f[signal], f[outcome]).mean(), 1 - f[outcome].mean(),
                risk_coverage(f[outcome], f[outcome]).mean())

    def gap_closed(f):
        by_signal, at_random, oracle = areas(f)
        return (at_random - by_signal) / (at_random - oracle) if at_random > oracle else float("nan")

    by_signal, at_random, oracle = areas(signals)
    gap = bootstrap(signals, gap_closed)
    return pd.Series({"area, by confidence": by_signal, "area, at random": at_random, "area, oracle": oracle,
                      "gap closed": gap["estimate"], "gap closed, low": gap["low"],
                      "gap closed, high": gap["high"]}).round(4)


def calibration(signals: pd.DataFrame, signal: str = "confidence") -> tuple[pd.DataFrame, float]:
    """Walks grouped by confidence (0.0 to 0.1, ..., 0.9 to 1.0, and exactly 1.00), with how often each group was
    exactly right, and the expected calibration error: the average gap between the two, weighted by walks. Abstaining
    only needs the ranking; one threshold that works on every benchmark also needs calibration."""
    group = (np.floor(signals[signal] * 10) / 10).rename("confidence from")
    table = signals.groupby(group).agg(walks=("exact", "size"), confidence=(signal, "mean"),
                                       exactly_correct=("exact", "mean"))
    ece = float((table["walks"] * (table["confidence"] - table["exactly_correct"]).abs()).sum() / table["walks"].sum())
    return table, ece


# ---------------------------------------------------------------- disagreement trees
def outcome(pick: int | None, calls: list[dict]) -> str:
    """Where a model's choice leads: the chosen option's number, or "invalid" or "api_error" if a call failed."""
    error = next((c["error"] for c in calls if c["error"]), None)
    if error:
        return "invalid" if error == "invalid_choice" else "api_error"
    return str(pick)


async def grow(benchmark: str, question: dict, dm, strong, http, driver, max_steps: int,
               threshold: float | None = None) -> dict:
    """One question's disagreement tree. At every step both models choose; where they agree the walk goes on once,
    and where they disagree it follows both choices. Every step works as in walk.walk(), so following the decision
    model's choices everywhere gives the decision model's own walk, and following the strong model's gives its walk.

    With a threshold, only that self-aware walk is grown: the strong model is asked only where the decision model's
    confidence is below the threshold, and only the choice that is followed is walked on. This is the real run of the
    policy, in the same shape, so it is scored the same way.

    A decision point holds the step ("hop", "current", "options"), both models' calls, where each choice leads
    ("next": an option number, "invalid" or "api_error") and the "children" by that key. An end ("leaf") holds how
    the walk stopped, its answer ("final"), its score, and whether it stood on the same entities twice ("looped")."""
    # ponytail: a tree doubles at each disagreement, so it has at most 2**max_steps ends; fine for two models
    def leaf(stopped_by: str, final: list[str] | None, looped: bool) -> dict:
        return {"leaf": True, "stopped_by": stopped_by, "final": final, "looped": looped} | scoring.score(
            benchmark, question, final)

    async def step(current: list[str], path: list[str], hop: int, seen: frozenset, looped: bool) -> dict:
        if hop == max_steps:
            return leaf("step_limit", current, looped)
        rows = await graph.neighbours(driver, benchmark, question, current)
        options = graph.build_options(rows, current, benchmark, question)
        if len(options) == 1:  # nothing to follow: STOP is the only option
            return leaf("nothing_to_follow", current, looped)
        state = {"question": question["question"], "current_entities": current, "path_so_far": path, "hop": hop}
        offer_stop = not (BENCHMARKS[benchmark].no_stop_on_unnamed and walk.all_unnamed(current))
        node = {"hop": hop, "current": current, "options": [o["text"] for o in options], "calls": {"dm": [], "strong": []}}
        if not offer_stop and len(options) == 2:  # one connection left: followed without asking a model
            node["next"] = {"dm": "1", "strong": "1"}
        else:
            def ask(model):
                return walk.decide(model, http, state, options, offer_stop)

            strong_pick = None
            if threshold is None:
                (dm_pick, node["calls"]["dm"]), (strong_pick, node["calls"]["strong"]) = await asyncio.gather(
                    ask(dm), ask(strong))
            else:
                dm_pick, node["calls"]["dm"] = await ask(dm)
                if confidence(node["calls"]["dm"]) < threshold:
                    strong_pick, node["calls"]["strong"] = await ask(strong)
            node["next"] = {"dm": outcome(dm_pick, node["calls"]["dm"]),
                            "strong": outcome(strong_pick, node["calls"]["strong"]) if node["calls"]["strong"] else None}
        if threshold is None or not node["calls"]["dm"]:
            keys = sorted({key for key in node["next"].values() if key is not None})
        else:
            keys = [node["next"]["strong"] if node["calls"]["strong"] else node["next"]["dm"]]

        async def follow(key: str) -> dict:
            if key in ("invalid", "api_error"):
                return leaf("invalid_choice", [], looped) if key == "invalid" else leaf("api_error", None, looped)
            option = options[int(key)]
            if option["step"] == "STOP":
                return leaf("model_stop", current, looped)
            here = seen | {frozenset(current)}
            return await step(option["targets"], path + [graph.describe_step(current, option)], hop + 1, here,
                              looped or frozenset(option["targets"]) in here)

        node["children"] = dict(zip(keys, await asyncio.gather(*(follow(key) for key in keys))))
        return node

    return await step(list(question["start"]), [], 0, frozenset(), False)


def nodes(tree: dict):
    """Every point of a tree: its decision points and its ends."""
    yield tree
    for child in tree.get("children", {}).values():
        yield from nodes(child)


def calls_cost(calls: list[dict]) -> float:
    return sum(c["cost"] or 0 for c in calls)


def tree_cost(tree: dict) -> float:
    return sum(calls_cost(n["calls"]["dm"]) + calls_cost(n["calls"]["strong"]) for n in nodes(tree) if not n.get("leaf"))


def tree_folder(benchmark_folder: Path, dm_label: str, strong_label: str, threshold: float | None = None) -> Path:
    name = f"{folder_name(dm_label)}__vs__{folder_name(strong_label)}"
    return benchmark_folder / "trees" / (name if threshold is None else f"{name}__threshold-{threshold}")


async def grow_trees(run: Run, dm, strong, threshold: float | None = None, concurrency: int = 8,
                     budget_usd: float | None = None) -> Path:
    """Grow the tree of every question of the run, into benchmark/runs/<run>/<benchmark>/trees/. Questions already
    done are skipped, and questions lost to an API error are tried again. With a threshold, this is the real run of
    that self-aware walk (see grow). budget_usd stops once that much is spent; questions in flight still finish."""
    folder = tree_folder(run.folder, dm.label, strong.label, threshold)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "trees.jsonl"
    done = {r["qid"] for r in map(json.loads, path.read_text().splitlines()) if not r["failed"]} if path.exists() else set()
    queue: asyncio.Queue = asyncio.Queue()
    for question in run.questions:
        if question["qid"] not in done:
            queue.put_nowait(question)
    spent = 0.0
    driver = graph.connect_async()
    try:
        async with models.client() as http:
            info = {"decision model": await dm.check(http), "strong model": await strong.check(http),
                    "threshold": threshold}
            (folder / "models.json").write_text(json.dumps(info, indent=1))
            bar = tqdm(total=len(run.questions), initial=len(done), desc=folder.name, unit="question")

            async def worker():
                nonlocal spent
                while not queue.empty() and (budget_usd is None or spent < budget_usd):
                    question = queue.get_nowait()
                    tree = await grow(run.benchmark, question, dm, strong, http, driver,
                                      BENCHMARKS[run.benchmark].max_steps, threshold)
                    record = {"qid": question["qid"], "failed": any(n.get("stopped_by") == "api_error" for n in nodes(tree)),
                              "cost": tree_cost(tree), "tree": tree}
                    with open(path, "a") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    spent += record["cost"]
                    bar.update(1)
                    bar.set_postfix(spent=f"${spent:.4f}")

            await asyncio.gather(*(worker() for _ in range(concurrency)))
            bar.close()
    finally:
        await driver.close()
    if not queue.empty():
        print(f"Stopped at the budget of ${budget_usd}; {queue.qsize()} questions left. Run again to continue.")
    return path


def load_trees(run: str, benchmark: str, dm_label: str, strong_label: str, threshold: float | None = None,
               root: Path = RUNS) -> dict[str, dict]:
    """The trees of one run, by question. Questions whose tree holds an API error are left out, so every policy is
    scored on the same questions. A tree grown with a threshold can only be scored with that threshold."""
    path = tree_folder(root / run / benchmark, dm_label, strong_label, threshold) / "trees.jsonl"
    records = {r["qid"]: r for r in map(json.loads, path.read_text().splitlines())}  # a question tried again: last wins
    trees = {qid: r["tree"] for qid, r in records.items() if not r["failed"]}
    if len(trees) < len(records):
        print(f"{len(records) - len(trees)} of {len(records)} questions left out after an API error")
    return trees


# ---------------------------------------------------------------- scoring escalation policies on the trees
def threshold_policy(threshold: float):
    """Escalate a step when the decision model's confidence in its choice is below the threshold. 0 never
    escalates; anything above 1 escalates every step."""
    return lambda node: float(confidence(node["calls"]["dm"]) < threshold)


def random_policy(rate: float):
    """Escalate each step with probability `rate`. It is scored exactly, as an expectation, not by sampling."""
    return lambda node: rate


def evaluate(node: dict, escalate) -> dict:
    """The expected outcome of one policy on one tree. `escalate(node)` is the probability that the policy sends this
    step to the strong model. The decision model is paid at every step, since the policy needs its probabilities; the
    strong model only where the step is escalated."""
    if node.get("leaf"):
        return {"score": node["score"], "exact": float(node["exact"]), "dm_cost": 0.0, "strong_cost": 0.0,
                "decisions": 0.0, "escalated": 0.0, "step_limit": float(node["stopped_by"] == "step_limit"),
                "looped": float(node["looped"])}
    if not node["calls"]["dm"]:  # followed without asking a model
        return evaluate(node["children"][node["next"]["dm"]], escalate)
    p = escalate(node)
    weights: dict[str, float] = {}
    for key, weight in ((node["next"]["dm"], 1 - p), (node["next"]["strong"], p)):
        if weight > 0:
            weights[key] = weights.get(key, 0.0) + weight
    parts = [(weight, evaluate(node["children"][key], escalate)) for key, weight in weights.items()]
    out = {k: sum(weight * part[k] for weight, part in parts) for k in parts[0][1]}
    out["dm_cost"] += calls_cost(node["calls"]["dm"])
    out["strong_cost"] += p * calls_cost(node["calls"]["strong"])
    out["decisions"] += 1
    out["escalated"] += p
    return out


def outcomes(trees: dict[str, dict], escalate) -> pd.DataFrame:
    """Each question's expected outcome under one policy."""
    return pd.DataFrame.from_dict({qid: evaluate(tree, escalate) for qid, tree in trees.items()},
                                  orient="index").rename_axis("qid")


def policy_summary(frame: pd.DataFrame, pay_dm: bool = True) -> dict:
    """A policy's results over the questions. pay_dm=False for the strong model on its own."""
    decisions = frame["decisions"].sum()
    cost = frame["strong_cost"] + (frame["dm_cost"] if pay_dm else 0)
    return {"escalated steps": frame["escalated"].sum() / decisions if decisions else 0.0,
            "score": frame["score"].mean(), "exactly correct": frame["exact"].mean(),
            "$ per 1,000 questions": 1000 * cost.mean(),
            "hit step limit": frame["step_limit"].mean(), "looped": frame["looped"].mean()}


THRESHOLDS = [i / 100 for i in range(102)]  # 0.00 escalates nothing, 1.01 escalates every step


def sweep(trees: dict[str, dict], thresholds=THRESHOLDS) -> pd.DataFrame:
    """Every threshold, random escalation of exactly the same shares of steps, and the strong model on its own."""
    rows = [{"policy": "threshold", "threshold": t} | policy_summary(outcomes(trees, threshold_policy(t)))
            for t in thresholds]
    for rate in sorted({row["escalated steps"] for row in rows}):
        rows.append({"policy": "random", "threshold": None} | policy_summary(outcomes(trees, random_policy(rate))))
    rows.append({"policy": "strong model alone", "threshold": None}
                | policy_summary(outcomes(trees, random_policy(1)), pay_dm=False))
    return pd.DataFrame(rows)


def plot_sweep(table: pd.DataFrame) -> None:
    """Score against the share of steps escalated, and against cost. The probabilities make a difference only where
    the threshold line is above the random line."""
    _, axes = plt.subplots(1, 2, figsize=(11, 4))
    for policy, style in (("threshold", "o-"), ("random", "s--")):
        rows = table[table["policy"] == policy]
        for ax, x in zip(axes, ("escalated steps", "$ per 1,000 questions")):
            line = rows.sort_values(x)
            ax.plot(line[x], line["score"], style, markersize=3, label=f"{policy} escalation")
    alone = table[table["policy"] == "strong model alone"].iloc[0]
    axes[1].plot(alone["$ per 1,000 questions"], alone["score"], "k*", markersize=10, label="strong model alone")
    axes[0].set(xlabel="share of steps sent to the strong model", ylabel="score")
    axes[1].set(xlabel="US$ per 1,000 questions", ylabel="score")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.show()


def pick_threshold(trees: dict[str, dict], max_escalated: float) -> float:
    """The highest threshold that sends at most `max_escalated` of the steps to the strong model on these questions."""
    picked = 0.0
    for t in THRESHOLDS:
        if policy_summary(outcomes(trees, threshold_policy(t)))["escalated steps"] <= max_escalated:
            picked = t
    return picked


def cross_fit(trees: dict[str, dict], max_escalated: float, seed: int = 0,
              tune_size: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The main test. Split the questions into two halves, pick the threshold on one half (on at most `tune_size` of
    its questions) and score it on the other, then swap. So every question is scored with a threshold picked without
    it. Each half is also scored with random escalation of exactly the share of steps the threshold escalated there.

    Returns each question's outcome (the columns ending in "_random" are random escalation), and the threshold picked
    on each half, with the share of steps it escalated on the half it was picked on and on the half it was tested on."""
    qids = sorted(trees)
    random.Random(seed).shuffle(qids)
    halves = [qids[0::2], qids[1::2]]
    scored, picks = [], []
    for tune, test in (halves, halves[::-1]):
        tuning = {q: trees[q] for q in tune[:tune_size]}
        threshold = pick_threshold(tuning, max_escalated)
        testing = {q: trees[q] for q in test}
        policy = outcomes(testing, threshold_policy(threshold))
        rate = policy_summary(policy)["escalated steps"]
        scored.append(policy.join(outcomes(testing, random_policy(rate)), rsuffix="_random"))
        picks.append({"threshold": threshold, "tuning questions": len(tuning),
                      "escalated, half it was picked on": policy_summary(
                          outcomes(tuning, threshold_policy(threshold)))["escalated steps"],
                      "escalated, half it was tested on": rate})
    return pd.concat(scored), pd.DataFrame(picks)


def headroom(dm: pd.DataFrame, strong: pd.DataFrame) -> pd.Series:
    """The most that any router between the two models could gain: on each question, the better of their two
    answers. Also the shares of questions where only one of them was exactly right. Takes two frames with "score" and
    "exact" by question, such as outcomes() of each model alone, or walk_signals() of two runs."""
    both = dm[["score", "exact"]].join(strong[["score", "exact"]], lsuffix="_dm", rsuffix="_strong", how="inner")
    return pd.Series({
        "questions": len(both),
        "decision model": both["score_dm"].mean(),
        "strong model": both["score_strong"].mean(),
        "better of the two": both[["score_dm", "score_strong"]].max(axis=1).mean(),
        "only the decision model exactly right": ((both["exact_dm"] == 1) & (both["exact_strong"] == 0)).mean(),
        "only the strong model exactly right": ((both["exact_dm"] == 0) & (both["exact_strong"] == 1)).mean(),
    }).round(4)


# ---------------------------------------------------------------- picking the answer
PICK_INSTRUCTIONS = (
    "We walked a knowledge graph to answer the question in the state. The walk has ended, and the options are the "
    "entities it ended on. Some of them may not answer the question. Choose the entity that best answers it.")
SURE = 0.9  # chosen on d1-ultra's WebQSP test walks, then confirmed with no tuning on fresh WebQSP and CWQ questions


def path_lines(decisions: list[dict]) -> list[str]:
    """The walk's path in the words the walk showed the model (graph.describe_step), rebuilt from its decision log."""
    lines = []
    for d in decisions:
        if d["chosen"] in (None, 0):
            continue  # a failed call, or STOP (always option 0)
        relation, direction, targets = re.fullmatch(r"follow (\S+) \((outgoing|incoming)\) to: (.*)",
                                                    d["options"][d["chosen"]], re.S).groups()
        arrow = "->" if direction == "outgoing" else "<-"
        lines.append(f"{{{graph.shown_names(d['current'])}}} {relation} ({direction} {arrow}) {{{targets}}}")
    return lines


def candidates(question: dict, final: list[str] | None) -> list[str]:
    """What the pick chooses from: the walk's answer without the question's own entities."""
    return [e for e in final or [] if e not in question["start"]]


def pick_request(question: dict, decisions: list[dict], options: list[str]) -> dict:
    path = path_lines(decisions)
    state = {"question": question["question"], "current_entities": options, "path_so_far": path, "hop": len(path)}
    return {"state": state, "questions": {"next_step": {"type": "choice", "instructions": PICK_INSTRUCTIONS,
                                                        "criteria": {str(i): e for i, e in enumerate(options)}}}}


def picked_answer(final: list[str], options: list[str], call: dict | None, sure: float = SURE) -> list[str]:
    """The answer after the pick: the chosen entity alone if the model gave it at least `sure`, otherwise the walk's
    answer as it was (also when there was no pick, or its call failed)."""
    if call and not call["error"] and (call["probabilities"] or {}).get(call["option"], 0.0) >= sure:
        return [options[int(call["option"])]]
    return final


async def pick_answers(run: str, benchmark: str, dm, concurrency: int = 16, budget_usd: float | None = None,
                       root: Path = RUNS) -> Path:
    """After each of the decision model's walks in the run that ended on two or more entities, ask the same model
    which one answers the question: one call per walk, saved as picks.jsonl in the model's folder of the run. Walks
    already asked are skipped, and failed calls are tried again. budget_usd stops once that much is spent."""
    # ponytail: a walk that ended on more than 255 entities is not asked (a choice takes at most 255 options) and
    # keeps its answer; a tournament as in walk.decide would cover it, if such walks turn out to matter
    folder = root / run / benchmark / folder_name(dm.label)
    questions = {q["qid"]: q for q in json.loads((folder.parent / "questions.json").read_text())}
    walks = {r["qid"]: r for r in map(json.loads, (folder / "results.jsonl").read_text().splitlines())}
    decisions = {r["qid"]: r["decisions"] for r in map(json.loads, (folder / "decisions.jsonl").read_text().splitlines())}
    path = folder / "picks.jsonl"
    done = {r["qid"] for r in map(json.loads, path.read_text().splitlines()) if not r["call"]["error"]} if path.exists() else set()
    queue: asyncio.Queue = asyncio.Queue()
    for qid, w in walks.items():
        if qid not in done and 2 <= len(candidates(questions[qid], w["final"])) <= walk.MAX_OPTIONS:
            queue.put_nowait(qid)
    spent = 0.0
    async with models.client() as http:
        await dm.check(http)
        bar = tqdm(total=queue.qsize(), desc=f"picks of {dm.label}", unit="walk")

        async def worker():
            nonlocal spent
            while not queue.empty() and (budget_usd is None or spent < budget_usd):
                qid = queue.get_nowait()
                options = candidates(questions[qid], walks[qid]["final"])
                choice = await dm.choose(http, pick_request(questions[qid], decisions[qid], options))
                with open(path, "a") as f:
                    f.write(json.dumps({"qid": qid, "options": options, "call": asdict(choice)}, ensure_ascii=False) + "\n")
                spent += choice.cost or 0
                bar.update(1)
                bar.set_postfix(spent=f"${spent:.4f}")

        await asyncio.gather(*(worker() for _ in range(concurrency)))
        bar.close()
    if not queue.empty():
        print(f"Stopped at the budget of ${budget_usd}; {queue.qsize()} walks left. Run again to continue.")
    return path


def pick_outcomes(run: str, benchmark: str, dm_label: str, sure: float = SURE, root: Path = RUNS) -> pd.DataFrame:
    """Every walk of the decision model in the run (except those lost to an API error), scored four ways: as walked;
    after the pick; keeping one entity at random wherever the pick kept one, as an exact expectation; and by an oracle
    that keeps exactly the entities that are gold answers. Each way has a score column and an exact column."""
    folder = root / run / benchmark / folder_name(dm_label)
    questions = {q["qid"]: q for q in json.loads((folder.parent / "questions.json").read_text())}
    walks = {r["qid"]: r for r in map(json.loads, (folder / "results.jsonl").read_text().splitlines())}
    path = folder / "picks.jsonl"
    picks = {r["qid"]: r for r in map(json.loads, path.read_text().splitlines())} if path.exists() else {}  # last wins
    rows = []
    for qid, w in walks.items():
        if w["score"] is None:
            continue
        record = picks.get(qid)
        row = {"qid": qid, "asked": bool(record and not record["call"]["error"]), "cut": False}
        row |= dict.fromkeys(["walk", "picked", "at random", "oracle"], w["score"])
        row |= dict.fromkeys(["walk exact", "picked exact", "at random exact", "oracle exact"], float(w["exact"]))
        if row["asked"]:
            score = lambda answer: scoring.score(benchmark, questions[qid], answer)  # noqa: E731
            options, answer = record["options"], picked_answer(w["final"], record["options"], record["call"], sure)
            singles = [score([e]) for e in options]
            if answer != w["final"]:
                s = score(answer)
                row |= {"cut": True, "picked": s["score"], "picked exact": float(s["exact"]),
                        "at random": float(np.mean([x["score"] for x in singles])),
                        "at random exact": float(np.mean([x["exact"] for x in singles]))}
            gold = [e for e, x in zip(options, singles) if x["score"] > 1e-9]  # WebQSP's F1 adds 1e-40 to the hits
            if gold and (s := score(gold))["score"] > row["walk"]:
                row |= {"oracle": s["score"], "oracle exact": float(s["exact"])}
        rows.append(row)
    return pd.DataFrame(rows).set_index("qid")


def plot_picks(runs: dict[str, pd.DataFrame], title: str | None = "What the pick added, in points",
               path: Path | str | None = None) -> None:
    """A bar chart of what the pick added to the walk's answers, in points (hundredths of the score, or percentage
    points of exactly correct answers), for each run: a name and its pick_outcomes. The gray bars cut the same answers
    to one entity at random instead, which is what the pick has to beat. The thin lines are the pick's 95% intervals.
    `path` also saves the figure, for example as a PNG."""
    # The pick in blue, what it is compared with in gray.
    surface, ink, label_ink, muted, grid, blue = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#2a78d6"
    rows = [(f"{name}: {measure}", frame["picked" + suffix], frame["at random" + suffix], frame["walk" + suffix])
            for name, frame in runs.items() for measure, suffix in (("score", ""), ("exactly correct", " exact"))]
    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.8 * len(rows)), layout="constrained", facecolor=surface)
    for i, (_, picked, random_cut, walk) in enumerate(rows):
        t = paired_test(picked, walk)
        gain, low, high = 100 * t["difference"], 100 * t["low"], 100 * t["high"]
        chance = 100 * (random_cut - walk).mean()
        first = i == 0  # one legend entry each
        ax.barh(i - 0.18, gain, height=0.32, color=blue, label="the pick" if first else None)
        ax.barh(i + 0.18, chance, height=0.32, color=muted, label="cutting at random" if first else None)
        ax.errorbar(gain, i - 0.18, xerr=[[gain - low], [high - gain]], fmt="none", ecolor=ink, elinewidth=1,
                    capsize=3, label="95% interval of the pick" if first else None)
        ax.annotate(f"{gain:+.1f}", (high, i - 0.18), xytext=(6, 0), textcoords="offset points", va="center", color=ink)
        ax.annotate(f"{chance:+.1f}", (max(chance, 0), i + 0.18), xytext=(6, 0), textcoords="offset points",
                    va="center", color=label_ink)
    ax.axvline(0, color="#c3c2b7", linewidth=1)
    ax.set_yticks(range(len(rows)), [label for label, *_ in rows])
    ax.invert_yaxis()
    low, high = ax.get_xlim()
    ax.set_xlim(low, high + 0.12 * (high - low))  # room for the labels
    ax.set_xlabel("points added to the walk's answers", color=label_ink)
    ax.set_facecolor(surface)
    ax.grid(axis="x", color=grid, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=muted)
    ax.tick_params(axis="y", length=0, labelcolor=label_ink)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    if title:
        ax.set_title(title, loc="left", color=ink, fontsize=11)
    fig.legend(loc="outside lower center", ncols=3, frameon=False, labelcolor=label_ink)
    if path:
        fig.savefig(path, dpi=150, facecolor=surface)
    plt.show()


# ---------------------------------------------------------------- statistics
def bootstrap(frame: pd.DataFrame, statistic, n: int = 1000, seed: int = 0) -> dict:
    """A statistic of the questions, with a 95% interval from resampling the questions."""
    rng = np.random.default_rng(seed)
    values = [statistic(frame.iloc[rng.integers(0, len(frame), len(frame))]) for _ in range(n)]
    low, high = np.nanpercentile(values, [2.5, 97.5])
    return {"estimate": statistic(frame), "low": float(low), "high": float(high)}


def paired_test(a: pd.Series, b: pd.Series) -> dict:
    """The mean of a - b over the questions that both have, with a 95% interval and a two-sided p-value. A paired
    z-test; on yes/no outcomes it is close to McNemar's test."""
    # ponytail: normal approximation, fine for the hundreds of questions a test needs; too rough for a tiny pilot
    d = (a - b).dropna()
    mean, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
    p = math.erfc(abs(mean / se) / math.sqrt(2)) if se > 0 else float(mean == 0)
    return {"questions": len(d), "difference": mean, "low": mean - 1.96 * se, "high": mean + 1.96 * se, "p_value": p}


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values, for running the main test on several benchmarks: compare each with 0.05."""
    adjusted, running = {}, 0.0
    for i, (name, p) in enumerate(sorted(p_values.items(), key=lambda item: item[1])):
        running = max(running, min(1.0, (len(p_values) - i) * p))
        adjusted[name] = running
    return adjusted


def questions_needed(variance: float, gain: float, tests: int = 3, power: float = 0.8) -> int:
    """Questions a paired test needs to detect a mean gain, with 0.05 split over `tests` tests. `variance` is the
    variance of the per-question differences in a pilot; for yes/no outcomes it is about the share of questions
    whose outcome differs between the two policies."""
    z = NormalDist().inv_cdf
    return math.ceil((z(1 - 0.05 / tests / 2) + z(power)) ** 2 * variance / gain ** 2)


# ---------------------------------------------------------------- self-check: uv run python -m utils.self_aware
def _check() -> None:
    """A tiny film graph and two fake models. The decision model is unsure at the second step (0.6) and picks the
    wrong relation; the strong model picks the right one."""
    import os
    import tempfile

    from . import runs

    triples = [("Film A", "starred_actors", "Tom"), ("Film B", "starred_actors", "Tom"),
               ("Film A", "directed_by", "Dir X"), ("Film B", "directed_by", "Dir Y"), ("Film A", "written_by", "Ann")]

    async def neighbours(driver, benchmark, question, current):
        return ([{"relation": r, "direction": "outgoing", "target": t} for h, r, t in triples if h in current]
                + [{"relation": r, "direction": "incoming", "target": h} for h, r, t in triples if t in current])

    class Fake:
        kind = "fake"

        def __init__(self, label, likes, cost, probabilities=True):
            self.label, self.likes, self.cost, self.probabilities = label, likes, cost, probabilities

        async def check(self, http):
            return {"slug": self.label}

        async def choose(self, http, request):
            self.last = request
            criteria = request["questions"]["next_step"]["criteria"]
            key, p = next((k, p) for text, p in self.likes for k, v in criteria.items() if text in v)
            rest = (1 - p) / (len(criteria) - 1)
            probabilities = {k: round(p if k == key else rest, 2) for k in criteria} if self.probabilities else None
            return models.Choice(option=key, probabilities=probabilities, cost=self.cost)

    dm = Fake("fake/decision", [("written_by (outgoing)", 0.6), ("starred_actors (incoming)", 0.9), ("STOP", 0.95)], 0.001)
    strong = Fake("fake/strong", [("directed_by (outgoing)", 1), ("starred_actors (incoming)", 1), ("STOP", 1)], 0.01,
                  probabilities=False)
    question = {"qid": "q1", "question": "Who directed the films that Tom starred in?", "start": ["Tom"],
                "answers": ["Dir X", "Dir Y"]}
    graph.neighbours = neighbours

    async def main():
        alone = {m.label: await walk.walk("metaqa", question, m, None, None, 4) for m in (dm, strong)}
        seen = strong.last["state"]["path_so_far"]  # the path the strong walk showed the model at its last step
        tree = await grow("metaqa", question, dm, strong, None, None, 4)
        real = await grow("metaqa", question, dm, strong, None, None, 4, threshold=0.7)
        with tempfile.TemporaryDirectory() as tmp:
            runs.RUNS = Path(tmp)
            os.environ.setdefault("OPENROUTER_API_KEY", "test-only")  # models.client() wants one; fakes never send it
            await grow_trees(runs.open_run("metaqa", [question], "check"), dm, strong)
            loaded = load_trees("check", "metaqa", dm.label, strong.label, root=Path(tmp))
            # The strong model's walk saved as a run, then the pick on it by a model sure of one director.
            folder = Path(tmp) / "check" / "metaqa" / folder_name(strong.label)
            folder.mkdir()
            record = {k: v for k, v in alone[strong.label].items() if k != "decisions"}
            (folder / "results.jsonl").write_text(json.dumps(record | scoring.score("metaqa", question, record["final"])))
            (folder / "decisions.jsonl").write_text(json.dumps({"qid": "q1", "decisions": alone[strong.label]["decisions"]}))
            await pick_answers("check", "metaqa", Fake(strong.label, [("Dir X", 0.95)], 0.001), root=Path(tmp))
            picked = pick_outcomes("check", "metaqa", strong.label, root=Path(tmp)).loc["q1"]
            unsure = pick_outcomes("check", "metaqa", strong.label, sure=0.99, root=Path(tmp)).loc["q1"]
        return alone, seen, tree, real, loaded, picked, unsure

    alone, seen, tree, real, loaded, picked, unsure = asyncio.run(main())
    assert alone["fake/decision"]["final"] == ["Ann"] and alone["fake/strong"]["final"] == ["Dir X", "Dir Y"]

    # The tree holds both models' own walks: never escalating is the decision model's walk, always the strong one's.
    never, always = evaluate(tree, threshold_policy(0)), evaluate(tree, threshold_policy(1.01))
    assert never["score"] == 0 and math.isclose(never["dm_cost"], alone["fake/decision"]["cost"])
    assert always["score"] == 1 and always["escalated"] == always["decisions"] == 3
    # A threshold of 0.7 escalates exactly the unsure step, and pays for one strong call.
    at_07 = evaluate(tree, threshold_policy(0.7))
    assert at_07["score"] == 1 and at_07["escalated"] == 1 and at_07["decisions"] == 3
    assert math.isclose(at_07["dm_cost"] + at_07["strong_cost"], 3 * 0.001 + 0.01)
    # Random escalation is an exact expectation, and escalates exactly the share it is given.
    half = evaluate(tree, random_policy(0.5))
    assert math.isclose(half["score"], 0.5) and math.isclose(half["escalated"] / half["decisions"], 0.5)
    # The real run of the threshold scores as the tree predicts, and costs less than the tree.
    assert evaluate(real, threshold_policy(0.7)) == at_07 and tree_cost(real) < tree_cost(tree)
    assert loaded == {"q1": tree}  # saved and loaded back unchanged

    # The checks on a run, on the decision model's own walk (step confidences 0.9, 0.6 and 0.95).
    record = alone["fake/decision"] | scoring.score("metaqa", question, alone["fake/decision"]["final"])
    signals = walk_signals(pd.DataFrame([record]), {"q1": record["decisions"]})
    assert signals.loc["q1", "confidence"] == 0.6 and math.isclose(signals.loc["q1", "product"], 0.9 * 0.6 * 0.95)
    assert resolution({"q1": record["decisions"]})["0.90 to 0.99"] == 2 / 3

    # The statistics.
    assert auroc([1, 2, 3, 4], [0, 0, 1, 1]) == 1 and auroc([4, 3, 2, 1], [0, 0, 1, 1]) == 0
    assert auroc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5
    perfect = pd.DataFrame({"confidence": [0.9, 0.8, 0.3, 0.2], "score": [1.0, 1.0, 0.0, 0.0]})
    assert abstention(perfect)["gap closed"] == 1  # a perfect signal abstains as well as the oracle
    assert holm({"a": 0.125, "b": 0.25, "c": 0.5}) == {"a": 0.375, "b": 0.5, "c": 0.5}
    assert questions_needed(0.06, 0.02) == 1571

    # The pick, on the strong walk, which ended on both directors: the model is asked with the walk's own path.
    assert path_lines(alone["fake/strong"]["decisions"]) == seen and len(seen) == 2
    assert picked["asked"] and picked["cut"] and unsure["asked"] and not unsure["cut"]
    # Dir X alone is a right answer but not the whole answer, as is any one director kept at random.
    assert (picked["walk"], picked["walk exact"], picked["picked"], picked["picked exact"]) == (1, 1, 1, 0)
    assert (picked["at random"], picked["at random exact"], picked["oracle exact"], unsure["picked exact"]) == (1, 0, 1, 1)
    failed = asdict(models.Choice(None, None, error="http_500"))
    assert picked_answer(["A", "B"], ["A", "B"], failed) == ["A", "B"] == picked_answer(["A", "B"], ["A", "B"], None)
    print("self-check passed")


if __name__ == "__main__":
    _check()
