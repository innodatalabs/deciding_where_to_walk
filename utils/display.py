"""Tables and charts for the notebooks."""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import pandas as pd

from .data import BENCHMARKS


def options_table(options: list[str], call: dict | None = None, top: int | None = 12) -> pd.DataFrame:
    """The options of one step, with each option's probability (decision models) and the model's choice.

    A call from the decision log offers some of the step's options (`option_indexes`), numbered from 0 in its request.
    Its answers are mapped back to the step's own option numbers. Options that the call did not offer, such as STOP
    on unnamed records, are left out."""
    call = call or {}
    indexes = call.get("option_indexes") or list(range(len(options)))
    probabilities = {indexes[int(k)]: p for k, p in (call.get("probabilities") or {}).items()}
    chosen = indexes[int(call["option"])] if call.get("option") is not None else None
    rows = [{"option": i, "text": options[i], "probability": probabilities.get(i), "chosen": "◀" if i == chosen else ""}
            for i in indexes]
    table = pd.DataFrame(rows).set_index("option")
    if probabilities:
        table = table.sort_values("probability", ascending=False)
    if table["probability"].isna().all():
        table = table.drop(columns="probability")
    return table.head(top) if top else table


def walk_log(decisions: list[dict]) -> pd.DataFrame:
    """One row per step of a walk: where it stood, what it chose, and how sure the model was."""
    rows = []
    for d in decisions:
        call = d["calls"][-1] if d["calls"] else {}
        chosen = d["options"][d["chosen"]] if d["chosen"] is not None else f"(error: {call.get('error')})"
        if not d["calls"] and d["chosen"] is not None:
            chosen += "  (the only connection left, followed without asking the model)"
        p = (call.get("probabilities") or {}).get(call.get("option"))
        rows.append({"step": d["hop"] + 1, "standing on": ", ".join(d["current"][:5]) + (" ..." if len(d["current"]) > 5 else ""),
                     "options": len(d["options"]), "chosen": chosen, "probability": p,
                     "cost ($)": sum(c["cost"] or 0 for c in d["calls"])})
    return pd.DataFrame(rows).set_index("step")


def decision_log(metadata: dict, top: int | None = 4) -> pd.DataFrame:
    """Every decision of one DecisionWalkRetriever search (neo4j-graphrag), with each option's probability."""
    tables: dict[str, pd.DataFrame] = {}
    for d in metadata["decisions"]:
        label = "start node" if d["kind"] == "start" else f"step {d['hop'] + 1}"
        while label in tables:  # a step with more options than one call takes is decided in rounds
            label += " (next round)"
        tables[label] = options_table(d["options"], {"probabilities": d["probabilities"], "option": d["choice"]}, top)
    return pd.concat(tables, names=["decision"])


def summary(results: pd.DataFrame) -> pd.DataFrame:
    """One row per run, benchmark and model: the score with its 95% interval, exact answers, walks that ran out of
    steps, failures, and the cost per 1,000 questions."""
    rows = []
    for (run, bench, model), g in results.groupby(["run", "benchmark", "model"], sort=False):
        scored = g[g["score"].notna()]
        n = len(scored)
        mean = scored["score"].mean() if n else math.nan
        half = 1.96 * scored["score"].std(ddof=1) / math.sqrt(n) if n > 1 else math.nan
        rows.append({"run": run, "benchmark": BENCHMARKS[bench].title, "model": model,
                     "questions": len(g), "failed": int((g["stopped_by"] == "api_error").sum()),
                     "score": round(mean, 3), "± 95%": round(half, 3),
                     "metric": BENCHMARKS[bench].metric,
                     "exactly correct": f"{scored['exact'].mean():.0%}" if n else "",
                     "hit step limit": f"{(g['stopped_by'] == 'step_limit').mean():.0%}",
                     "$ per 1,000 questions": round(g["cost"].mean() * 1000, 3)})
    return pd.DataFrame(rows)


def plot_score_cost(table: pd.DataFrame) -> None:
    """Score against cost per 1,000 questions, one panel per benchmark. Up and to the left is better."""
    benches = list(dict.fromkeys(table["benchmark"]))
    fig, axes = plt.subplots(1, len(benches), figsize=(5.5 * len(benches), 4), squeeze=False)
    for ax, bench in zip(axes[0], benches):
        t = table[table["benchmark"] == bench]
        for _, row in t.iterrows():
            ax.errorbar(row["$ per 1,000 questions"], row["score"], yerr=row["± 95%"], fmt="o", capsize=3)
            ax.annotate(f"{row['model']}\n({row['run']})", (row["$ per 1,000 questions"], row["score"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("US$ per 1,000 questions (log scale)")
        ax.set_ylabel(t["metric"].iloc[0])
        ax.set_title(bench)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
