"""Scoring a walk's answer with each benchmark's official evaluation.

A walk ends on a set of entities that were all reached by the same choice, so they are tied. Where a benchmark's
metric looks at the top answer (Hits@1, P@1), the score is the chance that a random pick from the set is correct: the
share of the set that is a gold answer.

  MetaQA  Hits@1  share of the answer set that is a gold answer (also exact match and F1)
  WebQSP  F1      the official eval.py: precision, recall and F1 against the gold answers of each annotated parse of
                  the question, keeping the best parse
  CWQ     P@1     the official eval_script.py: an answer counts if it matches a gold answer or one of its aliases;
                  the score is the share of the answer set that matches (also "reached": whether any of it matches)

"Exactly correct" means the answer is fully right: the exact gold set, an F1 of 1, or a P@1 of 1.
"""
from __future__ import annotations

import re
import unicodedata


def score(benchmark: str, question: dict, final: list[str] | None) -> dict:
    if final is None:  # the walk failed with an API error, so it has no answer
        return {"score": None, "exact": None}
    if benchmark == "metaqa":
        return _metaqa(final, question)
    if benchmark == "webqsp":
        return _webqsp(final, question)
    return _cwq(final, question)


# ---------------------------------------------------------------- MetaQA
def _metaqa(final: list[str], question: dict) -> dict:
    answer, gold = set(final), set(question["answers"])
    hits = len(answer & gold)
    precision = hits / len(answer) if answer else 0.0
    recall = hits / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if hits else 0.0
    return {"score": precision, "exact": answer == gold, "f1": f1}


# ---------------------------------------------------------------- WebQSP (official eval.py)
def _prf1(gold: list[str], predicted: list[str]) -> tuple[float, float, float]:
    """Precision, recall and F1, as CalculatePRF1 in the official eval.py."""
    if not gold:
        return (1.0, 1.0, 1.0) if not predicted else (0.0, 1.0, 0.0)
    if not predicted:
        return 1.0, 0.0, 0.0
    tp = 1e-40 + sum(g in predicted for g in gold)
    fn = sum(g not in predicted for g in gold)
    fp = sum(p not in gold for p in predicted)
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    return precision, recall, 2 * precision * recall / (precision + recall)


def _webqsp(final: list[str], question: dict) -> dict:
    """The gold answers are Freebase IDs, and the walk returns names, so each name is mapped to its gold ID."""
    parses = question["gold"]["parses"]
    name_to_id = {a["name"]: a["id"] for parse in parses for a in parse if a["name"]}
    predicted = list(dict.fromkeys(name_to_id.get(n, n) for n in final if n not in question["start"]))
    precision, recall, f1 = max((_prf1([a["id"] for a in parse], predicted) for parse in parses), key=lambda x: x[2])
    return {"score": f1, "exact": f1 == 1.0, "precision": precision, "recall": recall}


# ---------------------------------------------------------------- CWQ (official eval_script.py)
def _normalize(answer: str) -> str:
    text = unicodedata.normalize("NFKD", answer).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"\W", " ", text).lower().strip()
    for article in ("the ", "a ", "an "):
        if text.startswith(article):
            text = text[len(article):]
    return text


def _matches(prediction: str, gold: list[str], question: str) -> bool:
    asks_for_year = "year" in question.lower()
    for answer in (g.lower().strip() for g in gold):
        if prediction in (answer, _normalize(answer)):
            return True
        if asks_for_year and (year := re.search("([1-2][0-9]{3})", answer)) and prediction == year.group(0):
            return True
    return False


def _cwq(final: list[str], question: dict) -> dict:
    gold = [text for a in question["gold"]["answers"] for text in (a["answer"], *a["aliases"])]
    answer = [name.lower().strip() for name in final if name not in question["start"]]
    hits = [_matches(a, gold, question["question"]) for a in answer]
    p_at_1 = sum(hits) / len(hits) if hits else 0.0
    return {"score": p_at_1, "exact": p_at_1 == 1.0, "reached": any(hits)}
