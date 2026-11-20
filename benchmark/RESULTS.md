# Benchmark results

Can a small decision model walk a knowledge graph as well as an LLM? On the three benchmarks below, the best decision
models answered as accurately as GPT-5.6 Luna, or more accurately, and GPT-5.6 Luna cost 6 to 12 times as much.

We gave six decision models and one LLM the same questions. Each model answered by walking a knowledge graph, one
step at a time, and every model saw exactly the same options at every step. This page shows what we found. The
notebooks in this folder run the same benchmark, so you can check our numbers or test your own models.

## The short version

- **Mercury Decide** (Inception) and **d1-ultra** (Liquid AI) were as accurate as GPT-5.6 Luna, or more accurate, on
  all three benchmarks. GPT-5.6 Luna cost about 10 to 12 times as much as Mercury Decide, and about 6 times as much as
  d1-ultra.
- **Jev 1.13** (Typesafe) was about as accurate as GPT-5.6 Luna on all three benchmarks, and GPT-5.6 Luna cost 3 to 5
  times as much.
- **Jev 1.14** (Typesafe) was more accurate than GPT-5.6 Luna on MetaQA and about as accurate on the other two.
  **Decision-1** (Microsoft) was about as accurate on MetaQA and ComplexWebQuestions, and less accurate on WebQSP.
- **GPT-6 Luna Decisions** (OpenAI) was less accurate than GPT-5.6 Luna on all three benchmarks. It was also the most
  expensive of the six decision models.
- **Every walk is also a Cypher query** that gives exactly the walk's answer, so an answer can be checked and run again
  in milliseconds, without a model. On MetaQA, reusing the query of an earlier walk answered 82% of 3,000 questions
  without a model call. On the three benchmarks, having GPT-5.6 Luna write the query itself (Text2Cypher) was much
  less accurate than walking. On the business graphs of Mind the Query it was the other way round: Text2Cypher was
  right on 80% of 900 questions, and starting from d1-ultra's walk lowered that to 72%. See
  [Every walk as a Cypher query](#every-walk-as-a-cypher-query).

## What we measured

| Benchmark | The questions | The graph |
|-|-|-|
| MetaQA | Questions about films that need 1, 2 or 3 steps, e.g. "Who directed the movies that Tom Hanks acted in?" We used 1,500 questions, 500 for each number of steps. | One film graph with 43,000 entities |
| WebQSP | Real questions that people typed into a search engine. We used all 1,639 test questions. | A small part of Freebase for each question |
| ComplexWebQuestions (CWQ) | Harder questions built from WebQSP, which need more steps or more conditions. We used its 3,519 development questions. | A small part of Freebase for each question |

A walk ends on a set of entities, and each benchmark scores that set in its own way. All scores run from 0 to 1, and
higher is better.

- **Hits@1** (MetaQA): the chance that a random pick from the walk's answer is right.
- **F1** (WebQSP): how well the walk's answer matches the right answers. It is high only when the answer has most of the
  right answers and few wrong ones.
- **P@1** (CWQ): like Hits@1, and an answer also counts when it matches another known name of a right answer.
- **Exactly correct**: the share of questions whose answer was fully right.

Notebook 2 in [`notebooks/`](../notebooks/) shows a worked example of each score.

## How accurate each model was

![The score of each model on each benchmark, with GPT-5.6 Luna as a dashed line](results/scores.png)

Each dot is a model's score, and its line shows the range that the score would likely fall in if we ran many more
questions (a 95% interval). The dashed line is GPT-5.6 Luna.

<!-- results-tables:start -->
### MetaQA

1,500 questions. The score is Hits@1.

| Model | Maker | Hits@1 | Exactly correct | $ per 1,000 questions | Accuracy compared with GPT-5.6 Luna | GPT-5.6 Luna costs |
|-|-|-|-|-|-|-|
| d1-ultra | Liquid AI | 0.97 | 95% | $0.045 | better | 6.0 times as much |
| Mercury Decide | Inception | 0.92 | 90% | $0.023 | better | 11.8 times as much |
| GPT-6 Luna Decisions | OpenAI | 0.84 | 81% | $0.177 | worse | 1.5 times as much |
| Jev 1.13 | Typesafe | 0.90 | 88% | $0.086 | about the same | 3.1 times as much |
| Jev 1.14 | Typesafe | 0.91 | 89% | $0.081 | better | 3.3 times as much |
| Decision-1 | Microsoft | 0.91 | 87% | $0.038 | about the same | 7.1 times as much |
| GPT-5.6 Luna (LLM) | OpenAI | 0.89 | 88% | $0.270 | (the LLM baseline) |  |

### WebQSP

1,639 questions. The score is F1.

| Model | Maker | F1 | Exactly correct | $ per 1,000 questions | Accuracy compared with GPT-5.6 Luna | GPT-5.6 Luna costs |
|-|-|-|-|-|-|-|
| d1-ultra | Liquid AI | 0.70 | 60% | $0.229 | about the same | 6.0 times as much |
| Mercury Decide | Inception | 0.70 | 61% | $0.135 | about the same | 10.1 times as much |
| GPT-6 Luna Decisions | OpenAI | 0.65 | 56% | $0.590 | worse | 2.3 times as much |
| Jev 1.13 | Typesafe | 0.68 | 58% | $0.318 | about the same | 4.3 times as much |
| Jev 1.14 | Typesafe | 0.68 | 58% | $0.319 | about the same | 4.3 times as much |
| Decision-1 | Microsoft | 0.67 | 57% | $0.237 | worse | 5.7 times as much |
| GPT-5.6 Luna (LLM) | OpenAI | 0.70 | 60% | $1.364 | (the LLM baseline) |  |

### ComplexWebQuestions

3,519 questions. The score is P@1.

| Model | Maker | P@1 | Exactly correct | $ per 1,000 questions | Accuracy compared with GPT-5.6 Luna | GPT-5.6 Luna costs |
|-|-|-|-|-|-|-|
| d1-ultra | Liquid AI | 0.38 | 31% | $0.302 | about the same | 6.1 times as much |
| Mercury Decide | Inception | 0.39 | 32% | $0.175 | better | 10.5 times as much |
| GPT-6 Luna Decisions | OpenAI | 0.35 | 29% | $0.766 | worse | 2.4 times as much |
| Jev 1.13 | Typesafe | 0.37 | 28% | $0.397 | about the same | 4.7 times as much |
| Jev 1.14 | Typesafe | 0.36 | 28% | $0.403 | about the same | 4.6 times as much |
| Decision-1 | Microsoft | 0.36 | 28% | $0.307 | about the same | 6.0 times as much |
| GPT-5.6 Luna (LLM) | OpenAI | 0.37 | 30% | $1.845 | (the LLM baseline) |  |

<!-- results-tables:end -->

On MetaQA, the decision models also answered all 39,093 test questions, and their scores there are within about 0.01
of the scores above.

## Accuracy against cost

![Score against the cost per 1,000 questions, for each model and benchmark](results/score_against_cost.png)

Up and to the left is better: more accurate and cheaper. The cost axis is logarithmic, so each step to the right is
several times more expensive. The cost of a question is the cost of every step of its walk, as billed by OpenRouter.

## Compared with the LLM, question by question

![Each decision model's score minus GPT-5.6 Luna's, with a verdict](results/compared_with_llm.png)

Here each decision model is compared with GPT-5.6 Luna on exactly the same questions. A dot to the right of the dashed
line means that the decision model scored higher. The lines are a little wider than usual 95% intervals, because we
compare six models at once and want to avoid a lucky result. We call a model **better** or **worse** only when its whole
line is on one side of zero. Otherwise we call it **about the same**.

## Every walk as a Cypher query

A walk starts from the question's entities, and each step follows one relationship type, in one direction, to all the
entities behind it. So every walk can be written as one Cypher query, with one `MATCH` for each step. For example, a
walk that answers "Who directed the movies that Tom Hanks acted in?" in two steps is this query:

```cypher
MATCH (n0:KGEntity) WHERE n0.key IN $start_keys
MATCH (n0)<-[:KG_EDGE {relation: 'starred_actors'}]-(n1:KGEntity) WHERE NOT n1.name IN $leave_out AND n1 <> n0
WITH DISTINCT n1
MATCH (n1)-[:KG_EDGE {relation: 'directed_by'}]->(n2:KGEntity) WHERE NOT n2.name IN $leave_out
WITH DISTINCT n2
RETURN n2.name AS name ORDER BY name
```

With the query, anyone can see how an answer was found, check it, and run it again without a model. We tried three
things with it. The code is in [`utils/cypher.py`](../utils/cypher.py). Every query ran in a read transaction, in which
Neo4j refuses to write.

### The query gives the walk's answer

We turned 6,478 saved walks of d1-ultra into queries, ran them, and compared what they returned with the walk's answer.

| | MetaQA | WebQSP | CWQ |
|-|-|-|-|
| Walks | 400 | 4,678 | 1,400 |
| The query returned exactly the walk's answer | 400 | 4,652 | 1,399 |
| The question's entity is not in the graph | 0 | 26 | 1 |
| Other differences | 0 | 0 | 0 |
| Median time: the query, and the walk | 1.6 ms, and 1.8 s | 2.0 ms, and 1.9 s | 4.6 ms, and 2.7 s |

The only differences are walks from an entity that is not in the graph. Such a walk has nothing to follow, so it stops
at once and answers with the entity's name from the question, while the query finds nothing. Walks of neo4j-graphrag's
`DecisionWalkRetriever`, which walks graphs with named relationship types, turn into queries in the same way: on 12
questions on Neo4j's public movie recommendations database, each query returned the walk's entities.

### Walk once, then reuse the query

Many questions differ only in their entity, such as "What movies did [E] direct?". So we tried a cache. It keeps the
question with its entity replaced by [E], and the path of a walk that ended with the model's STOP. A later question with
the same words gets the query of that path, and no model is called. If the query finds nothing, the question is walked
as usual. There were three rules for which path goes in the cache:

- **first**: the path of the first walk.
- **sure**: the path of the first walk that d1-ultra was at least 50% sure of. How sure it was is the product of the
  probabilities it gave the options it chose.
- **agree**: a path that two walks took.

We replayed d1-ultra's saved walks in order: 3,000 random MetaQA test questions, with 488 different patterns, and the
1,639 WebQSP test questions, after filling the cache with 1,000 WebQSP training questions.

| Benchmark | Rule | Answered from the cache | Score: walking every question, then with the cache | Change (95% interval) | Cost, compared with walking every question | Of those answered from the cache: better than the walk, and worse |
|-|-|-|-|-|-|-|
| MetaQA | first | 83% | 0.977, then 0.983 | +0.005 (+0.001 to +0.010) | 16% | 1.8%, and 0.9% |
| MetaQA | sure | 82% | 0.977, then 0.987 | +0.010 (+0.006 to +0.014) | 18% | 1.7%, and 0.2% |
| MetaQA | agree | 67% | 0.977, then 0.986 | +0.009 (+0.005 to +0.013) | 31% | 2.0%, and 0.4% |
| WebQSP | first | 19% | 0.701, then 0.696 | −0.006 (−0.011 to +0.000) | 82% | 3.9%, and 7.1% |
| WebQSP | sure | 10% | 0.701, then 0.700 | −0.001 (−0.003 to +0.001) | 91% | 0.0%, and 1.8% |
| WebQSP | agree | 11% | 0.701, then 0.702 | +0.001 (−0.004 to +0.005) | 91% | 5.2%, and 4.0% |

A question answered from the cache took about 2 ms instead of about 2 seconds. On MetaQA the cache filled quickly: with
the first rule, it answered 39% of the first 500 questions and 99% of the last 500.

On MetaQA the cache was even a little more accurate than walking every question, because a path that worked once fits
the next question of the same pattern better than a new walk, which can go wrong. That works because MetaQA's questions
were made from templates, and each template always needs the same path. WebQSP's questions were typed by people. Few of
their patterns repeat, and the same words can need a different path for a different entity, so the cache answered few
questions and did not help. How much a cache saves depends on how often questions repeat with only the entity changed.

### Walking, writing the query, or both

The usual way to answer a question with a Cypher query is Text2Cypher: an LLM reads the graph's schema and writes the
query. We compared it with walking on the 200 sample questions of each benchmark, with GPT-5.6 Luna (without reasoning)
as the LLM:

- **Walking**: d1-ultra walks, as on the rest of this page, and GPT-5.6 Luna walks for comparison.
- **Text2Cypher**: GPT-5.6 Luna gets the graph's relations (on MetaQA with one example fact each), the line that finds
  the question's entities, and the question, and writes one query.
- **Walk + edit**: GPT-5.6 Luna gets d1-ultra's walk as a query, with its answer, and may add what walking cannot do,
  such as a condition on a date or a number, or picking the largest. It must keep the walk's lines. If it sees nothing
  to add, drops a line of the walk, or its query fails or finds nothing, the walk's answer stays.

| Benchmark | Method | Score | Exactly correct | $ per 1,000 questions | Change from d1-ultra's walk (95% interval) |
|-|-|-|-|-|-|
| MetaQA | d1-ultra walks | 0.97 | 95% | $0.041 | |
| MetaQA | GPT-5.6 Luna walks | 0.88 | 86% | $0.242 | −0.089 (−0.129 to −0.049) |
| MetaQA | Text2Cypher: GPT-5.6 Luna writes the query | 0.75 | 72% | $0.163 | −0.218 (−0.280 to −0.157) |
| MetaQA | Walk + edit: d1-ultra walks, GPT-5.6 Luna may edit the query | 0.97 | 95% | $0.189 | no change |
| WebQSP | d1-ultra walks | 0.69 | 59% | $0.229 | |
| WebQSP | GPT-5.6 Luna walks | 0.70 | 60% | $1.401 | +0.009 (−0.035 to +0.054) |
| WebQSP | Text2Cypher: GPT-5.6 Luna writes the query | 0.41 | 38% | $0.762 | −0.275 (−0.342 to −0.209) |
| WebQSP | Walk + edit: d1-ultra walks, GPT-5.6 Luna may edit the query | 0.68 | 58% | $1.032 | −0.006 (−0.022 to +0.010) |
| CWQ | d1-ultra walks | 0.39 | 31% | $0.312 | |
| CWQ | GPT-5.6 Luna walks | 0.38 | 30% | $1.853 | −0.011 (−0.051 to +0.029) |
| CWQ | Text2Cypher: GPT-5.6 Luna writes the query | 0.11 | 10% | $0.775 | −0.282 (−0.343 to −0.222) |
| CWQ | Walk + edit: d1-ultra walks, GPT-5.6 Luna may edit the query | 0.40 | 33% | $1.172 | +0.009 (−0.008 to +0.026) |

- **Text2Cypher was much less accurate than walking**, on all three benchmarks. Most of its wrong answers came from
  queries that found nothing: 47, 48 and 115 of the 200 on MetaQA, WebQSP and CWQ. Another 10 queries on WebQSP and 37
  on CWQ were not valid Cypher. In all, 5, 15 and 31 queries used a relation that the graph does not have. When its
  query did return entities, it was as accurate as the walk on MetaQA (0.980 against 0.965 on those questions), and
  less accurate on WebQSP (0.581 against 0.796) and CWQ (0.479 against 0.604).
- **Editing the walk left accuracy about the same.** GPT-5.6 Luna saw something to add on 6 of the 200 MetaQA walks,
  43 on WebQSP and 107 on CWQ. Most of what it added found nothing, so the walk's answer stayed (77 of the 107 on CWQ).
  Of the edits that did return entities, 5 of 17 scored higher than the walk on CWQ and 4 lower, 3 of 17 higher and 4
  lower on WebQSP, and none of the 6 on MetaQA changed the score. The edit also made each question 4 to 5 times as
  expensive.

Some things to keep in mind:

- **200 questions per benchmark**, so the intervals in the last table are wide.
- **One LLM and one prompt for each method, without reasoning.** Text2Cypher with reasoning, example queries, or a
  second try after an error may do better.
- **These benchmarks mostly need relations to be followed.** Questions that count, filter or sort need a query, so we
  also compared the two on Mind the Query, below.

### On business graphs: Mind the Query

[Mind the Query](https://github.com/endeavorXx/Mind-the-Query) (EMNLP 2025 Industry Track) pairs questions with gold
Cypher queries on real Neo4j graphs with typed relationships, such as police investigations, drug side-effect
reports, bank accounts, the US Congress and the 2019 Women's World Cup. Most of its questions ask for property values,
counts or averages, often with a condition, which a walk alone cannot give. So here the walk only helps the LLM:

- **Text2Cypher**: GPT-5.6 Luna gets the graph's schema and the question, and writes the query.
- **Walk + edit**: d1-ultra first walks the graph with neo4j-graphrag's `DecisionWalkRetriever`, which finds its start
  node with a full-text search for the question over every text property of the graph. GPT-5.6 Luna then gets the
  walk as a query, and writes the final query. It keeps the walk's lines and adds what the question asks for, or, if
  the walk does not fit, writes its own query.

We took 20 random test questions of each of the five kinds on each graph that comes as a Neo4j dump, and served each
graph from its own Neo4j. A query counts as right when it returns the same rows as the gold query, compared as sets
of values, so the names and the order of the columns do not matter. The code is in
[`utils/mind_the_query.py`](../utils/mind_the_query.py).

| Kind of question | Questions | Text2Cypher right | Walk + edit right | Change, in points (95% interval) | Luna kept the walk |
|-|-|-|-|-|-|
| Simple retrieval | 180 | 95% | 83% | −11.7 (−16.9 to −6.5) | 39% |
| Complex retrieval | 180 | 76% | 70% | −6.1 (−11.9 to −0.3) | 15% |
| Simple aggregation | 180 | 92% | 80% | −12.2 (−17.5 to −7.0) | 32% |
| Complex aggregation | 180 | 39% | 35% | −4.4 (−9.3 to +0.4) | 24% |
| Evaluation (lookups) | 180 | 98% | 89% | −8.9 (−13.3 to −4.4) | 46% |
| All | 900 | 80% | 72% | −8.7 (−11.0 to −6.4) | 31% |

- **Walk + edit was less accurate on every kind of question**, and cost about 75% more: $0.76 against $0.44 per
  1,000 questions. Allowing extra columns in the answer changes these numbers by less than 2 points.
- **The walk often started at the wrong node.** A full-text search for the whole question often finds a node that
  only shares words with it, such as a tweet that mentions "United States" for a question about trolls located in the
  United States. On the questions that quote a value, the walk started at that value only 46% of the time (a rough
  check, by the start node's name). Luna kept the walk on 31% of the questions, and there it was right 69% of the
  time, while Text2Cypher was right 89% of the time on the same questions. Where a kept walk had not started at a
  quoted value, walk + edit was right 38% of the time, against 92%.
- **Even from the right start, the walk did not help.** Where the walk started at a quoted value and Luna kept it,
  walk + edit was right 91% of the time, and Text2Cypher 96%. These graphs have a few labels and relationship types
  with clear names, and most questions quote the exact value to look for, so Luna wrote most queries well on its own.
  On WebQSP and CWQ, every fact is the same generic relationship, with one of hundreds of Freebase relations as a
  property, and there Text2Cypher failed.
- **On one more graph, OpenStreetMap, the walk could not start.** Its full-text index covers about 300 properties, so
  a search for a whole question goes over Lucene's limit of 1,024 clauses. Walk + edit fell back to Text2Cypher there,
  so OpenStreetMap is left out of the table. The Star Wars graph comes without a dump, and is left out too.

To check the queries of your own runs, call `cypher.verify(["<run name>"])`. `python -m utils.cypher` does it for
`run-001`.

## How we ran it

- **The walk.** At each step, the options are the relationships around the entities that the walk stands on: one
  option for each relationship type and direction, plus STOP, which means "these entities are the answer". The model
  picks one option, and the walk follows it to all of its entities. Notebook 1 shows this step by step.
- **The same for every model.** Every model got the same question, the same instructions and the same options, and
  had the same step limit: 4 steps on MetaQA and CWQ, and 3 on WebQSP.
- **Decision models** were called through OpenRouter's decision API. A decision model returns a probability for every
  option, and the walk follows the option that it chose.
- **The LLM**, GPT-5.6 Luna, was called through OpenRouter and answered with a structured answer: a Pydantic model whose
  only field must be one of the option numbers. It did not use reasoning.
- **Unnamed records.** Freebase has records without a name, shown by their ID (such as `m.0j4jq57`), which link other
  entities together. STOP is not offered while the walk stands only on such records, so the walk moves on to the named
  entities behind them. This only happens on WebQSP and CWQ.
- **Scoring** uses each benchmark's own scoring rules and right answers.
- **When.** The runs were made in October and November 2026, with these model versions:

| Model | Maker | Name on OpenRouter | Version |
|-|-|-|-|
| d1-ultra | Liquid AI | `liquid/d1-ultra` | `liquid/d1-ultra-20261015` |
| Mercury Decide | Inception | `inception/mercury-decide` | `inception/mercury-decide-20260930` |
| GPT-6 Luna Decisions | OpenAI | `openai/gpt-6-luna-decisions` | `openai/gpt-6-luna-decisions-20261006` |
| Jev 1.13 | Typesafe | `typesafe/jev-1.13` | `typesafe/jev-1.13-20260917` |
| Jev 1.14 | Typesafe | `typesafe/jev-1.14` | `typesafe/jev-1.14-20261014` |
| Decision-1 | Microsoft | `microsoft/microsoft-decision-1` | `microsoft/microsoft-decision-1-20261009` |
| GPT-5.6 Luna (LLM) | OpenAI | `openai/gpt-5.6-luna` | `openai/gpt-5.6-luna` |

The numbers behind this page are in [`results/summary.csv`](results/summary.csv) and
[`results/paired.csv`](results/paired.csv), and those for the Cypher queries are in
[`results/walk_as_cypher_verified.csv`](results/walk_as_cypher_verified.csv),
[`results/walk_as_cypher_reuse.csv`](results/walk_as_cypher_reuse.csv),
[`results/walk_as_cypher_vs_text2cypher.csv`](results/walk_as_cypher_vs_text2cypher.csv) and
[`results/walk_as_cypher_mind_the_query.csv`](results/walk_as_cypher_mind_the_query.csv).

## Run it yourself

The three notebooks in this folder run the same benchmark with any model on OpenRouter:

1. [`1_load_benchmark.ipynb`](1_load_benchmark.ipynb) loads the questions and their graphs into Neo4j.
2. [`2_run_benchmark.ipynb`](2_run_benchmark.ipynb) runs decision models and LLMs on them.
3. [`3_compare_results.ipynb`](3_compare_results.ipynb) compares your runs.

The repository comes with 200 questions of each benchmark, so a run takes minutes and costs cents. With 200 questions
the scores are less precise than the ones on this page (their range is about ±0.02 on MetaQA and ±0.06 on WebQSP and
CWQ), and models give slightly different answers from run to run. So expect your numbers to be close to ours, but not
the same.
