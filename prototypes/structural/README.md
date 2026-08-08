# Structural features — prototype

**Status: prototype. Nothing imports this.** It does not touch the feature
contract, the model, `safeship_ci`, or the serving path.

```bash
python -m prototypes.structural                    # HEAD~1..HEAD
python -m prototypes.structural v1.2.0 HEAD
```

## Why

SafeShip's ten features are entirely volume, process and timing. Not one says
anything about the *shape* of the code being changed.

The Prime Video deployment-risk study ablated exactly that split:

| Feature set | F1 |
|---|---|
| Change-volume only (what SafeShip has) | 0.565 |
| Code-level structural metrics | 0.809 |
| Structural, with volume removed | **0.848** |

Their conclusion: *"structural code complexity, not raw change volume, is the
primary driver."*

A benchmark on ApacheJIT put the alternative in perspective — swapping
RandomForest for gradient boosting moved AUC-PR by **+0.006**, while swapping
synthetic data for real moved it by **+0.130**. The model is not the constraint.
This is an attempt at the thing that is.

## What it extracts

Eleven signals, from the diff and the post-change file contents:

| Feature | Meaning |
|---|---|
| `added_complexity` | Cyclomatic complexity added by the change |
| `max_nesting_depth` | Deepest block nesting among changed lines |
| `decls_added` / `decls_removed` | Functions, classes and methods introduced or deleted |
| `hunk_count` | Separate change regions — diffusion *within* files |
| `max_hunk_size` | Largest contiguous change |
| `comment_ratio` | Share of changed lines that are comments |
| `test_file_ratio` | Share of changed files that are tests |
| `config_file_ratio` | Share that are config/infra |
| `languages_touched` | Distinct languages in one change |
| `formatting_ratio` | Share of churn that is pure formatting |

## Two paths, deliberately

**Python** goes through the `ast` module — exact, stdlib, no dependency.
**Everything else** goes through a lexical estimator counting control-flow
keywords and tracking brace/indent depth.

The lexical path is an approximation (a keyword inside a string literal counts),
so `method` travels with the value. Mixing exact and estimated numbers as though
equivalent is how a metric quietly stops meaning anything.

Prime Video solved multi-language by putting an LLM in the extractor. That is a
reasonable next step, but it puts a network call and a bill on the critical path
of every build — so this establishes what is reachable without one first.

## Validation

Run across 40 real commits of this repository:

- **All eleven features vary.** None degenerate.
- **Median 63ms, max 457ms** — the pipeline budget is 3000ms.
- **`added_complexity` correlates +0.00 with `diff_size`.** This is the result
  that matters: it carries genuinely new information rather than restating
  change size, which is the premise the Prime Video finding rests on.

### A bug this found — which also affects the shipped `diff_size`

Commit `0360402` normalised line endings. `git diff` reported **35,445 changed
lines**; `git diff -w` reported **15**. Every line had moved and none of them
meant anything.

The first version of this extractor scored it higher for structural complexity
than almost any real change in the repository — a mechanical reformat rating as
one of the riskiest things the project had ever shipped.

Fixed by diffing with `-w` and reporting `formatting_ratio` separately, so "big
diff, all formatting" and "big diff, all logic" can get opposite verdicts.

**SafeShip's live `diff_size` has the same blind spot.** A Prettier run or an
import reorder currently reads as an enormous, risky change. That is backwards —
a mechanical reformat is among the safest things you can deploy. Worth fixing in
`safeship_ci/gitinfo.py` regardless of whether these features ever ship.

## What this does *not* establish

These features vary, are cheap, and are not proxies for change size. **Nobody has
shown they predict anything**, because that needs commits with both diffs and
outcome labels — ApacheJIT has the labels but not the diffs, so testing that
means cloning the Apache repositories and extracting at each commit.

That is the next experiment, and it should happen before any of this goes near
the feature contract. Adding a feature to the model changes the contract, the
stored rows, and every deployed model — it deserves its own deliberate step.
