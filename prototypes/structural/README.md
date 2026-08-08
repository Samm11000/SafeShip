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

## Verdict: do not promote these. They make the model worse.

The experiment the section above called for has now been run, and it came back
negative.

**Method.** Cloned `apache/zookeeper` and `apache/zeppelin`, extracted structural
features at all 2,288 commits that ApacheJIT labels, joined them to the existing
ten features, and compared with a time-ordered split (train ≤2017, n=1,786; test
>2017, n=502), averaged over 15 seeds and paired so only the feature set varies.

| Feature set | AUC | AUC-PR |
|---|---|---|
| SafeShip's 10 | **0.772** ±0.002 | **0.574** ±0.005 |
| 10 + 11 structural | 0.754 ±0.002 | 0.525 ±0.005 |

Δ AUC **−0.017**, Δ AUC-PR **−0.049**, both far outside the paired standard
error. Adding these features reliably *hurts*.

**It is not simply that more columns overfit.** The control settles that:

| | AUC-PR |
|---|---|
| Baseline (10) | 0.574 |
| + 11 columns of random noise | 0.544 |
| + 11 structural features | **0.525** |

The structural features do worse than random noise. They are not diluting the
signal, they are actively misleading the model. Even adding only the single
strongest one drops AUC-PR to 0.548.

## Why this does not refute the Prime Video result

**Every file in both repositories went through the lexical fallback — 533 files,
100% lexical, 0% AST.** ApacheJIT is entirely JVM projects, and the exact path
here is Python-only. So what was tested is keyword-counting, not parsing.

Prime Video's 0.809 came from real structural analysis — cyclomatic complexity,
nesting depth and declaration counts from an actual parse, using an LLM to cover
multiple languages. Counting `if`, `for` and `&&` in added lines is a much
cruder proxy, and on Java it evidently tracks something other than risk.

So the finding is narrow and specific: **cheap lexical approximation of structural
complexity is worse than nothing.** The underlying hypothesis is untested.

## If someone picks this up again

1. **Use a real parser.** `tree-sitter` has maintained grammars for Java, Go,
   JavaScript, Rust and the rest, and would give the AST path everywhere instead
   of only Python. That is the experiment that would actually test the
   hypothesis.
2. **Or take the LLM route** Prime Video used, accepting a network call and a
   bill on the critical path of every build.
3. **Re-run this exact comparison** before promoting anything. The script is
   what produced the tables above; a negative result is cheap to reproduce and a
   positive one needs to survive the random-noise control.

## What it did find that is worth keeping

Two things, independent of whether structural features ever ship:

- **`diff_size` counts formatting as risk.** Commit `0360402` normalised line
  endings: `git diff` reported 35,445 changed lines, `git diff -w` reported 15.
  A Prettier run or an import reorder currently reads to SafeShip as one of the
  largest, riskiest changes a team has ever shipped — which is backwards. Fixing
  that in `safeship_ci/gitinfo.py` is cheap and does not depend on any of this.
- **`_git` raised on non-UTF-8 bytes.** `text=True` decodes strictly, and real
  repositories are full of binary blobs and latin-1 source. It died at commit 20
  of 839 on the first real run over `apache/zookeeper`.

  `safeship_ci/gitinfo.py` uses the same `text=True` pattern, and my first
  reading was that it shared the bug. Checking properly, it does not fire there:
  gitinfo only ever asks git for `--numstat`, `--name-only`, `rev-parse` and
  `rev-list`, all of which return ASCII counts and paths — and git quotes
  non-ASCII paths by default. Reproduced against a repository containing latin-1
  file content and `diff_size` came back correctly.

  This prototype hits it because it uses `git show <sha>:<path>` to read whole
  files, which gitinfo never does. So the pattern is worth hardening defensively
  if anyone adds a content-reading call there, but it is not a live bug today.
