# llm-eval-harness

Regression testing for RAG applications: catch the release that quietly made retrieval worse.

> **Status — in development.** This README is the specification I am building to.
> Nothing below is claimed as working unless it is checked off in
> [Project status](#project-status). Numbers appear only when a real run produces them.

---

## The problem

A RAG system can return a perfectly good answer while its retrieval is broken — the model
fills the gap from parametric memory, and the answer looks right to a human reviewer. The
failure only surfaces later, on a question where memory does not cover the gap.

That is why "we read through some outputs and they looked fine" is not a test. Answer quality
and retrieval quality are separate signals, they move independently between releases, and only
one of them is visible without measuring it deliberately.

---

## What this catches

Filled per row as each metric layer produces a real run. Retrieval is measured;
generation and operations are not implemented yet, so those rows stay empty on purpose.

| Failure mode | Detected by | Baseline | Current |
|---|---|---|---|
| Retrieved context misses the evidence | `context_recall` | — | 0.948 |
| Evidence retrieved but ranked low | `context_precision` | — | 0.836 |
| Claims not supported by context | `faithfulness` | — | — |
| Answer drifts off the question | `answer_relevancy` | — | — |
| Silent cost or latency regression | `cost_p95`, `latency_p95` | — | — |
| Model became evasive after a prompt change | `refusal_rate` | — | — |

Retrieval numbers are from a single local run against the full 79-case golden set
(76 scored, 3 excluded as intentionally unrelated queries), `EmbeddingRetriever` on
`all-MiniLM-L6-v2`. No baseline exists yet — run versioning and comparison are not
built (see [Project status](#project-status)), so "Current" is the only number there is.

---

## Quick start

```bash
pip install -e .
eval-harness run   --config configs/reference.yaml
eval-harness report --against baseline
```

The third command prints a per-metric diff against the stored baseline, with confidence
intervals, and exits non-zero if any metric regressed beyond its threshold.

---

## Metrics, and why these

Most eval suites report one "quality score". A single number tells you something got worse and
nothing about where, which means you cannot act on it. This harness reports four layers, because
each one fails for different reasons and gets fixed by a different person.

### Retrieval

| Metric | Question it answers | Why it is here |
|---|---|---|
| `context_recall` | Did we retrieve the evidence the answer needs? | The failure described above. Ranked first because it is the one that hides behind good-looking answers. |
| `context_precision` | Is that evidence near the top of the context? | Evidence buried at rank 9 still costs tokens and still degrades generation. Catches chunking and reranker regressions that recall alone misses. |

### Generation

| Metric | Question it answers | Why it is here |
|---|---|---|
| `faithfulness` | Is every claim in the answer supported by retrieved context? | The operational definition of hallucination in RAG. Unsupported-but-true is still a defect: it means the system is guessing. |
| `answer_relevancy` | Does the answer address the question asked? | Catches the opposite failure — grounded, accurate, and beside the point. |

### Operations

`cost_p50/p95`, `latency_p50/p95`, `refusal_rate`. These regress silently and nobody
notices until the invoice or the support queue does. A prompt edit that improves faithfulness
by two points and triples cost is not an improvement, and the harness should say so in the
same report.

### The judge itself

Three of the metrics above are computed by an LLM judge, which makes the judge a measuring
instrument that can drift. So it is measured too:

- `judge_agreement` — Cohen's κ between the judge and a human-labelled calibration set.
- **No judge-based metric is reported when κ falls below 0.6.** The run fails with a message
  saying the instrument needs recalibration, rather than printing numbers nobody should trust.

### Deliberately not included

- **BLEU / ROUGE.** N-gram overlap with a reference answer measures phrasing, not correctness.
  Open-ended answers have many correct forms.
- **A single composite score.** Aggregating the layers above destroys exactly the information
  that makes a report actionable.
- **Uncalibrated LLM-as-judge.** A judge whose agreement with humans was never measured is an
  unvalidated instrument, and its output is decoration.

---

## Reading the numbers honestly

Small evaluation sets produce noisy results, and most eval reports present that noise as
signal. This harness does three things about it:

- Reports **bootstrap confidence intervals**, not bare point estimates.
- Compares versions with a **paired test** on the same cases, not by subtracting two averages.
- **Refuses to call a difference a regression when it sits inside the interval.** A metric that
  moved from 0.81 to 0.78 with a ±0.05 interval did not move.

The report states the sample size required to detect the effect size you care about, so a
suite too small to answer the question says so instead of guessing.

---

## Continuous integration

```
In progress
```

The job posts a metric diff as a pull-request comment and fails the build on a regression
outside its confidence interval. Thresholds live in the config, not in the workflow, so
changing a gate is a reviewable commit.

---

## Reference dataset

The harness ships with a golden dataset built from a game-content scenario — NPC dialogue
retrieval and user-generated-content moderation. It was chosen over the usual documentation-QA
example for two reasons: the correct answer is frequently ambiguous, which stresses judge
calibration in a way clean factual QA does not, and the cost ceiling per query is low enough
that the economics of the pipeline actually matter.

Bring your own dataset with `--config`; nothing in the metric layer is domain-specific.

---

## Limitations

What this harness does **not** do. Reading this section before adopting it will save you
a bad assumption.

- **No adversarial coverage.** Prompt injection, jailbreaks and tool-misuse are a different
  threat model and a different tool. Do not read a green run here as a security result.
- **No multi-turn evaluation.** Single question, single answer. Conversational state,
  clarification, and context carry-over are out of scope.
- **A golden dataset is a sample, not your traffic.** It measures the cases you thought of.
  Production drift needs online monitoring, which this is not.
- **Judge bias is mitigated, not eliminated.** Position and verbosity bias are controlled for;
  the judge still inherits the preferences of the model behind it, and κ tells you how much
  that matters on your data — it does not remove it.
- **Metrics are comparative, not absolute.** They answer "did this release get better or
  worse than the last one". They do not support a claim like "our bot is 97% accurate".
- **English only** for now. The judge prompts and the reference dataset are untested elsewhere.

---

## Project status

- [x] Metric layer: retrieval (`context_recall`, `context_precision`)
- [ ] Metric layer: generation (`faithfulness`, `answer_relevancy`)
- [ ] Metric layer: operations (cost, latency, refusal rate)
- [ ] Judge calibration and κ gate
- [ ] Bootstrap intervals and paired comparison
- [ ] Reference golden dataset (target: 100+ cases)
- [ ] Run versioning and baseline storage
- [ ] GitHub Actions integration and PR comment output
- [ ] First full run — populates *What this catches*

---

## Why I built it

Fifteen years of test automation, now applied to systems that don't give the same answer twice.
The methodology here — separate the failure modes, measure the instrument before trusting it,
refuse to report noise as signal — is ordinary test engineering. Applying it to probabilistic
systems is the part that is new.

Notes on the approach: [methodology](docs/methodology.md).

## License

Apache-2.0
