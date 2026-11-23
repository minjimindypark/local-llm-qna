# Reliable Q&A on a local LLM

A question-answering service for the readers of a book, built so it can be **trusted enough to sell** while running on **one ordinary 16 GB machine with no paid API**. Local models are cheap and private but are rarely trusted for real products — left alone, a small model invents answers. This repo is the pipeline and the experiments that made one reliable enough to ship.

> This is the **public** version: the engineering, the pipeline code, and the measured results. The book text, the author's answer bank, and the trained adapter weights are withheld (copyright); the code runs against your own data in the same format.

## What it does

Every reader question takes one of three paths — and the system never fills a gap with a guess:

```
                                   ┌─► Answer        (from the source only)
Question ─► Find passages ─► Local 12B model ─┼─► Ask the author  (when the source can't)
            book + author answers              └─► Refuse        (misuse or off-topic)
```

- **Answers only from the source.** Retrieval over the book plus the author's vetted answers; the model is instructed to answer from the retrieved passages, not from its own memory.
- **Hands off instead of guessing.** When the source has no answer, the question is routed to the author rather than hallucinated.
- **Refuses misuse.** Abuse, data-exfiltration, and reckless legal/tax specifics are declined.
- **$0 per answer.** No cloud, no API bill; every reader's question stays on the machine.

## Pipeline

| Stage | What runs |
|---|---|
| Retrieve | `bge-m3` bi-encoder → top-30 by cosine similarity |
| Re-rank | `bge-reranker-v2-m3` cross-encoder → best-first top-5 |
| Author answers | the same re-ranker scores the author's vetted Q&A; the strongest matches are appended as evidence |
| Generate | Gemma-3 12B (SFT adapter, Q6_K GGUF, ~9 GB) under a strict source-only + routing system prompt |
| Route | the model self-judges: **answer** / **ask the author** / **refuse** |

Code: [`pipeline/server_local.py`](pipeline/server_local.py) is the full serving pipeline. [`train/`](train) holds the fine-tuning (`ft.py`), the hybrid retriever/re-ranker (`simple_hybrid.py`), and the evaluation harnesses (`score_all.py`, `test_local.py`).

## What I tested, and what moved the needle

Getting a small local model to this level meant trying a lot and keeping only what measurably helped — each change had to beat the run-to-run noise on two separate test sets before it counted.

| What I tried | Effect |
|---|---|
| Bigger model (8B → 12B) | **The biggest single gain in quality** |
| Quantized it to fit 16 GB | Kept the quality at roughly half the memory |
| Re-ranked the retrieved passages | Clearly better on the hard questions |
| Put the strongest passage first | Another solid gain on hard questions |
| Author-written answers for common questions | **What carried it to commercial quality** |
| Safety & refusal layer | Blocks misuse, legal risk, reckless advice |
| Distillation, preference learning (ORPO) | No measurable gain — dropped |
| Full training-hyperparameter sweep | No measurable gain — dropped |
| Answer post-processing | No measurable gain — dropped |

Most ideas did not beat the noise, and I cut them instead of shipping them.

## The hard part is everything around the model

Any model can generate text. Making one you can put in front of paying customers is a different job. This service:

- answers only from the source and hands a question to the author when the source has no answer, instead of guessing;
- never names a real company — caught the model doing that from the source notes and shut it down, so the product carries no legal risk;
- stays inside what the source actually says about things like tax, pointing people to a professional instead of inventing specifics;
- refuses misuse and keeps every reader's question on the machine.

## Built to hand off

It ships as a clean pipeline with a feedback log (`more` / `enough` signals per answer), so the team taking it live can keep improving it on real users instead of starting over.

## Running it on your own data

The code expects, in the pipeline's working directory:

- `chunks.json` — your source passages: `[{ "text", "src", "lines", "ctx" }, …]`
- `answers.md` — curated Q&A as `Q. …` / `A. …` pairs
- a GGUF chat model served by `llama-server` on `localhost:8099`

Then `python pipeline/server_local.py` serves the Q&A endpoint. Retrieval and re-ranking use `bge-m3` and `bge-reranker-v2-m3` via `sentence-transformers`.

---

*Minji Park — designed, built, and measured on a single local machine.*
