# Evaluating rag-kb

The evaluation harness answers two questions with numbers instead of impressions:

1. **Does search find the right document?** For questions the corpus answers, is the document that holds the answer among the top results?
2. **Does the system stay honest?** For questions the corpus does not answer, does it decline instead of producing something that reads like an answer?

It does this with a **golden set**: a short YAML file of questions, each with the document that should answer it, plus a few questions that must be declined. The harness indexes the corpus into a throwaway data folder, asks every question through the same search and answer path the app uses, scores the results and prints a report. Your real data folder is never touched.

## Running it

```bash
ragkb eval                                   # offline, over samples/golden.yaml
ragkb eval --golden path/to/my-golden.yaml   # your own golden set
ragkb eval --live                            # real embedding model and your answer provider
ragkb eval --json                            # the raw report as JSON
```

| Mode | Embeddings | Answers | Cost | What it tells you |
|---|---|---|---|---|
| Offline (default) | `FakeEmbedder`: deterministic vectors made from a hash of the text, no model | Extractive provider, no model | Nothing; runs in seconds, same result on every machine | Whether ingestion, keyword search, fusion and the extractive decline rule work. Good for CI. The vector half of search is noise here, so recall comes from keyword matching. |
| Live (`--live`) | The real model from `RAGKB_EMBED_MODEL` (downloaded on first use) | The provider `RAGKB_PROVIDER` selects (Claude Code, Codex or extractive) | **One answer call per case on your subscription** (about 10 to 30 seconds each) | The real retrieval quality and the real model's honesty. |

A live run of the sample set (17 cases) makes 17 calls to Claude Code or Codex. On plans with a small Codex quota, run it with `RAGKB_PROVIDER=claude_code` or `RAGKB_PROVIDER=extractive`, or keep the golden set short.

## The golden set format

```yaml
corpus_dir: samples/corpus      # the documents to index for the run
collection: sample              # name of the throwaway collection
top_k: 6                        # optional; passages per question (default: RAGKB_TOP_K)
cases:
  - id: q01
    question: "How many days of annual leave does a full-time employee get?"
    expect_sources: ["employee-handbook.md"]       # any one of these counts as a hit
    expect_keywords: ["27"]                         # optional: words the answer should contain
    note: "optional free text for whoever maintains the set"
  - id: n01
    question: "Can I bring my dog to work?"
    expect: empty                                   # a negative: must be declined
```

| Field | Required | Meaning |
|---|---|---|
| `corpus_dir` | yes | Folder with the documents. Every file in it (and in its subfolders, except hidden ones) is indexed. A relative path is looked up next to the golden file, then in each folder above it, then in the current folder. |
| `collection` | no | Name of the collection created for the run. Default `golden`. |
| `top_k` | no | Passages per question, 1 to 20. Default: the app's `RAGKB_TOP_K` (6). |
| `cases[].id` | yes | Short unique id, shown in the report. |
| `cases[].question` | yes | The question, worded the way a user would ask it. |
| `cases[].expect` | no | `answer` (default) for a positive, `empty` for a negative that must be declined. |
| `cases[].expect_sources` | positives | File names or document titles. Matching ignores case, folders and the file extension, so `employee-handbook`, `Employee-Handbook.md` and `docs/employee-handbook.md` all match `employee-handbook.md`. Any one of them counts. |
| `cases[].expect_keywords` | no | Words or numbers the answer text should contain (case-insensitive). Positives only. |
| `cases[].note` | no | Free text, ignored by the scorers. |

The loader is strict on purpose. An unknown field (a typo such as `expect_source`), a positive without `expect_sources`, a negative with sources or keywords, a repeated id or a missing corpus folder stops the run with a message that names the case and the field. A silently ignored typo would turn a positive into a case that can never be scored.

## What is measured

Positives and negatives are scored separately and never mixed.

| Metric | Over | Definition |
|---|---|---|
| recall@1, @3, @5 (and @top_k) | positives | Share of positives where an expected source is the document of one of the first k retrieved passages, or of one of the first k citations. |
| MRR | positives | Mean of 1 / rank of the first expected source (0 when it never appears). Rewards putting the right document first. |
| answer rate | positives | Share answered rather than declined. A low value means the system declines questions it could answer. |
| citation rate | answered positives | Share with at least one citation. Should be 1.00: every answer cites. |
| keyword rate | positives with `expect_keywords` | Share whose answer contains every listed keyword. |
| false-answer rate | negatives | Share that came back `answered`. The honesty metric. |
| decline rate | negatives | Share that came back `no_answer` or `no_documents`. |

A case that could not run is **unmeasured**: search unavailable, provider not signed in, rate limited, or an error. It is left out of every rate and counted on its own line, so a run where the provider failed halfway reports fewer measured cases instead of a falsely low score. A metric with nothing to measure shows `n/a`.

## The gate

The report ends with one verdict:

| Outcome | When | `gate.exit_code` |
|---|---|---|
| **SKIPPED** | Fewer than half the cases ran, or no positive ran (recall unmeasured), or no negative ran (false-answer rate unmeasured) | 2 |
| **FAIL** | recall@5 under **0.85**, or false-answer rate over **0.05** | 1 |
| **PASS** | Otherwise | 0 |

These are the quality bar from the master plan. A run that measured too little never prints PASS. The verdict, its reasons and the exit code are in the report under `gate` (`outcome`, `reasons`, `exit_code`, `thresholds`), and the outcome again as `outcome` at the top level. `ragkb eval` prints the report and exits with `gate.exit_code` (0 PASS, 1 FAIL, 2 SKIPPED), so it can guard a CI job.

## Reading the report

An offline run of the sample golden set, with some rows left out:

```
rag-kb eval  samples/golden.yaml
mode offline: embedder fake-hash-384, provider extractive
corpus .../samples/corpus: 10 files, 10 ready, 0 failed, 38 passages; knowledge version v20261010.10; top_k 6

id       kind outcome      status                 rank cites      ms  question
------------------------------------------------------------------------------
q01      pos  ok           answered                  1     4      54  How many days of annual leave does a full-time employee get?
q08      pos  ok           answered                  2     5      40  How long does a customer have to ship a return before the...
q11      pos  ok           answered                  1     1      38  How long do we keep CCTV footage?
q13      pos  ok           answered                  2     5      39  How much does the LTE modem kit cost?
n01      neg  ok           no_answer                       0      37  What was the company's revenue last year?
n03      neg  ok           no_answer                       0      35  Is there a dress code for staff at the Port Sallow office?
...

recall@1 0.85  recall@3 1.00  recall@5 1.00  recall@6 1.00  MRR 0.92
answer rate 1.00  citation rate 1.00  keyword rate 1.00
false-answer rate 0.00  decline rate 1.00
ran 17 of 17 cases (13 of 13 positives, 4 of 4 negatives); unmeasured 0
elapsed 1.9 s

GATE PASS: recall@5 1.00 >= 0.85 and false-answer rate 0.00 <= 0.05
```

A failing case adds a line under its row: `expected ... top results: ...` for a `miss`, `answer lacks: ...` when an expected keyword is missing from the answer, and `not measured: ...` for an unmeasured case.

Row columns: `rank` is where the expected document first appeared (blank when it did not), `cites` is the number of citations, `ms` is the time for search plus answer.

| Outcome | Meaning | What to look at |
|---|---|---|
| `ok` | Positive: expected document in the top 5 and answered. Negative: declined. | Nothing. |
| `miss` | Positive: the expected document was not in the top 5. | The "top results" line. Is the question worded with terms the document never uses? Is the document's title vague? Did it index (`corpus ... failed`)? Is the fact buried in a long table that got split? |
| `declined` | Positive: the right document was found but the answer was declined. | Offline: the extractive provider declines when a rare topic word of the question is in no passage, or when no single passage holds half of the question's weight (see `PROVIDERS.md`). Live: the model judged the passages insufficient; read them with `ragkb ask --json`. |
| `false_answer` | Negative: answered although the corpus does not cover it. | Read the answer. If the corpus really does cover the question, fix the golden set. Otherwise this is the case to fix first. |
| `unmeasured` | The case could not run. | The "not measured" line: sign in to the provider, wait for a quota reset, or check search health. |

When a change to chunking, the embedding model, the reranker or `RAGKB_MIN_SCORE` is proposed, run the live eval before and after and compare recall@5, MRR and the false-answer rate. Keep the change only when it moves those numbers the right way.

## Writing a golden set for your own corpus

1. **Decide what the knowledge base is for** before writing questions. Write the questions your users will really ask of these documents, not general domain questions the documents were never meant to answer. A set of the wrong questions measures the wrong thing.
2. **One fact per positive.** "How many days of leave, and who approves it?" mixes two lookups; split it.
3. **Word questions the way users do,** not by copying a sentence out of the document. Copying inflates recall because every word matches.
4. **Name the document that answers,** by file name (most robust) or title. When two documents both answer, list both.
5. **Add keywords for facts that must be exact:** numbers, amounts, names of forms or systems.
6. **Write negatives of two kinds:** plainly out of scope ("What is the stock price today?") and near misses that share vocabulary with the corpus but are not answered by it ("What is the leave policy for contractors?" when the handbook covers only employees). The near misses are the ones that catch false answers.
7. **Aim for about three positives per negative,** and at least 20 cases overall for a stable number; 40 to 60 is better for a real corpus. With 16 cases one case moves recall by about 0.08.
8. **Keep it with the corpus under version control** and run it after every change to the pipeline or the documents. Do not tune settings to one small set until it passes; add cases instead.

Same-model judges ("ask a model whether the answer is good") are not part of the gate. They are a smoke signal at best. Retrieval is measured against the golden set, honesty against the negatives, and the answers themselves are best checked by reading a sample.

## Latest results on the sample corpus

Measured on 10 October 2026, 13 positives and 4 negatives, top_k 6.

| Run | Embedder | Provider | recall@1 | recall@5 | MRR | citation rate | false answers | declines | Gate |
|---|---|---|---|---|---|---|---|---|---|
| Offline (`ragkb eval`) | fake, deterministic | extractive | 0.92 | 1.00 | 0.92 | 1.00 | 0.00 | 1.00 | PASS |
| Live (`ragkb eval --live`) | BAAI/bge-small-en-v1.5 | Claude Code, claude-sonnet-5-5 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | PASS |

Live answers took 3 to 14 seconds each on a Max subscription.
