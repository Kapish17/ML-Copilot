# ML Copilot — Retrieval Layer

Finds the evidence that bears on a question, from the two kinds of knowledge
this project already produces: the documentation that says how the system
works, and the experiment records that say what was actually run.

> **This layer returns evidence, never an answer.** It ranks passages and
> attributes them; it never writes prose, draws a conclusion or interprets a
> result. Turning evidence into an answer is `llm/`'s job, and that layer
> treats what comes from here as authoritative — **the model is not the source
> of truth; retrieved evidence is.** Everything below decides *what the model
> gets to see*, and makes every sentence of an answer traceable to a passage
> that actually exists.
>
> `rag/` does not import `llm/`. Retrieval stays usable, and testable, with no
> model involved, and a test enforces it.

This layer is also what the agent's `search_knowledge` tool wraps. That tool
is search only: there is deliberately no agent tool that indexes, edits or
deletes anything here, so nothing an agent does can change what the knowledge
base contains. See `agent/README.md`.

**Not implemented:** Qdrant, PostgreSQL, any vector database, LangChain,
LangGraph, autonomous tool calling, and any hosted embedding API.

## What RAG means here

Retrieval-augmented generation, in general, means looking things up before
answering instead of relying on what a model happens to remember. This is not
a "chat with your PDFs" feature bolted onto the project — it retrieves two
specific things the project knows about itself:

**Project documentation.** How preprocessing avoids leakage, why
cross-validation chooses the model and the test set is measured once, what the
API returns, what is deliberately not implemented.

**Experiment history.** Which models were tried on which dataset, what won,
what it scored, which features mattered. This is the part a general-purpose
retriever could not give you: it is knowledge this system generated about
itself, and without it a future assistant could describe the pipeline but
could not answer "did the forest ever beat the baseline on the renewals
data?"

The whole shape, with this layer's part on the left:

```
question → retriever → relevant documentation + experiment history
                     → LLM → grounded answer with citations
           ╰─ rag/ ─╯   ╰──────────── llm/ ────────────╯
```

This layer ends at the evidence. `llm/` takes it from there, checks that every
citation an answer makes was actually retrieved, and rejects the answer if not
— see `llm/README.md`.

## The pipeline

```
                 ┌─────────────────────┐
  README.md ────►│                     │
  ml/README.md ─►│  documentation      │
  backend/… ────►│  ingestion          │──┐
                 └─────────────────────┘  │
                                          ▼
                 ┌─────────────────────┐ ┌──────────┐ ┌────────────┐
  ExperimentStore│  experiment         │ │ chunking │ │ embedding  │
  ──────────────►│  ingestion          │►│ (heading │►│ provider   │
                 └─────────────────────┘ │  aware)  │ │            │
                                         └──────────┘ └─────┬──────┘
                                                            ▼
                                                     ┌──────────────┐
                                                     │ vector store │
                                                     │  (local)     │
                                                     └──────┬───────┘
                                                            ▼
  question ──► embed ──► metadata filter ──► cosine ranking ──► results
                                                                 + citations
```

## Structure

```
rag/
├── config.py              RagConfig — every knob, one place
├── errors.py              Typed errors; no HTTP meaning
├── documents.py           Document, Chunk, deterministic ids, content hashes
├── chunking.py            Structure-aware Markdown splitting
├── citations.py           Stable references back to a source
├── manifest.py            What is indexed, from what, with which embeddings
├── indexing.py            RagIndexer — chunk, embed, upsert, incrementally
├── evaluation.py          Recall@K and Hit@K over a fixed query set
├── embeddings/
│   ├── base.py            The EmbeddingProvider contract
│   ├── hashing.py         Offline default: no download, no key, no network
│   └── sentence_transformer.py   Optional neural provider, lazily loaded
├── stores/
│   ├── base.py            The VectorStore contract
│   └── local.py           Persistent index: vectors.npy + records.jsonl
├── ingestion/
│   ├── documentation.py   Project Markdown, from an allowlist
│   └── experiments.py     ExperimentRun → readable structured facts
├── retrieval/
│   ├── results.py         RetrievalResult / RetrievalResponse
│   └── service.py         RetrievalService — embed, filter, rank, attribute
├── prompts/               Placeholder for the LLM commit (empty)
├── tests/
└── requirements.txt
```

## Documentation ingestion

An **allowlist, not a crawl**. `RagConfig.documentation_files` names the files
to index — the five READMEs by default — plus one optional `docs/` directory.
Source code, datasets, model artefacts, virtual environments, `.git` and the
experiment store are not merely skipped; they are never candidates.

Three checks stand between a path and the index, and none of them is
overridable by configuration:

- **Containment** — every path is resolved and must lie inside the project
  root. A configured `../../etc/passwd` is refused.
- **Forbidden names** — `.env` and anything whose name suggests a credential
  (*secret*, *token*, *password*, `.pem`, `.key`) is refused even when
  explicitly listed. A key that reaches the index is a key in every future
  answer.
- **Extension and size** — Markdown only, up to a configured byte limit.

A document's title comes from its first heading, so a retrieved chunk of
`ml/README.md` is labelled "ML Copilot — ML Layer" rather than a file path.

## Experiment ingestion

Each `ExperimentRun` becomes one document of structured Markdown, with a
heading per section so the chunker splits it where the subject changes:

```
# Experiment exp_84a8d53a1f5f_20260828T134457Z_e420

## Overview
Experiment ID: exp_84a8d53a1f5f_20260828T134457Z_e420
Task: classification
Selected model: logistic_regression
Primary metric: f1

## Dataset
Dataset fingerprint: 86494cff7a45cb7f
Rows: 240
Target column: renewed

## Model selection
Selection strategy: cross_validation
Selection score: 0.8623 ± 0.0465
Candidate results:
- logistic_regression: 0.8623 ± 0.0465 (succeeded)
- random_forest_classifier: 0.7998 ± 0.0474 (succeeded)

## Final evaluation
Final test score: 0.8750
Baseline: most_frequent
- absolute_improvement: 0.8750
- beats_baseline: yes

## Explainability
Top features by importance:
1. income: 0.9962
2. tenure_months: 0.8579
```

**Only stored facts are written.** Every line is a value taken from the
record. This module never writes "the model performed well" or "the forest was
the better choice", because neither is in the record — that reading is a job
for a future model with the evidence in front of it, and putting invented
prose into the index would mean retrieving and citing it as fact later.

### Searchable metadata

Every chunk of an experiment carries the fields that make filtered retrieval
possible:

| Key | Example |
| --- | --- |
| `source_type` | `experiment` |
| `experiment_id` | `exp_84a8d53a1f5f_20260828T134457Z_e420` |
| `dataset_fingerprint` | `86494cff7a45cb7f` |
| `task_type` | `classification` |
| `target_column` | `renewed` |
| `selected_model` | `logistic_regression` |
| `primary_metric` | `f1` |
| `selection_score`, `test_score`, `is_unbiased`, `tags`, `created_at` | … |

### The dependency runs one way

```
ml/experiments  →  rag/ingestion  →  rag/retrieval
```

RAG reads the experiment store. Nothing in `ml/experiments` knows this layer
exists, so an experiment can be recorded with no index present, and the index
can be rebuilt from the store at any time. `sync_experiments()` is that
operation.

## Chunking

Cutting every N characters is the obvious approach and the wrong one: it
severs sentences, separates a table from its header, and produces passages
that no longer say what they are about. A retrieved chunk has to stand alone,
because a model will see it without the document around it.

So the splitter follows the document's own structure:

1. **Sections.** Markdown headings mark where the subject changes. A section
   becomes a chunk when it fits, and its heading path travels with every chunk
   it produces — a passage about one-hot encoding stays labelled
   "Preprocessing › Feature groups" even when read alone.
2. **Paragraphs.** A section too long for one chunk is split on blank lines,
   never mid-paragraph, and paragraphs are packed until the next would
   overflow.
3. **Hard wrapping, last.** A single paragraph longer than the limit is cut on
   a line boundary where possible, and only then at a character offset.

Two further rules earn their keep. **Fenced code blocks are never split** —
the fence markers have to stay with their content. And **tiny fragments are
merged** into a neighbour, because a heading with one line under it retrieves
nothing useful and dilutes the ranking.

Overlap (150 characters by default) repeats the tail of one chunk at the head
of the next, so a sentence falling across a boundary is findable from either
side.

## Embeddings

Retrieval depends on the `EmbeddingProvider` interface, never on a model.
Swapping providers is a configuration change; the chunker, the store and the
retrieval service do not know the difference. Every provider guarantees a
fixed dimension, unit-length vectors (so a dot product *is* the cosine) and
determinism — the same text embeds identically in this process and the next.

### The default: `hashing` — offline, deterministic, no download

Two hashed n-gram channels from scikit-learn's `HashingVectorizer`,
concatenated and L2-normalised:

- **words**, unigrams and bigrams, matching shared terminology;
- **characters within word boundaries**, 3–5 grams, which survive the
  morphology a word-only match trips over — *leak*, *leaks*, *leakage*,
  *leakage-safe* share substrings.

It is **stateless**: no vocabulary to fit, so nothing extra is persisted,
adding a document never invalidates existing vectors, and the index is
reproducible on any machine. Default dimension 512. No new dependency: numpy
and scikit-learn are already required by the ML layer.

**What it is, honestly.** These are *term-overlap* embeddings. They match a
question to a passage by the words and word-fragments they share. They do
**not** capture meaning: a question about "avoiding target leakage" will not
find a passage that only ever says "keeping the test set untouched". For a
corpus of project documentation and experiment records — where questions and
documents share vocabulary because both are about this project — it retrieves
well enough to be useful, and `rag/evaluation.py` reports exactly how well
rather than asserting it.

### The option: `sentence_transformer` — real semantics, real cost

| | |
| --- | --- |
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Dimension | 384 |
| Package footprint | pulls in PyTorch and `transformers`: roughly **4 GB** installed, against about 1 GB for the whole project without it |
| Model download | ~90 MB from the Hugging Face hub on first use, cached afterwards. **The only network access in this layer**, and only if this provider is selected |
| Runtime | CPU is fine; a few hundred short chunks embed in a few seconds |

Everything is lazy: the module imports without `sentence_transformers`
installed, constructing a provider downloads nothing, and the model loads on
the first call to embed. It is not installed by default — see
`rag/requirements.txt`.

```bash
pip install sentence-transformers
export RAG_EMBEDDING_PROVIDER=sentence_transformer
python -c "from rag import RagConfig, RagIndexer, config_from_env; RagIndexer(config_from_env()).rebuild()"
```

Changing provider **requires a rebuild** — vectors from two providers are not
in the same space. The manifest records which provider built the index, and
the indexer rebuilds rather than mixing them.

**No document or experiment content leaves the machine** with either provider.

## Vector store

Three files in one directory, kept in step:

```
rag/index/
  vectors.npy    (rows, dimension) float32, one row per chunk
  records.jsonl  one JSON object per chunk, in the same row order
  manifest.json  what was indexed, from what, with which embeddings
```

Row *i* of the matrix is the embedding of the chunk on line *i* of the record
file. That correspondence is the whole design, and it is checked on load: a
matrix and a record file of different lengths would return the wrong text for
the right score, so the store refuses to open rather than guess.

**Writes are atomic.** Both files go to a temporary name, are flushed,
`fsync`ed and moved into place with `os.replace`. A process killed mid-save
leaves the previous complete index, never a half-written one.

This is a real index on disk, not a dictionary that forgets: it survives
process restarts, and reopening the directory is all that is needed. It is
deliberately simple — an exact brute-force scan rather than an approximate
nearest-neighbour structure — which is the right trade for thousands of chunks
and the wrong one for millions.

## Semantic retrieval

**The similarity metric is cosine.** Every provider returns unit-length
vectors, so cosine similarity is the dot product, and searching is one
matrix-vector product over the candidate rows. Scores run from `1.0` (identical
direction) through `0.0` (nothing in common) to `-1.0`.

```python
from rag import RagConfig, RetrievalService

service = RetrievalService(RagConfig())
response = service.search("How does the project prevent data leakage?", top_k=3)

for result in response:
    print(result.rank, round(result.score, 3), result.citation)
```

```
1 0.234 docs:readme#leakage-prevention
2 0.227 docs:ml-readme#leakage-prevention
3 0.220 docs:ml-readme#preprocessing
```

Ties break by insertion order, so an identical query against an identical
index returns an identical ranking every time.

`top_k` is capped by configuration, and `similarity_threshold` drops weak
matches — raise it to trade recall for precision. A response reports
`candidate_count` as well as its results, which distinguishes "the index has
nothing about this" from "the filter excluded everything".

## Metadata filtering

Filtering happens **before** ranking, not after. Asking for the five best
classification experiments searches classification experiments; it does not
rank everything and discard the regressions — which is how a post-hoc filter
silently returns two results when it was asked for five.

```python
response = service.search_experiments(
    "which model scored best on the test set",
    task_type="classification",
    dataset_fingerprint="86494cff7a45cb7f",
    top_k=5,
)
```

That is the hybrid path: metadata narrows the candidates, semantic similarity
ranks within them. Lexical BM25 hybrid search is not implemented.

## Citations

Every chunk carries a stable reference, and a reference is enough on its own
to find the passage again:

```
docs:ml-readme#leakage-prevention
experiment:exp_84a8d53a1f5f_20260828T134457Z_e420#final-evaluation
```

An experiment citation *is* the id the experiment store and the HTTP API
already use, so `GET /api/v1/experiments/exp_84a8…` resolves it. A
documentation citation is a file slug and a heading anchor.

The point is not the syntax. It is that when a future answer says "according
to experiment exp_84a8…", the claim can be checked. `parse_citation()` splits
one back into its parts.

## Retrieval evaluation

A retriever that returns five confident passages about the wrong subject is
worse than one that returns nothing, because a model will use them. So quality
is measured, not assumed:

- **Hit@K** — did *any* relevant document appear in the top K? The question
  that matters when one good passage is enough to answer from.
- **Recall@K** — what *fraction* of the relevant documents appeared? Notices a
  retriever that always returns its favourite document.

Both are measured at the **document** level. A question is answered by "the
section of `ml/README.md` about leakage"; which of that section's chunks
surfaced is an implementation detail, and requiring a specific chunk id would
break the measurement every time a paragraph moves.

```python
from rag import evaluate_retrieval
print(evaluate_retrieval(service, k=5).as_text())
```

```
Retrieval evaluation over 5 queries at k=5
  Hit@5:    100.00%
  Recall@5: 90.00%
```

The set in `DEFAULT_EVALUATION_QUERIES` is small, hand-written and
deterministic — five questions whose answers genuinely live in the indexed
documentation, each naming the documents that should be found. It is a
regression check, not a benchmark: it catches a chunker change that quietly
stops retrieving the leakage section, which is otherwise invisible.
`experiment_queries()` generates the equivalent for experiments, whose ids are
not known when the set is written.

**Answer quality is not evaluated here.** These metrics measure whether the
right passages are found. Whether an answer built from them is well grounded is
enforced separately, by the citation validator in `llm/grounding.py`; whether
it is *well written* is not measured anywhere yet.

## Indexing and reindexing

```python
from rag import RagConfig, RagIndexer
from ml.experiments import LocalExperimentStore

indexer = RagIndexer(RagConfig())
indexer.index_documentation()             # incremental
indexer.sync_experiments(LocalExperimentStore())
indexer.rebuild(LocalExperimentStore())   # from scratch
```

Three properties matter more than speed:

**Indexing twice is not indexing twice.** Document and chunk ids are derived
from content and position, so re-indexing an unchanged source overwrites the
same rows with the same values. The manifest short-circuits it earlier still:
an unchanged source hash is skipped without being chunked or embedded.

**A changed source leaves nothing stale.** When a document's hash changes its
old chunks are deleted before the new ones are written. A paragraph removed
from a README disappears from the index rather than lingering as an orphan
that still retrieves.

**A changed embedding provider invalidates everything.** The manifest records
which provider built the index; a mismatch triggers a rebuild rather than a
mix of incomparable vectors.

### The manifest

One entry per document — id, source type, reference, content hash, chunk count
and ids, timestamp — plus the embedding identifier for the index as a whole.
That is what lets the indexer decide what needs work. **No secret is written
to it**: hashes, counts, identifiers and timestamps only, and no API key is
involved in the first place.

## Determinism

- Document ids derive from source type and reference.
- Chunk ids derive from the document id, the position and the content.
- Content hashes are SHA-256 of the UTF-8 text.
- No random UUID appears anywhere in chunk identity.
- Ties in ranking break by insertion order.

Two machines indexing the same repository produce the same ids, and the same
query against the same index returns the same ranking.

## Privacy

Nothing is sent anywhere. The default embedding provider needs no API key, no
network and no download; the optional one downloads model weights and then
runs locally. No document or experiment content leaves the machine, document
contents are not logged, and the ingestion allowlist plus the forbidden-name
rules mean `.env` files, credentials and raw datasets are never candidates for
indexing.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAG_INDEX_DIR` | `rag/index` | Where the index is written |
| `RAG_EMBEDDING_PROVIDER` | `hashing` | `hashing` or `sentence_transformer` |
| `RAG_EMBEDDING_MODEL` | — | Model id for the neural provider |
| `RAG_EMBEDDING_DIMENSION` | `512` | Vector length for the default provider |
| `RAG_CHUNK_SIZE` | `1200` | Target maximum chunk, in characters |
| `RAG_CHUNK_OVERLAP` | `150` | Characters repeated across a boundary |
| `RAG_TOP_K` | `5` | Results returned when the caller does not say |
| `RAG_SIMILARITY_THRESHOLD` | `0.0` | Minimum cosine similarity |
| `RAG_MAX_QUERY_LENGTH` | `2000` | Longest query accepted, in characters |

Everything else — the minimum chunk size, the `top_k` cap, the batch size, the
documentation allowlist, the forbidden names — is a field on `RagConfig` with
a named default constant. No module hard-codes a number of its own.

`RagConfig.resolve_query()` applies the empty-query and length rules, so a
library caller and an HTTP client are held to exactly the same limits by the
same code.

## Over HTTP

Retrieval is exposed by the backend as `POST /api/v1/search`, and used by the
answering endpoint to gather the evidence an answer is built from. **POST
/api/v1/ask returns evidence-grounded answers; the LLM is not the source of
truth** — this layer is where that truth comes from.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "How is data leakage prevented?",
       "top_k": 5,
       "filters": {"source_types": ["project_documentation"]}}'
```

The endpoint is an adapter, not a second implementation. Its `filters` field
becomes a `build_metadata_filter(...)` call — the same pre-ranking filter a
library caller would build — so filtering exists in one place and never as
list comprehensions in a route. Ranking, scoring and citation identifiers are
this layer's, unchanged.

Three states this layer does not distinguish are distinguished at the edge,
because only an HTTP client is misled by conflating them: **no relevant
evidence** is a `200` with an empty list; **no index built yet** is a `503`
saying to run the indexer; **an index that cannot be read** is a `503` from
this layer's own `CorruptIndexError`. See `backend/README.md`.

Nothing in `rag/` knows any of this. It raises its own errors, and
`backend/app/core/knowledge_errors.py` is the single place that gives them a
code and a status.

## Setup and tests

```bash
pip install -r rag/requirements.txt   # numpy + scikit-learn, already present
pytest rag/tests                       # from the repository root
```

Tests use a deterministic fake embedding provider and small synthetic
documents and experiment records. **No test downloads a model or touches the
network.** The one test that exercises the real sentence-transformer provider
skips itself unless the package is installed and the model is already cached.

## Future vector database migration

`VectorStore` is an interface. A `QdrantVectorStore` implementing the same
five operations would replace `LocalVectorStore` without changing
`RetrievalService`, the document model, the embedding provider or any caller.
The local store is the right answer while the corpus is thousands of chunks
and one machine; it is the wrong one for millions of chunks, concurrent
writers, or history shared across machines. **Qdrant is not implemented.**

## Current limitations

- **No answers here.** This layer returns evidence; `llm/` turns it into an
  answer. Retrieval quality bounds answer quality — a question this layer
  cannot find evidence for is answered `insufficient_evidence`, not guessed at.
- **The default embeddings are lexical, not semantic.** A question phrased in
  words the documents do not use will retrieve poorly. The neural provider is
  the fix, at the cost of ~4 GB of dependencies and a model download.
- **Exact search only.** Every candidate row is scored on every query. Fine
  for thousands of chunks; wrong for millions.
- **No lexical hybrid.** Semantic similarity plus metadata filtering only; no
  BM25, no reciprocal-rank fusion, no reranker.
- **No concurrent-writer story.** Two processes indexing the same directory at
  once is undefined, though atomic writes mean neither leaves a truncated
  file.
- **Markdown only.** No PDF, HTML, notebook or docstring ingestion.
- **English only in practice.** Nothing is language-specific by design, but
  the chunker's heuristics and the default embeddings were tuned against
  English documentation.
- **No relevance feedback or reranking loop.** A search is a single pass: one
  embedding, one ranked list. Nothing learns from what a caller found useful,
  including the agent, which sees the same ranking any other caller does.
