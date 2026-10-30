# RAG Learning

A hands-on build of **Retrieval-Augmented Generation (RAG)** in Python — first over a hard-coded string, then over real PDF documents persisted to disk.

The repository is written as a learning path: two notebooks that each implement the same pipeline at a different level of realism, backed by a small `uv`-managed package.

---

## Table of contents

1. [What is RAG](#1-what-is-rag)
2. [How it works under the hood](#2-how-it-works-under-the-hood)
3. [Implementing RAG, step by step](#3-implementing-rag-step-by-step)
4. [What is in this repository](#4-what-is-in-this-repository)
5. [Setup and running](#5-setup-and-running)
6. [Parameters worth tuning](#6-parameters-worth-tuning)
7. [Known limitations and next steps](#7-known-limitations-and-next-steps)

---

## 1. What is RAG

### The problem it solves

A large language model is trained on a snapshot of public data and then frozen. It does not know:

- your private documents (internal wikis, PDFs, tickets, contracts),
- anything published after its training cut-off,
- anything narrow enough to never have appeared in training data.

You cannot fix all of that by prompting, because the knowledge simply is not in the weights. And when you ask about something it does not know, the model does not stay silent — it generates a plausible-sounding answer. That is a **hallucination**.

The two traditional ways to give a model knowledge:

| Approach | What it does | Cost | When it breaks |
| --- | --- | --- | --- |
| **Fine-tuning** | Further trains the weights on your data | Expensive, needs thousands of examples, re-run whenever data changes | Cannot cite sources, cannot delete a fact, stale the moment documents change |
| **RAG** | Fetches the relevant passages at question time and puts them in the prompt | Cheap, no training, updates by re-indexing a file | Retrieval returns the wrong passages |

### Definition

> **Retrieval-Augmented Generation** is a pattern where, before the model answers, the system **retrieves** the most relevant pieces of an external knowledge base and **augments** the prompt with them, so the model **generates** its answer from supplied evidence instead of from memory alone.

The model is not taught anything new. Its context window — the input it reads before answering — is filled with the exact paragraphs needed for this one question. Knowledge lives outside the model, in a searchable index.

The consequence is that the two failure modes become separable and debuggable:

- **Bad answer because retrieval found nothing useful** → fix the index, chunking, or embedding model.
- **Bad answer because the model ignored good context** → fix the prompt or swap the model.

In a fine-tuned model these are the same opaque failure.

### The two phases

RAG is not one pipeline but two, with different trigger cadences:

```
INDEXING — runs once per document change

  documents ──► split into chunks ──► embed each chunk ──► vector store


SERVING — runs once per user question

  question ──► embed ──► search ──► top-k chunks
                                             │
                                             ▼
                                 prompt = chunks + question ──► LLM ──► answer
```

Everything before the vector store is **indexing**. Everything from the question onwards is **serving**. The rest of this document follows that split.

---

## 2. How it works under the hood

This section follows one question all the way through, using the same components this repository uses.

### 2.1 Embeddings: text to coordinates

An embedding model is a small transformer that maps a piece of text to a fixed-length list of numbers — a **vector**.

```
"OneLake is a single, unified, logical data lake"  ──►  [ 0.021, -0.114, 0.087, ... ]  (384 numbers)
```

`all-MiniLM-L6-v2`, used here, produces **384-dimensional** vectors.

The property that makes RAG work is geometric: the model is trained so that texts with similar *meaning* land close together, even when they share no words. "What is the company data lake?" and "OneLake is a single, unified logical data lake" end up near each other, while "the cat sat on the mat" is far away. This is why the system can answer a question whose wording never appears in your document.

The model does not compare words. It compares positions in a 384-dimensional space that it learned to organise semantically.

Notably, `all-MiniLM-L6-v2` truncates input at **256 tokens**. Any chunk longer than that is silently cut. This constrains chunk sizing — see [§6](#6-parameters-worth-tuning).

### 2.2 Similarity: how "closest" is measured

What you want to compare is direction, not magnitude — hence **cosine similarity**:

```
similarity(a, b) = (a · b) / (‖a‖ · ‖b‖)
```

This is the dot product of the two vectors divided by their lengths, bounded to `[-1, 1]`. A value of `1` means identical direction; `0` means unrelated.

Chroma stores a **distance** instead, and returns the smallest ones first. This collection is created without a `hnsw:space` override, so Chroma uses squared Euclidean distance. `all-MiniLM-L6-v2` vectors are all of similar magnitude, and for unit-length vectors the two measures are exactly equivalent:

```
‖a - b‖² = 2 - 2·cos(a, b)
```

`cos = 1` → distance `0`; `cos = 0` → distance `2`. The ranking is the same as cosine ranking, only the numbers differ.

Those numbers are also the cheapest relevance signal available. Measured on this repository's `Vector/` store with these chunks:

| Question | Best distance | Verdict |
| --- | --- | --- |
| "What are OneLake shortcuts?" | `0.73` | answered by the PDF |
| "How do I create a lakehouse?" | `0.62` | answered by the PDF |
| "What is OneLake and why do we need it?" | `0.91` | answered by the PDF |
| "How does Direct Lake mode work?" | `1.14` | weakly retrieved, still on topic |
| "What is the capital of France?" | `1.62` | not in the PDF |
| "Who won the 2018 football world cup?" | `1.58` | not in the PDF |
| "What is the recipe for sourdough bread?" | `1.68` | not in the PDF |

Nothing sits between `1.14` and `1.58`, which is what makes a cutoff at `1.25` work: it separates "the document is being asked about" from "the document is not". Distances are corpus-specific — measure your own before picking a threshold instead of copying this one.

### 2.3 Storage: the index in `Vector/`

Scanning every vector per query would be O(n), unacceptable at scale. Chroma builds an approximate nearest-neighbour index — **HNSW** (Hierarchical Navigable Small World), a multi-layer graph where each vector links to its near neighbours. A query starts at a sparse top layer, greedily descends to the region that looks promising, and lands in the dense bottom layer where it does a local search. Finding a near-optimal neighbour takes roughly logarithmic time instead of linear.

That is exactly what the files in `Vector/` are:

| File | Contents |
| --- | --- |
| `chroma.sqlite3` | Collection metadata, document text, chunk metadata (source file, page number), and the vector-to-ID mapping |
| `<collection-id>/data_level0.bin` | The dense base layer of the HNSW graph: the raw vectors |
| `<collection-id>/header.bin` | Index parameters — dimensionality, offsets, element counts |
| `<collection-id>/length.bin` | Number of neighbours per node |
| `<collection-id>/link_lists.bin` | The neighbour links being built |

Because the vectors live here, you pay for embedding **once**. Re-running a query only embeds the question.

### 2.4 Chunking: why documents are cut up

You might ask why not embed the whole PDF as one vector. Three reasons:

1. **Precision.** One vector for 382 pages is an average of everything in it and matches nothing specifically. Chunks keep one topic per vector.
2. **Fit.** The context window is finite, and the embedding model truncates at 256 tokens anyway.
3. **Attribution.** Small chunks let you answer "what exactly does the context say and where".

So a document is split into overlapping windows. `RecursiveCharacterTextSplitter` tries to break at the largest natural boundary first — paragraphs (`\n\n`), then lines (`\n`), then sentences, then words — so chunks stay semantically coherent instead of slicing mid-sentence.

**Overlap** exists because a boundary is arbitrary. A sentence defining OneLake might straddle the cut; the overlap repeats the tail of the previous chunk at the start of the next, so at least one chunk contains it whole.

### 2.5 Retrieval: one question, k neighbours

```
question "What is OneLake and why do we need it?"
   │
   ├─ embed with the SAME model used for the documents   ──►  [0.013, 0.092, ...]
   │
   └─ Chroma: nearest k
         ├─ page 3    (0.913)  "OneLake is a single, unified, logical data lake..."
         ├─ page 3    (0.913)  the same text again — the file was ingested twice
         ├─ page 142  (0.930)  "OneLake accepts almost all of the same headers..."
         └─ page 374  (1.005)  "defined in the OneLake experience. SQL permissions..."
```

That second row is real, and it is why raw retrieval is not good enough: half of the model's context budget goes on a duplicate, and a one-line list of distractors becomes a two-line list. `rag_learning.rag.retrieve()` over-fetches, collapses repeated text, and drops anything past the distance cutoff before the prompt is built.

**The query must be embedded by the same model as the documents.** Two different embedding models produce two incompatible coordinate systems; the vectors are meaningless to each other. This is why `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")` appears in every notebook before both writing and reading — a single shared instance.

Retrieval is **semantic**, not keyword. It will miss an exact identifier like `abfss://...@onelake.dfs.fabric.microsoft.com` if no nearby text talks about ABFS paths, and it will happily return a conceptually related page that never uses the asked word.

### 2.6 Augmentation and generation

The retrieved passages are formatted into the prompt as plain text. The naive version — the one this repo started with — looks like this:

```python
response = llm.invoke(
    f"What is AI? you can answer the question using this context: {context}"
)
```

`context` is a list of `Document` objects, and its `str()` form dumps each `page_content` with its metadata. The model does then read the context, and its job shifts from *recall* to **reading comprehension** — a task models are much better at.

But that prompt has two holes, and both let the model answer from its training data instead of your index:

1. **"You can answer using this context" reads as an option, not a rule.** Nothing forbids the model from preferring what it already knows about OneLake. It has seen Microsoft documentation during pretraining, so it can produce a fluent, plausible answer that never touches your passages.
2. **Retrieval always returns something.** With `k=3` you get three passages even when the question is unrelated to your documents. The model receives irrelevant text labelled "context", finds it useless, and answers from memory — with no signal that retrieval failed.

A third, quieter hole: when retrieval returns nothing at all, `context` is `[]` and the f-string silently becomes *"What is AI? you can answer the question using this context: []"*. The instruction to use context is still there and still unenforceable.

The fix has to be structural, not stylistic. One rule in the prompt is worth nothing if the retrieval layer keeps handing over distractors, and a strict retrieval cutoff is worth nothing if the prompt allows memory. Both have to hold, so both live in [src/rag_learning/rag.py](src/rag_learning/rag.py):

```python
GROUNDING_PROMPT = """You answer questions from a document index. Use ONLY the numbered passages below.

Rules:
- Every statement must be supported by a passage.
- Do not use your own knowledge, do not guess, and do not fill gaps from memory.
- Cite the passage you used inline, e.g. [1].
- If the passages do not contain the answer, reply with exactly: {refusal}

Passages:
{context}

Question: {question}

Answer:"""
```

and the pipeline refuses *before* reaching the model when retrieval comes back empty:

```python
passages = retrieve(store, question, k=k, max_distance=max_distance)
if not passages:
    return GroundedAnswer(answer=NO_CONTEXT_ANSWER)   # no model call at all
```

That early return is the part that actually guarantees the answer is not from memory: with no evidence, no generation happens, so there is nothing to hallucinate. The refusal is fixed text.

Passages are numbered by source so citations are checkable:

```
[1] fabric-onelake.pdf, page 3
OneLake, the OneDrive for data Article 07/25/2024 OneLake is a single, unified...

[2] fabric-onelake.pdf, page 142
OneLake accepts almost all of the same headers as Storage...
```

Under the hood, the whole prompt is tokenised and processed by the model's attention mechanism in a single forward pass, then tokens are generated one at a time. Because the context is in the input, it can attend to it directly. Nothing was stored or learned; the knowledge evaporates when the request ends. Nor is the refusal a memory either — it is a Python string, returned without a network call.

### 2.7 Where RAG still fails

Knowing these is what separates a working demo from a usable system:

| Failure | Cause | Status here |
| --- | --- | --- |
| Confident answer from the model's own knowledge | No boundary between context and memory | **Fixed** — prompt forbids outside knowledge, and an empty retrieval returns before the model is called |
| Same page returned repeatedly | Duplicate ingestion into the collection | **Fixed at query time** — `retrieve()` collapses repeats. The store still holds them; reindexing is a separate job |
| Irrelevant chunks crowd the window | `k` too large, no floor on relevance | **Fixed** — passages past the distance cutoff are dropped |
| No attribution | Metadata never surfaced | **Fixed** — passages are numbered in the prompt, `result.sources` carries the labels |
| Answer not in the retrieved context | Retrieval miss — chunking or embedding mismatch | **Open** — tune chunk size, use a stronger embedding model, add a reranker |
| Context present but answer is wrong | Model misread it | **Open** — lower temperature helps, a stronger model helps more |
| Slow first query | Model weights loaded on first embed | **Open** — pre-load embeddings once at service start |

---

## 3. Implementing RAG, step by step

Each step below links to the notebook cell that implements it and lists the decision being made.

### Step 0 — Environment

```bash
uv sync                                   # creates .venv, installs pyproject dependencies
```

Create a `.env` file at the repository root (already present and gitignored; loaded via `load_dotenv()` in the first cell of both notebooks):

```env
GROQ_API_KEY=your_key_here
```

The stack: [`langchain`](https://python.langchain.com/) for pipeline primitives, `langchain-groq` for the LLM, `langchain-huggingface` + `sentence-transformers` for embeddings, `langchain-chroma` for the vector store, `langchain-community` + `pypdf` for PDF loading.

### Step 1 — Load the documents

**Notebook:** `1.o.RAG_with_own_text.ipynb`, cell 3 · `RAG_PDF.ipynb`, cell 3

Every source is normalised to LangChain's `Document`: `page_content` (the text) plus `metadata` (where it came from).

```python
from langchain_core.documents import Document

docs = [
    Document(
        page_content=my_text,
        metadata={"source": "wikipedia", "Document_id": "1"},
    )
]
```

For PDFs, the loader creates one `Document` per page and fills metadata automatically:

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("./Docs/fabric-onelake.pdf")
docs = loader.load()        # 382 pages → 382 Documents with source/page/page_label
```

Metadata is what later lets you cite "fabric-onelake.pdf, page 3" instead of an anonymous paragraph. Keep it from the first step; inferring provenance afterwards is not possible.

The `Docs/fabric-admin.pdf` file is present but not yet indexed by either notebook.

### Step 2 — Split into chunks

**Notebook:** `1.o.RAG_with_own_text.ipynb`, cell 4 · `RAG_PDF.ipynb`, cell 4

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)      # 652 chunks from the 382-page PDF
len(chunks)
```

The one-line text demo uses the tighter `chunk_size=500, chunk_overlap=50`, which is what produced the four short chunks in its output — and the reason the last two chunks in that output overlap.

`split_documents` (not `split_text`) keeps each chunk's metadata attached, so every chunk still knows its source page.

### Step 3 — Choose and load the embedding model

**Notebook:** `1.o.RAG_with_own_text.ipynb`, cell 5 · `RAG_PDF.ipynb`, cell 5

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

Runs locally on CPU, ~90 MB, no API key, 384 dimensions. The "Loading weights: 103/103" stderr line is the model being materialised — cheap here, but worth doing once at service start rather than per request.

### Step 4 — Embed and store

**Notebook:** `1.o.RAG_with_own_text.ipynb`, cell 6 · `RAG_PDF.ipynb`, cell 6

In-memory, ephemeral:

```python
from langchain_chroma import Chroma

db = Chroma.from_documents(chunks, embeddings)
```

Persisted to disk — this is what created `Vector/`:

```python
from langchain_chroma import Chroma

Chroma.from_documents(chunks, embeddings, persist_directory="./Vector/")
```

Reindexing creates a new collection; it does not overwrite silently. That is how the current `Vector/` ended up holding every chunk twice: `add_documents` was run against an already-populated collection. The line is kept commented out in the retrieval cell as a reminder.

### Step 5 — Open an existing index

**Notebook:** `RAG_PDF.ipynb`, cell 7

On later runs, skip steps 1–4 entirely and attach to the persisted store:

```python
vector_persist = Chroma(
    persist_directory="./Vector",
    embedding_function=embeddings,
)
```

The same `embeddings` object is mandatory — a different model here would return nonsense against the stored 384-dimensional vectors.

### Step 6 — Retrieve, filtered

**Notebook:** `1.o.RAG_with_own_text.ipynb`, cell 7 · `RAG_PDF.ipynb`, cell 8 (raw) and the first half of cell 9 (filtered)

Raw retrieval is worth seeing once, because it shows the two problems the filtered version fixes:

```python
# RAG_PDF.ipynb, cell 8 — nearest neighbours, duplicates included
vector_persist.similarity_search("What is OneLake and why do we need it?", k=3)
```

That call returns `page 3` and `page 3` again — the store holds 1304 vectors for 652 chunks, so every hit is doubled. Use the filtered version instead, which is what every later cell relies on:

```python
from rag_learning.rag import retrieve

context = retrieve(db, "what is AI?", k=3)                          # in-memory demo
context = retrieve(vector_persist, "What is OneLake and why do we need it?", k=4)   # PDF demo
```

`retrieve` over-fetches, collapses repeated text, drops passages above `DEFAULT_MAX_DISTANCE`, and returns `k` *distinct* passages. It is the single place where `k` and the cutoff live.

### Step 7 — Generate, grounded

**Notebook:** `1.o.RAG_with_own_text.ipynb`, cell 8 · `RAG_PDF.ipynb`, cell 9

```python
from langchain_groq import ChatGroq
from rag_learning.rag import answer_from_context

llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0)

result = answer_from_context(db, llm, "What is AI?", k=3)
print(result.answer)        # cites [1], [2] ... by passage
print(result.sources)       # ('wikipedia', ...) — the evidence behind the answer
```

Verified output from the text demo:

```
Artificial intelligence (AI) is the capability of computational systems to perform
tasks typically associated with human intelligence, such as learning, reasoning,
problem-solving, perception, and decision-making [1]. It is a field of research in
engineering, mathematics, and computer science ... [1]
```

`temperature=0` keeps the answer deterministic and anchored to the passages — desirable for factual lookup, undesirable for creative writing.

`GroundedAnswer` also exposes `prompt`, so you can read the exact string the model saw. When an answer looks wrong, check the prompt before blaming the model.

### Step 8 — Prove it is not answering from memory

**Notebook:** `1.o.RAG_with_own_text.ipynb`, cell 9 · `RAG_PDF.ipynb`, cell 10

Ask something the corpus cannot answer:

```python
off_topic = answer_from_context(vector_persist, llm, "Who won the 2018 football world cup?")
print(off_topic.answer)
print(off_topic.used_retrieval)      # False
```

```
The indexed documents do not cover this. No retrieved passage was close enough
to the question to answer from.
False
```

`used_retrieval == False` means no passage passed the cutoff and **the model was never called**. That is the assertion worth putting in a test suite: a RAG system that answers an out-of-corpus question is not grounded, whatever the prompt says.

To read the evidence rather than trust it, print `format_context(result.passages)` — the numbered blocks the model was allowed to use. If the answer contains a claim that is not in those blocks, the prompt is being ignored and that is a finding, not a phrasing problem to paper over.

If you want the same pipeline as an LCEL chain, wrap the module rather than re-deriving it:

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

chain = (
    RunnableLambda(lambda question: build_prompt(question, retrieve(vector_persist, question)))
    | llm
    | StrOutputParser()
)

print(chain.invoke("What is OneLake and why do we need it?"))
```

The retrieval filter and the grounding prompt stay in one module either way.

---

## 4. What is in this repository

```
RAG Learning/
├── 1.o.RAG_with_own_text.ipynb   RAG over an in-memory string — the minimal pipeline
├── RAG_PDF.ipynb                 RAG over a 382-page PDF, persisted to disk
├── Docs/
│   ├── fabric-onelake.pdf        Indexed by RAG_PDF.ipynb
│   └── fabric-admin.pdf          Present, not yet indexed
├── Vector/                       Persisted Chroma store built by RAG_PDF.ipynb
│   ├── chroma.sqlite3            Text, metadata, vector-ID map
│   └── 9abbae0c-.../             HNSW index binaries (1304 vectors for 652 chunks)
├── src/rag_learning/
│   ├── rag.py                    Grounded pipeline: retrieve, format, prompt, answer
│   └── __init__.py               Package exports (`rag-learning` console script)
└── pyproject.toml                uv-managed dependencies
```

### The two notebooks side by side

| | `1.o.RAG_with_own_text.ipynb` | `RAG_PDF.ipynb` |
| --- | --- | --- |
| Source | Hard-coded string | `./Docs/fabric-onelake.pdf` |
| Loader | `Document(...)` | `PyPDFLoader` |
| `chunk_size` / `chunk_overlap` | `500` / `50` | `1000` / `100` |
| Chunks produced | 4 | 652 |
| Vector store | In-memory `Chroma` | Persisted to `./Vector` |
| On-topic distance range | `0.43`–`0.76` | `0.62`–`1.14` |
| Off-topic distance range | `1.86`–`1.99` | `1.58`–`1.75` |
| Purpose | Isolate each step, see every intermediate value | Realistic ingestion with metadata and reuse across sessions |

Read them in that order. The first shows the mechanics with no files to inspect; the second shows what changes when the corpus is a real document and the index must survive a kernel restart. Both end with the same two cells: a grounded answer, and an off-topic question that is refused without a model call.

### The grounded answering module

`src/rag_learning/rag.py` holds the logic the notebooks call, so the pipeline is reusable without copy-paste:

```
retrieve()            over-fetch -> drop duplicates -> drop above cutoff -> take k
format_context()      numbered "[1] file, page 3" blocks
build_prompt()        the grounding template + context + question
answer_from_context() retrieve -> refuse if empty -> prompt -> GroundedAnswer
```

`GroundedAnswer` carries `answer`, `passages`, `sources`, `prompt`, and `used_retrieval`, which is what makes grounding assertions possible instead of aspirational.

---

## 5. Setup and running

**Prerequisites:** Python ≥ 3.14 (per `pyproject.toml`; the notebooks were run on 3.14.2) and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Then set `GROQ_API_KEY` in `.env` and open the notebooks in VS Code, selecting the `.venv` interpreter as the kernel.

To rebuild the index from scratch, delete `Vector/` and re-run the `Chroma.from_documents(..., persist_directory="./Vector/")` cell. Deleting the directory is the clean way to remove stale collections accumulated from earlier runs — there is no partial-reindex step.

---

## 6. Parameters worth tuning

| Parameter | Current value | Effect of increasing | Effect of decreasing |
| --- | --- | --- | --- |
| `chunk_size` | `1000` (PDF), `500` (text) | More context per hit, fewer chunks, but dilutes the embedding and risks exceeding the 256-token embedding limit | Sharper vectors, more chunks, more boundary loss |
| `chunk_overlap` | `100` / `50` | Nothing falls on a boundary, higher storage cost | Cheaper, but definitions get cut in half |
| `k` | `3` / `4` | Better recall, more irrelevant context, more tokens per request | Fewer distractions, higher chance of missing the answer |
| `max_distance` | `1.25` | More passages reach the model, more chance of answering, higher chance of a memory-flavoured answer | Tighter grounding, more refusals on questions the corpus does answer |
| Embedding model | `all-MiniLM-L6-v2` (384-d) | Larger models (e.g. `bge-base`, `nomic-embed-text`) retrieve noticeably better and cost more compute | Faster, cheaper, weaker on domain vocabulary |
| LLM `temperature` | `0` | More variation, more drift from the context | More deterministic, more grounded |

The two knobs with the largest quality impact, in order: **embedding model**, then **chunk size**.

---

## 7. Known limitations and next steps

Grounded answering is in place. What remains, in rough order of value:

- **The store still holds every chunk twice.** 1304 vectors for 652 chunks. `retrieve()` hides it at query time, but it wastes disk and skews the HNSW graph — reindex from an empty `Vector/` with the `Chroma.from_documents(..., persist_directory="./Vector/")` cell.
- **The relevance cutoff is tuned to this corpus.** `1.25` sits in the gap measured for these 1000-character PDF chunks. Change the chunk size or the embedding model and the band moves; re-measure rather than trusting the constant.
- **Retrieval quality is the weak link.** "How does Direct Lake mode work?" scores `1.14` and lands on pages 7–8 rather than the page that explains Direct Lake. Larger chunks of Microsoft Learn PDF text are noisy: navigation lists, "Was this page helpful?", page furniture. Cleaning the text before splitting would improve every answer more than prompt tuning will.
- **No reranking.** `retrieve` takes the nearest by embedding distance alone. A cross-encoder rerank over the top ~20 candidates is the standard next step for multi-document corpora.
- **One PDF only.** `Docs/fabric-admin.pdf` is present and unindexed; ingestion is per-file, with no dedupe across files.
- **No evaluation harness.** The off-topic refusal in Step 8 is one assertion. A small set of question → expected-page pairs turned into a score is the fastest way to know whether a change helped or hurt.
- **No streaming or token budget.** The whole answer arrives at once. On the free Groq tier a 4-passage prompt can exceed the output-tokens-per-minute limit, which is why `k` is not pushed higher.
- **Some notebook outputs predate the change.** Cells I edited had their output cleared; others still carry results from earlier out-of-order runs. Re-run all cells before quoting numbers.

