# BasicRAG

**Local-first Retrieval-Augmented Generation** over any PDF document.  
Parsing, chunking, and embedding run entirely on your machine (zero token cost);  
only the final answer generation calls Groq's hosted LLM.

---

## How It Works

BasicRAG is a two-stage pipeline: **ingest once, query many times**.

### Stage 1 — Ingestion (`ingest.py`)

Converts a PDF into searchable vectors stored on disk.

```
                          ingest.py
  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │   PDF Document                                                  │
  │       │                                                         │
  │       ▼                                                         │
  │   ┌─────────────────┐                                           │
  │   │  PyMuPDF4LLM    │  Extracts text as layout-aware Markdown   │
  │   │  (PDF Parser)   │  preserving headings, tables, structure   │
  │   └────────┬────────┘                                           │
  │            │  List of page Documents                            │
  │            ▼                                                    │
  │   ┌─────────────────────────┐                                   │
  │   │  RecursiveCharacter     │  Splits text into overlapping     │
  │   │  TextSplitter           │  chunks of ~1000 characters       │
  │   │  (chunk=1000, overlap=200)  with 200-char overlap           │
  │   └────────┬────────────────┘                                   │
  │            │  List of chunk Documents                           │
  │            ▼                                                    │
  │   ┌─────────────────────────┐                                   │
  │   │  HuggingFace MiniLM     │  Converts each chunk into a      │
  │   │  (all-MiniLM-L6-v2)    │  384-dimension vector             │
  │   │  Runs on LOCAL CPU     │  (array of 384 decimal numbers)   │
  │   └────────┬────────────────┘                                   │
  │            │  Vectors + original text                           │
  │            ▼                                                    │
  │   ┌──────────────┐    ┌──────────────────┐                      │
  │   │  Chroma DB   │◄───│ SQLRecordManager │                      │
  │   │  (./chroma_db)    │ (./index_cache)  │                      │
  │   │              │    │                  │                      │
  │   │  Stores      │    │  Tracks chunk    │                      │
  │   │  vectors +   │    │  hashes for      │                      │
  │   │  text on     │    │  incremental     │                      │
  │   │  disk        │    │  updates         │                      │
  │   └──────────────┘    └──────────────────┘                      │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘
```

### Stage 2 — Query (`query.py`)

Finds relevant chunks and generates an answer using an LLM.

```
                           query.py
  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │   User Question: "What is RAG?"                                 │
  │       │                                                         │
  │       ▼                                                         │
  │   ┌─────────────────────────┐                                   │
  │   │  HuggingFace MiniLM     │  Same model as ingestion!        │
  │   │  embed_query()          │  Converts question into a        │
  │   │                         │  384-dimension vector             │
  │   └────────┬────────────────┘                                   │
  │            │  Query vector: [0.023, -0.087, 0.153, ...]        │
  │            ▼                                                    │
  │   ┌─────────────────────────┐                                   │
  │   │  Chroma DB              │  Cosine similarity search:       │
  │   │  Similarity Search      │  compares query vector against   │
  │   │  (k=3)                  │  ALL stored chunk vectors        │
  │   └────────┬────────────────┘                                   │
  │            │  Top 3 most similar chunks                        │
  │            ▼                                                    │
  │   ┌─────────────────────────┐                                   │
  │   │  ChatPromptTemplate     │  Stuffs all 3 chunks into one    │
  │   │  (Stuff Strategy)       │  prompt with system instructions │
  │   │                         │                                   │
  │   │  "Answer using ONLY     │                                   │
  │   │   this context:         │                                   │
  │   │   [Chunk1][Chunk2]...   │                                   │
  │   │   Question: What is...?"│                                   │
  │   └────────┬────────────────┘                                   │
  │            │  Complete prompt                                   │
  │            ▼                                                    │
  │   ┌─────────────────────────┐                                   │
  │   │  Groq LLM               │  ← Only API call (cloud)        │
  │   │  (llama-3.1-8b-instant) │  Generates answer grounded      │
  │   │  temperature=0.2        │  in retrieved context            │
  │   └────────┬────────────────┘                                   │
  │            │                                                    │
  │            ▼                                                    │
  │   Answer: "RAG stands for Retrieval-Augmented Generation..."    │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘
```

### Vector Embedding — What Actually Happens

When MiniLM processes text, it converts human language into a point in 384-dimensional space.
Similar meanings land near each other; unrelated meanings land far apart.

```
  Text Chunk                          384-Dimension Vector
  ─────────────────────────────────────────────────────────────
  "RAG combines retrieval             [0.023, -0.087, 0.153,
   with generation"          ──►       0.041, -0.012, 0.098,
                                       ...382 more numbers...]

  "Retrieval-augmented                [0.025, -0.082, 0.149,    ← very close
   generation merges search            0.039, -0.015, 0.101,      (similar meaning)
   with LLM output"         ──►       ...382 more numbers...]

  "The weather in Paris              [-0.134, 0.067, -0.045,    ← far away
   is sunny today"           ──►       0.112, 0.088, -0.023,      (unrelated meaning)
                                       ...382 more numbers...]
```

Chroma stores these vectors and uses **cosine similarity** to find the closest matches:

```
  Query vector ──────────────────────► ● (your question)
                                      ╱│╲
                   distance = 0.12  ╱  │  ╲  distance = 0.89
                                  ╱    │    ╲
                                ●      │      ●
                            Chunk A    │    Chunk C
                          (relevant)   │   (irrelevant)
                                       │
                                       ● Chunk B
                                    distance = 0.15
                                    (relevant)

  Returns: [Chunk A, Chunk B]  ← closest vectors = most relevant text
```

---

## Project Layout

```
BasicRAG/
├── ingest.py             # Parse → chunk → embed → store vectors
├── query.py              # Interactive Q&A over stored vectors
├── .env                  # Your Groq API key (git-ignored)
├── .env.example          # Template — copy to .env
├── .gitignore            # Excludes venv, secrets, generated data
├── requirements.txt      # Python dependencies
├── RAG_for_LLM.pdf       # Source document (your PDF)
├── chroma_db/            # Generated: persistent vector store
└── index_cache_base/     # Generated: SQLite record manager ledger
```

| File | Purpose |
|------|---------|
| `ingest.py` | Parses PDF → splits into chunks → embeds locally → syncs vectors to Chroma on disk |
| `query.py` | Embeds user question → retrieves top-3 similar chunks → sends to Groq LLM → returns answer |
| `.env` | Stores `GROQ_API_KEY` (never committed to Git) |
| `requirements.txt` | Pinned dependencies for reproducible installs |

---

## Setup

### Prerequisites

- Python 3.11+
- A free [Groq API key](https://console.groq.com/keys)

### Installation (Windows / PowerShell)

```powershell
cd E:\Prince\Python\RAG\BasicRAG
python -m venv BasicRAG_Env
.\BasicRAG_Env\Scripts\Activate.ps1
pip install -r requirements.txt

copy .env.example .env      # then paste your Groq key into .env
```

> **First run downloads ~90 MB** for the `all-MiniLM-L6-v2` embedding model (cached at `~/.cache/huggingface`).

---

## Usage

### Step 1 — Ingest your PDF

```powershell
python ingest.py
```

Output:
```
Initializing local embedding engine...
Loading document: RAG_for_LLM.pdf
Syncing documents with local Chroma DB storage...

--- Ingestion Sync Report ---
Num Added:   42
Num Updated: 0
Num Skipped: 0
Num Deleted: 0
Vector Store update complete and saved safely to disk.
```

### Step 2 — Ask questions

```powershell
python query.py
```

```
Vector Search Engine Online. Type 'exit' to quit.

Ask your PDF a question: What is retrieval-augmented generation?
RAG (Retrieval-Augmented Generation) is a technique that combines...

Ask your PDF a question: exit
```

### Re-ingestion

`ingest.py` is **idempotent** thanks to the SQLRecordManager:

| Situation | What happens |
|-----------|--------------|
| PDF unchanged | Every chunk **skipped** — no re-embedding |
| PDF edited | Only changed chunks **updated**; stale ones **deleted** |
| PDF replaced | Old vectors **purged**, new ones **added** |

---

## Architecture Deep Dive

### End-to-End Data Flow

```
  ┌───────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐
  │           │     │            │     │            │     │            │
  │    PDF    │────►│  PyMuPDF   │────►│  Splitter  │────►│  MiniLM    │
  │           │     │  4LLM      │     │  1000/200  │     │  Embedder  │
  │           │     │            │     │            │     │  (384-dim) │
  └───────────┘     └────────────┘     └────────────┘     └─────┬──────┘
                                                                │
                 YOUR MACHINE (free, offline)                    │
  ──────────────────────────────────────────────────────────────│──────
                                                                │
                                                                ▼
                                                          ┌────────────┐
                     ┌──────────────────────────────────── │  Chroma DB │
                     │                                     │  (on disk) │
                     │                                     └─────┬──────┘
                     │                                           │
                     │  cosine similarity search                 │
                     │                                           │
  ┌───────────┐     ┌┴───────────┐     ┌────────────┐     ┌─────┴──────┐
  │           │     │            │     │            │     │            │
  │  Answer   │◄────│  Groq LLM  │◄────│  Prompt    │◄────│  Top 3     │
  │           │     │  (cloud)   │     │  Template  │     │  Chunks    │
  │           │     │            │     │            │     │            │
  └───────────┘     └────────────┘     └────────────┘     └────────────┘

                     ▲ only API call
                     │ (the only cost)
                GROQ CLOUD
```

### The LangChain Chain Pattern

```
  create_retrieval_chain(retriever, question_answer_chain)

  ┌─────────────────── rag_chain (outer) ──────────────────────────┐
  │                                                                 │
  │  {"input": "What is RAG?"}                                      │
  │       │                                                         │
  │       ├──────────────────────────────┐                          │
  │       ▼                              │                          │
  │   retriever                          │                          │
  │   (MiniLM + Chroma, k=3)            │                          │
  │       │                              │                          │
  │       ▼                              ▼                          │
  │   ┌─────── question_answer_chain (inner) ─────────────────┐    │
  │   │                                                        │    │
  │   │  {"input": "What is RAG?", "context": [3 chunks]}     │    │
  │   │       │                                                │    │
  │   │       ▼                                                │    │
  │   │   Stuff all chunks into one prompt                     │    │
  │   │       │                                                │    │
  │   │       ▼                                                │    │
  │   │   Send to Groq LLM → get answer                       │    │
  │   │                                                        │    │
  │   └────────────────────────────────────────────────────────┘    │
  │                                                                 │
  │  Returns: {"input": ..., "context": [...], "answer": "..."}     │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘
```

---

## Configuration

### Environment Variables (`.env`)

| Variable | Required | Example | Purpose |
|----------|----------|---------|---------|
| `GROQ_API_KEY` | Yes | `gsk_abc123...` | Groq API authentication |

### Constants (in code)

| Setting | Value | File | Purpose |
|---------|-------|------|---------|
| `CHUNK_SIZE` | `1000` | `ingest.py` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | `ingest.py` | Overlap between consecutive chunks |
| `k` | `3` | `query.py` | Number of chunks retrieved per query |
| `model` | `llama-3.1-8b-instant` | `query.py` | Groq LLM model |
| `temperature` | `0.2` | `query.py` | LLM creativity (lower = more factual) |
| `collection_name` | `telecom_collection` | both | Chroma collection (must match) |
| `embedding_model` | `all-MiniLM-L6-v2` | both | Must be identical in both scripts |

---

## Key Concepts

### Why RAG?

LLMs are trained on public data up to a cutoff date. They cannot answer questions about **your** documents.
RAG solves this by **retrieving** relevant excerpts from your document and feeding them to the LLM as context,
so the answer is grounded in your data rather than hallucinated.

### Why local embeddings?

The embedding step (converting text → vectors) can run either in the cloud (OpenAI, Cohere) or locally.
This project uses local MiniLM embeddings, which means:
- **Zero cost** — no API calls for embedding
- **Zero data leakage** — your PDF never leaves your machine
- **Fast** — MiniLM is lightweight; runs on CPU in milliseconds

The **only** network call is the final LLM generation via Groq.

### Why Chroma?

Chroma is a lightweight, file-based vector database. It stores vectors as files in `./chroma_db/`,
requires no server process, and survives restarts. For a single-document POC, it is the simplest option
with zero infrastructure.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: langchain_pymupdf4llm` | `pip install langchain-pymupdf4llm` |
| `ModuleNotFoundError: langchain.chains` | `pip install langchain-classic` and update imports |
| `model_decommissioned` from Groq | Change model to `llama-3.1-8b-instant` in `query.py` |
| `GROQ_API_KEY not set` | Create `.env` from `.env.example` with your key |
| Empty or wrong answers | Ensure `collection_name` and `embedding_model` match in both scripts |
| `UnicodeDecodeError` with pipreqs | `pipreqs . --force --ignore BasicRAG_Env,chroma_db` |

---

## Tech Stack

| Component | Technology | Runs |
|-----------|-----------|------|
| PDF Parsing | PyMuPDF4LLM | Local |
| Text Splitting | LangChain RecursiveCharacterTextSplitter | Local |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (384-dim) | Local CPU |
| Vector Store | ChromaDB (file-based) | Local disk |
| Index Tracking | SQLRecordManager (SQLite) | Local disk |
| LLM | Groq `llama-3.1-8b-instant` | Cloud API |
| Orchestration | LangChain Retrieval Chain | Local |

---

## License

This project is for educational and personal use.