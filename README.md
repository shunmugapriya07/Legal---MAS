# Legal RAG Pipeline

A Retrieval-Augmented Generation (RAG) based Multi Agent System (MAS) orchestration pipeline tailored for dense legal PDFs. Built on LangGraph, LlamaIndex, and FAISS.

## Architecture

- **Embedding:** `mxbai-embed-large-v1` (runs locally)
- **Vector DB:** FAISS
- **Retrieval:** Hybrid (BM25 + Dense FAISS) + Auto-Merging
- **Reranker:** `ms-marco-MiniLM-L-6-v2` (runs locally)
- **Orchestration:** LangGraph (Query Processor -> Researcher -> Synthesizer)
- **LLMs:** GPT-4o (Query Optimization), Claude Sonnet 4.6 (Synthesis) via OpenRouter

## Features

- **Semantic Cache:** Intercepts incoming questions, calculates mathematical distance against past queries, and returns cached answers instantly if similar.
- **Persistent Storage:** Embeddings and hierarchical text chunks are aggressively cached to `./storage`, preventing massive boot delays.
- **Strict Citations:** The Synthesizer is hard-prompted to cite exact source filenames and page numbers for every claim.

## Setup

1. Create a virtual environment: `python3 -m venv venv && source venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt --no-cache-dir`
3. Ensure your OpenRouter API key is set in `keys.env`.
4. Run it: `python main.py`
5. Test case: What happened to Facebook’s stock price after the Cambridge Analytica allegations became public?
6. Why did the Ninth Circuit reject Facebook’s argument that the earlier Guardian report had already made the Cambridge Analytica data misuse public?
7. Which statements concerning Facebook users’ control over their data were challenged, and how did the Ninth Circuit rule on those claims?