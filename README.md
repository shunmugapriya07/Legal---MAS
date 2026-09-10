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
5. Test cases: What happened to Facebook’s stock price after the Cambridge Analytica allegations became public?
Why did the Ninth Circuit reject Facebook’s argument that the earlier Guardian report had already made the Cambridge Analytica data misuse public?
Which statements concerning Facebook users’ control over their data were challenged, and how did the Ninth Circuit rule on those claims?

## Approach Followed

The proposed system uses a multi-agent Retrieval-Augmented Generation (RAG) architecture for answering legal questions from a collection of legal documents. The workflow consists of three main agents: Query Processor Agent, Researcher Agent, and Synthesizer Agent.

1. Query Processor Agent

The Query Processor Agent receives the user's legal question and analyzes it to identify the main legal concepts, entities, and information required to answer the query. It reformulates the question into suitable search terms so that relevant information can be retrieved from the legal documents.

2. Researcher Agent

The Researcher Agent performs retrieval over the processed query. The legal documents are first divided using a hierarchical chunking strategy (http://www.lrec-conf.org/proceedings/lrec2026/pdf/2026.lrec2026-1.903.pdf) , where documents are organized into larger sections and then smaller meaningful chunks. The chunks are converted into embeddings and stored in a vector database.

For a given query, the agent retrieves the most relevant document chunks based on semantic similarity. These retrieved chunks provide the legal context required for answering the question.

3. Synthesizer Agent

The Synthesizer Agent receives the original legal question together with the retrieved document context. It analyzes the retrieved information and generates the final answer.

The agent is instructed to base the answer on the retrieved legal documents and provide the relevant source references. This helps the system produce an answer that is grounded in the available legal evidence rather than relying only on the language model's general knowledge.
