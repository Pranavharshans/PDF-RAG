# CLAUDE.md — PDF-RAG

Streamlit RAG chatbot that answers questions from PDFs using Pinecone vector search and Groq LLM.

## Key files

- `app.py` — Streamlit UI: PDF upload, chat interface, source display
- `indexer.py` — PDF ingestion: extracts text, creates Pinecone vectors
- `utils/` — helper modules for text processing and retrieval

## Running

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires Pinecone and Groq API keys in environment variables.
