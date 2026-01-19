# College Department PDF Chatbot (RAG MVP)

A Streamlit-based RAG chatbot that answers questions strictly from a fixed set of PDFs using Pinecone for retrieval and Groq for fast generation.

## Features

- **Hybrid Search**: Combines semantic understanding with keyword matching for better retrieval
- **Streaming Responses**: See answers appear in real-time
- **PDF-only answers**: Responses are generated exclusively from your uploaded PDFs
- **Source citations**: Every answer includes PDF filename and page number
- **Auto-indexing**: PDFs are automatically indexed on first run
- **Fast inference**: Uses Groq for rapid LLM responses
- **No hallucinations**: Strict prompt engineering ensures factual answers

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Embeddings | OpenRouter (text-embedding-3-small) |
| Keyword Search | BM25 (pinecone-text) |
| Vector DB | Pinecone (hosted) |
| LLM | Groq API |
| PDF Extraction | pdfplumber |

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/Pranavharshans/PDF-RAG.git
cd PDF-RAG
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your API keys:
- `OPENROUTER_API_KEY` - Get from [OpenRouter](https://openrouter.ai/)
- `GROQ_API_KEY` - Get from [Groq Console](https://console.groq.com/)
- `PINECONE_API_KEY` - Get from [Pinecone Console](https://app.pinecone.io/)
- `PINECONE_INDEX_NAME` - Name for your Pinecone index

### 3. Create Pinecone Index

Create a serverless index in Pinecone with these **exact settings**:

| Setting | Value |
|---------|-------|
| **Dimensions** | `1536` |
| **Metric** | ⚠️ `dotproduct` (required for hybrid search) |

> **Important**: Must use `dotproduct` metric, not `cosine`!

### 4. Add Your PDFs

Place your PDF files in the `data/pdfs/` directory.

### 5. Run the Application

```bash
streamlit run app.py
```

The application will automatically index your PDFs on first run.

## Project Structure

```
PDF-RAG/
├── app.py                    # Main Streamlit application
├── indexer.py                # Automatic indexing logic (hybrid)
├── data/
│   ├── pdfs/                 # Place PDF files here
│   └── bm25_model.json       # Fitted BM25 model (auto-generated)
├── utils/
│   ├── __init__.py
│   ├── bm25_encoder.py       # BM25 sparse encoder
│   ├── chunking.py           # Text splitting logic
│   ├── embeddings.py         # OpenRouter embedding calls
│   ├── pinecone_utils.py     # Pinecone operations (hybrid)
│   └── pdf_loader.py         # PDF text extraction
├── requirements.txt
├── .env.example
├── GUIDE.md                  # Detailed setup guide
└── README.md
```

## Usage

1. Open the Streamlit app in your browser
2. Type your question about the college department
3. The system will:
   - Search using hybrid search (semantic + keyword)
   - Stream the answer in real-time
   - Display source citations (PDF name + page number)

## Safety Features

- Answers are generated **only** from indexed PDF content
- If information is not found, the system responds with a fallback message
- All answers include source citations (PDF name + page number)
- No external knowledge or speculation is used

## Documentation

See [GUIDE.md](GUIDE.md) for detailed setup instructions and troubleshooting.

## License

MIT
