# Complete Setup Guide

A step-by-step guide to run the College Department PDF Chatbot from scratch.

---

## Prerequisites

- Python 3.10 or higher
- Git
- API keys for:
  - [OpenRouter](https://openrouter.ai/) (for embeddings)
  - [Groq](https://console.groq.com/) (for LLM)
  - [Pinecone](https://app.pinecone.io/) (for vector database)

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/Pranavharshans/PDF-RAG.git
cd PDF-RAG
```

---

## Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

> **Windows users:** Use `venv\Scripts\activate` instead.

---

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 4: Create Pinecone Index

> ⚠️ **IMPORTANT:** This app uses **Hybrid Search** which requires specific settings!

1. Go to [Pinecone Console](https://app.pinecone.io/)
2. Sign up or log in
3. Click **"Create Index"**
4. Configure the index with these **EXACT settings**:

| Setting | Value | Why? |
|---------|-------|------|
| **Index Name** | `college-pdf-rag` (or any name) | Must match `.env` |
| **Dimensions** | `1536` | Matches OpenAI embedding model |
| **Metric** | ⚠️ **`dotproduct`** | **REQUIRED for hybrid search** |
| **Index Type** | `Serverless` | Recommended |
| **Cloud Provider** | AWS | Or your preference |
| **Region** | `us-east-1` | Or your preference |

5. Click **"Create Index"**
6. Wait for the index status to become **"Ready"**

> ⚠️ **Critical:** The metric MUST be `dotproduct` (not `cosine`). Hybrid search with sparse vectors requires dotproduct metric!

---

## Step 5: Get Your API Keys

### OpenRouter API Key
1. Go to [OpenRouter Keys](https://openrouter.ai/keys)
2. Sign up or log in
3. Click **"Create Key"**
4. Copy the key (starts with `sk-or-...`)

### Groq API Key
1. Go to [Groq Console](https://console.groq.com/keys)
2. Sign up or log in
3. Click **"Create API Key"**
4. Copy the key (starts with `gsk_...`)

### Pinecone API Key
1. Go to [Pinecone Console](https://app.pinecone.io/)
2. Click on **"API Keys"** in the left sidebar
3. Copy your API key

---

## Step 6: Create Environment File

```bash
cp .env.example .env
```

Open the `.env` file in a text editor and fill in your API keys:

```env
# OpenRouter API Key (for embeddings)
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxx

# Groq API Key (for LLM generation)
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx

# Pinecone Configuration
PINECONE_API_KEY=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
PINECONE_INDEX_NAME=college-pdf-rag
```

> **Important:** 
> - No quotes around values
> - No spaces around `=`
> - `PINECONE_INDEX_NAME` must match the index name you created in Step 4

---

## Step 7: Add Your PDF Files

Place your PDF files in the `data/pdfs/` folder:

```bash
ls data/pdfs/
```

The repository comes with sample PDFs. You can replace them with your own.

---

## Step 8: Run the Application

```bash
streamlit run app.py
```

The app will:
1. Open in your browser at `http://localhost:8501`
2. Check if the Pinecone index is empty
3. **First run only:** Automatically index all PDFs with hybrid search (takes 1-2 minutes)
4. Display the chat interface with streaming responses

---

## Features

### Hybrid Search
The app combines two search methods for better results:
- **Semantic Search**: Understands meaning and context
- **Keyword Search (BM25)**: Matches exact terms and names

This means queries like "Dr. Anandakumar" or "CSE PAC 2024" will find exact matches!

### Streaming Responses
Answers appear word-by-word as they're generated, providing a faster, more responsive experience.

---

## Usage

1. Type your question in the chat input
2. Press Enter or click Send
3. The chatbot will:
   - Search using hybrid search (semantic + keyword)
   - Stream the answer in real-time
   - Show sources (PDF name + page number) for every answer

---

## Troubleshooting

### "PINECONE_API_KEY environment variable is not set"
- Make sure `.env` file exists in the project root
- Check that there are no typos in variable names
- Ensure no extra spaces or quotes around values

### "Index not found" error
- Verify the `PINECONE_INDEX_NAME` in `.env` matches your Pinecone index exactly
- Check that the index status is "Ready" in Pinecone console

### "Dimension mismatch" error
- Delete and recreate your Pinecone index with **1536 dimensions**

### "Invalid sparse vector" or hybrid search errors
- Make sure your Pinecone index uses **`dotproduct`** metric (not `cosine`)
- Delete and recreate the index with the correct metric

### Empty answers or "couldn't find information"
- Ensure PDFs are in `data/pdfs/` folder
- Check that PDFs contain extractable text (not scanned images)
- Try rephrasing your question

### Re-indexing PDFs
If you add new PDFs and want to re-index:
```bash
python indexer.py --force
```

### Migrating from old index (cosine metric)
If you previously created an index with `cosine` metric:
1. Delete the old index in Pinecone Console
2. Create a new index with `dotproduct` metric
3. Run `python indexer.py --force` to re-index

---

## Project Structure

```
PDF-RAG/
├── app.py                 # Main Streamlit application
├── indexer.py             # PDF indexing logic (hybrid)
├── data/
│   ├── pdfs/              # Your PDF files go here
│   └── bm25_model.json    # Fitted BM25 model (auto-generated)
├── utils/
│   ├── bm25_encoder.py    # BM25 sparse encoder for hybrid search
│   ├── chunking.py        # Text splitting
│   ├── embeddings.py      # OpenRouter embeddings
│   ├── pdf_loader.py      # PDF text extraction
│   └── pinecone_utils.py  # Vector database operations
├── requirements.txt       # Python dependencies
├── .env.example           # Environment template
├── .env                   # Your API keys (create this)
└── README.md
```

---

## Commands Reference

| Command | Description |
|---------|-------------|
| `source venv/bin/activate` | Activate virtual environment |
| `streamlit run app.py` | Start the chatbot |
| `python indexer.py --status` | Check index status |
| `python indexer.py --force` | Force re-index all PDFs |
| `deactivate` | Exit virtual environment |

---

## Technical Details

### Search Architecture
```
User Query
    ↓
┌───────────────────────────────────────┐
│  Dense Embedding (OpenRouter)         │ → Semantic understanding
│  Sparse Embedding (BM25)              │ → Keyword matching
└───────────────────────────────────────┘
    ↓
Pinecone Hybrid Query (dotproduct fusion)
    ↓
Top 6 most relevant chunks
    ↓
Groq LLM (streaming response)
    ↓
Answer + Sources
```

### Why dotproduct metric?
Pinecone's hybrid search combines dense and sparse vectors using the dotproduct metric. Cosine similarity doesn't support sparse vectors properly, which is why `dotproduct` is required.

---

## Support

If you encounter issues:
1. Check the Troubleshooting section above
2. Ensure all API keys are valid and have sufficient credits
3. Verify Pinecone index settings match the requirements (especially `dotproduct` metric!)
