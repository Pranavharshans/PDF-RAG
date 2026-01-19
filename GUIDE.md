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
git clone https://github.com/YOUR_USERNAME/Pdf-rag.git
cd Pdf-rag
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

1. Go to [Pinecone Console](https://app.pinecone.io/)
2. Sign up or log in
3. Click **"Create Index"**
4. Configure the index with these **exact settings**:

| Setting | Value |
|---------|-------|
| **Index Name** | `college-pdf-rag` (or any name you prefer) |
| **Dimensions** | `1536` |
| **Metric** | `cosine` |
| **Index Type** | `Serverless` |
| **Cloud Provider** | AWS (or your preference) |
| **Region** | `us-east-1` (or your preference) |

5. Click **"Create Index"**
6. Wait for the index status to become **"Ready"**

> **Important:** The dimensions MUST be `1536` to match the OpenAI embedding model.

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
3. **First run only:** Automatically index all PDFs (takes 1-2 minutes)
4. Display the chat interface

---

## Usage

1. Type your question in the chat input
2. Press Enter or click Send
3. The chatbot will:
   - Search the indexed PDFs for relevant information
   - Generate an answer using only the PDF content
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

### Empty answers or "couldn't find information"
- Ensure PDFs are in `data/pdfs/` folder
- Check that PDFs contain extractable text (not scanned images)
- Try rephrasing your question

### Re-indexing PDFs
If you add new PDFs and want to re-index:
```bash
python indexer.py --force
```

---

## Project Structure

```
Pdf-rag/
├── app.py                 # Main Streamlit application
├── indexer.py             # PDF indexing logic
├── data/
│   └── pdfs/              # Your PDF files go here
├── utils/
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

## Support

If you encounter issues:
1. Check the Troubleshooting section above
2. Ensure all API keys are valid and have sufficient credits
3. Verify Pinecone index settings match the requirements
