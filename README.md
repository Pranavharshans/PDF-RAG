# College Department PDF Chatbot (RAG MVP)

A Streamlit-based RAG chatbot that answers questions strictly from a fixed set of PDFs using Pinecone for retrieval and Groq for fast generation.

## Features

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
| Vector DB | Pinecone (hosted) |
| LLM | Groq API |
| PDF Extraction | PyPDF2 |

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy the example environment file and fill in your API keys:

```bash
cp .env.example .env
```

Required keys:
- `OPENROUTER_API_KEY` - Get from [OpenRouter](https://openrouter.ai/)
- `GROQ_API_KEY` - Get from [Groq Console](https://console.groq.com/)
- `PINECONE_API_KEY` - Get from [Pinecone Console](https://app.pinecone.io/)
- `PINECONE_INDEX_NAME` - Name for your Pinecone index

### 3. Create Pinecone Index

Create a serverless index in Pinecone with:
- **Dimensions**: 1536
- **Metric**: cosine

### 4. Add Your PDFs

Place your PDF files in the `data/pdfs/` directory.

### 5. Run the Application

```bash
streamlit run app.py
```

The application will automatically index your PDFs on first run if the Pinecone index is empty.

## Project Structure

```
project/
├── app.py                    # Main Streamlit application
├── indexer.py                # One-time automatic indexing logic
├── data/
│   └── pdfs/                 # Place PDF files here
├── utils/
│   ├── __init__.py
│   ├── chunking.py           # Text splitting logic
│   ├── embeddings.py         # OpenRouter embedding calls
│   ├── pinecone_utils.py     # Pinecone operations
│   └── pdf_loader.py         # PDF text extraction
├── requirements.txt
├── .env.example
└── README.md
```

## Usage

1. Open the Streamlit app in your browser
2. Type your question about the college department
3. The system will:
   - Embed your question
   - Search for relevant content in the PDFs
   - Generate an answer using only the retrieved content
   - Display the answer with source citations

## Safety Features

- Answers are generated **only** from indexed PDF content
- If information is not found, the system responds with a fallback message
- All answers include source citations (PDF name + page number)
- No external knowledge or speculation is used

## License

MIT
