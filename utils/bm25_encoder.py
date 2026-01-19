"""
BM25 Sparse Encoder for Hybrid Search
Provides keyword-based sparse vectors to complement semantic dense vectors.
"""

from pathlib import Path

from pinecone_text.sparse import BM25Encoder


# Path to save fitted BM25 model
BM25_MODEL_PATH = Path(__file__).parent.parent / "data" / "bm25_model.json"

# Global encoder instance (lazy loaded)
_bm25_encoder: BM25Encoder | None = None


def get_bm25_encoder() -> BM25Encoder:
    """
    Get or load the BM25 encoder.
    
    Returns:
        Fitted BM25Encoder instance
        
    Raises:
        FileNotFoundError: If model hasn't been fitted yet
    """
    global _bm25_encoder
    
    if _bm25_encoder is None:
        if BM25_MODEL_PATH.exists():
            _bm25_encoder = BM25Encoder().load(str(BM25_MODEL_PATH))
        else:
            raise FileNotFoundError(
                f"BM25 model not found at {BM25_MODEL_PATH}. "
                "Run indexing first to fit the model."
            )
    
    return _bm25_encoder


def fit_bm25_encoder(texts: list[str]) -> BM25Encoder:
    """
    Fit BM25 encoder on corpus and save to disk.
    
    Args:
        texts: List of document texts to fit on
        
    Returns:
        Fitted BM25Encoder instance
    """
    global _bm25_encoder
    
    print(f"Fitting BM25 encoder on {len(texts)} documents...")
    _bm25_encoder = BM25Encoder()
    _bm25_encoder.fit(texts)
    
    # Ensure directory exists
    BM25_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    _bm25_encoder.dump(str(BM25_MODEL_PATH))
    print(f"BM25 model saved to {BM25_MODEL_PATH}")
    
    return _bm25_encoder


def encode_document(text: str) -> dict:
    """
    Encode a document for sparse vector storage.
    
    Args:
        text: Document text to encode
        
    Returns:
        Dict with 'indices' and 'values' for sparse vector
    """
    encoder = get_bm25_encoder()
    return encoder.encode_documents([text])[0]


def encode_documents_batch(texts: list[str]) -> list[dict]:
    """
    Encode multiple documents for sparse vector storage.
    
    Args:
        texts: List of document texts to encode
        
    Returns:
        List of dicts with 'indices' and 'values' for sparse vectors
    """
    encoder = get_bm25_encoder()
    return encoder.encode_documents(texts)


def encode_query(query: str) -> dict:
    """
    Encode a query for sparse vector search.
    
    Args:
        query: Search query text
        
    Returns:
        Dict with 'indices' and 'values' for sparse vector
    """
    encoder = get_bm25_encoder()
    return encoder.encode_queries([query])[0]
