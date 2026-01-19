"""
Pinecone Utils Module
Handles all Pinecone vector database operations.
"""

import os
from dataclasses import dataclass

from pinecone import Pinecone

from utils.chunking import TextChunk
from utils.embeddings import EMBEDDING_DIMENSIONS


# Batch size for upserting vectors
UPSERT_BATCH_SIZE = 100


@dataclass
class RetrievedChunk:
    """Represents a chunk retrieved from Pinecone with similarity score."""
    id: str
    text: str
    source_pdf: str
    page: int
    score: float


def get_pinecone_client() -> Pinecone:
    """
    Create a Pinecone client.
    
    Returns:
        Pinecone client instance
        
    Raises:
        ValueError: If PINECONE_API_KEY is not set
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable is not set")
    
    return Pinecone(api_key=api_key)


def get_index_name() -> str:
    """
    Get the Pinecone index name from environment.
    
    Returns:
        Index name string
        
    Raises:
        ValueError: If PINECONE_INDEX_NAME is not set
    """
    index_name = os.getenv("PINECONE_INDEX_NAME")
    if not index_name:
        raise ValueError("PINECONE_INDEX_NAME environment variable is not set")
    return index_name


def get_index(client: Pinecone | None = None):
    """
    Get the Pinecone index.
    
    Args:
        client: Optional pre-initialized Pinecone client
        
    Returns:
        Pinecone index instance
    """
    if client is None:
        client = get_pinecone_client()
    
    index_name = get_index_name()
    return client.Index(index_name)


def is_index_empty(client: Pinecone | None = None) -> bool:
    """
    Check if the Pinecone index is empty.
    
    Args:
        client: Optional pre-initialized Pinecone client
        
    Returns:
        True if index is empty, False otherwise
    """
    index = get_index(client)
    stats = index.describe_index_stats()
    total_vectors = stats.get("total_vector_count", 0)
    return total_vectors == 0


def get_index_stats(client: Pinecone | None = None) -> dict:
    """
    Get statistics about the Pinecone index.
    
    Args:
        client: Optional pre-initialized Pinecone client
        
    Returns:
        Dictionary with index statistics
    """
    index = get_index(client)
    return index.describe_index_stats()


def upsert_chunks(
    chunks: list[TextChunk],
    embeddings: list[list[float]],
    client: Pinecone | None = None,
    show_progress: bool = True
) -> int:
    """
    Upsert chunks with their embeddings to Pinecone.
    
    Args:
        chunks: List of TextChunk objects
        embeddings: List of embedding vectors (same order as chunks)
        client: Optional pre-initialized Pinecone client
        show_progress: Whether to print progress updates
        
    Returns:
        Number of vectors upserted
    """
    if len(chunks) != len(embeddings):
        raise ValueError("Number of chunks must match number of embeddings")
    
    index = get_index(client)
    
    # Prepare vectors for upsert
    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        vectors.append({
            "id": chunk.id,
            "values": embedding,
            "metadata": {
                "text": chunk.text,
                "source_pdf": chunk.source_pdf,
                "page": chunk.page,
            }
        })
    
    # Upsert in batches
    total_upserted = 0
    for i in range(0, len(vectors), UPSERT_BATCH_SIZE):
        batch = vectors[i:i + UPSERT_BATCH_SIZE]
        index.upsert(vectors=batch)
        total_upserted += len(batch)
        
        if show_progress:
            print(f"Upserted: {total_upserted}/{len(vectors)} vectors")
    
    return total_upserted


def query_similar_chunks(
    query_embedding: list[float],
    top_k: int = 6,
    client: Pinecone | None = None
) -> list[RetrievedChunk]:
    """
    Query Pinecone for similar chunks.
    
    Args:
        query_embedding: Embedding vector for the query
        top_k: Number of results to return
        client: Optional pre-initialized Pinecone client
        
    Returns:
        List of RetrievedChunk objects sorted by similarity
    """
    index = get_index(client)
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    retrieved_chunks: list[RetrievedChunk] = []
    
    for match in results.get("matches", []):
        metadata = match.get("metadata", {})
        retrieved_chunks.append(RetrievedChunk(
            id=match["id"],
            text=metadata.get("text", ""),
            source_pdf=metadata.get("source_pdf", "unknown"),
            page=metadata.get("page", 0),
            score=match.get("score", 0.0)
        ))
    
    return retrieved_chunks


def delete_all_vectors(client: Pinecone | None = None) -> None:
    """
    Delete all vectors from the index.
    Use with caution - this clears the entire index.
    
    Args:
        client: Optional pre-initialized Pinecone client
    """
    index = get_index(client)
    index.delete(delete_all=True)
    print("All vectors deleted from index")
