"""
Embeddings Module
Handles embedding generation using OpenRouter API with OpenAI's text-embedding-3-small model.
"""

import os
from typing import Sequence

from openai import OpenAI

from utils.chunking import TextChunk


# OpenRouter configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = "openai/text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# Batch size for embedding generation (OpenRouter limit)
BATCH_SIZE = 100


def get_embedding_client() -> OpenAI:
    """
    Create an OpenAI client configured for OpenRouter.
    
    Returns:
        OpenAI client instance
        
    Raises:
        ValueError: If OPENROUTER_API_KEY is not set
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set")
    
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
    )


def generate_embedding(text: str, client: OpenAI | None = None) -> list[float]:
    """
    Generate embedding for a single text string.
    
    Args:
        text: The text to embed
        client: Optional pre-initialized OpenAI client
        
    Returns:
        List of floats representing the embedding vector
    """
    if client is None:
        client = get_embedding_client()
    
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    
    return response.data[0].embedding


def generate_embeddings_batch(
    texts: Sequence[str],
    client: OpenAI | None = None,
    show_progress: bool = True
) -> list[list[float]]:
    """
    Generate embeddings for multiple texts in batches.
    
    Args:
        texts: Sequence of texts to embed
        client: Optional pre-initialized OpenAI client
        show_progress: Whether to print progress updates
        
    Returns:
        List of embedding vectors (same order as input texts)
    """
    if client is None:
        client = get_embedding_client()
    
    all_embeddings: list[list[float]] = []
    total_texts = len(texts)
    
    for i in range(0, total_texts, BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        
        if show_progress:
            batch_end = min(i + BATCH_SIZE, total_texts)
            print(f"Generating embeddings: {batch_end}/{total_texts}")
        
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=list(batch),
        )
        
        # Sort by index to maintain order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        batch_embeddings = [item.embedding for item in sorted_data]
        all_embeddings.extend(batch_embeddings)
    
    return all_embeddings


def generate_chunk_embeddings(
    chunks: list[TextChunk],
    client: OpenAI | None = None
) -> list[tuple[TextChunk, list[float]]]:
    """
    Generate embeddings for a list of text chunks.
    
    Args:
        chunks: List of TextChunk objects
        client: Optional pre-initialized OpenAI client
        
    Returns:
        List of tuples containing (chunk, embedding)
    """
    if not chunks:
        return []
    
    if client is None:
        client = get_embedding_client()
    
    texts = [chunk.text for chunk in chunks]
    embeddings = generate_embeddings_batch(texts, client)
    
    return list(zip(chunks, embeddings))


def generate_query_embedding(query: str) -> list[float]:
    """
    Generate embedding for a user query.
    
    Args:
        query: The user's question
        
    Returns:
        Embedding vector for the query
    """
    return generate_embedding(query)
