"""
Indexer Module
Handles automatic one-time indexing of PDF documents into Pinecone.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from utils.pdf_loader import load_all_pdfs, get_total_pages
from utils.chunking import chunk_documents
from utils.embeddings import generate_chunk_embeddings, get_embedding_client
from utils.pinecone_utils import (
    is_index_empty,
    upsert_chunks,
    get_index_stats,
    get_pinecone_client,
)


# Default path to PDF directory
DEFAULT_PDF_DIR = Path(__file__).parent / "data" / "pdfs"


def run_indexing(pdf_directory: str | Path | None = None, force: bool = False) -> bool:
    """
    Run the indexing pipeline to load PDFs into Pinecone.
    
    This function checks if the index is empty before indexing.
    If the index already contains vectors, indexing is skipped unless force=True.
    
    Args:
        pdf_directory: Path to directory containing PDF files. 
                      Defaults to data/pdfs/
        force: If True, index even if vectors already exist
        
    Returns:
        True if indexing was performed, False if skipped
    """
    # Load environment variables
    load_dotenv()
    
    # Set default PDF directory
    if pdf_directory is None:
        pdf_directory = DEFAULT_PDF_DIR
    pdf_directory = Path(pdf_directory)
    
    print("=" * 50)
    print("PDF RAG Indexer")
    print("=" * 50)
    
    # Initialize clients
    print("\n[1/6] Initializing clients...")
    pinecone_client = get_pinecone_client()
    embedding_client = get_embedding_client()
    
    # Check if index already has vectors
    print("\n[2/6] Checking Pinecone index status...")
    if not force and not is_index_empty(pinecone_client):
        stats = get_index_stats(pinecone_client)
        vector_count = stats.get("total_vector_count", 0)
        print(f"Index already contains {vector_count:,} vectors.")
        print("Skipping indexing. Use force=True to re-index.")
        return False
    
    # Load PDFs
    print(f"\n[3/6] Loading PDFs from {pdf_directory}...")
    documents = load_all_pdfs(pdf_directory)
    
    if not documents:
        print("No PDF documents found. Please add PDFs to the data/pdfs/ directory.")
        return False
    
    total_pages = get_total_pages(documents)
    print(f"Loaded {len(documents)} documents with {total_pages} total pages.")
    
    # Chunk documents
    print("\n[4/6] Chunking documents...")
    chunks = chunk_documents(documents)
    
    if not chunks:
        print("No chunks generated from documents.")
        return False
    
    # Generate embeddings
    print("\n[5/6] Generating embeddings...")
    chunk_embedding_pairs = generate_chunk_embeddings(chunks, embedding_client)
    
    # Separate chunks and embeddings for upsert
    indexed_chunks = [pair[0] for pair in chunk_embedding_pairs]
    embeddings = [pair[1] for pair in chunk_embedding_pairs]
    
    # Upsert to Pinecone
    print("\n[6/6] Upserting to Pinecone...")
    upserted_count = upsert_chunks(indexed_chunks, embeddings, pinecone_client)
    
    print("\n" + "=" * 50)
    print("Indexing Complete!")
    print("=" * 50)
    print(f"Documents indexed: {len(documents)}")
    print(f"Total pages: {total_pages}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Vectors upserted: {upserted_count}")
    
    return True


def check_index_status() -> dict:
    """
    Check the current status of the Pinecone index.
    
    Returns:
        Dictionary with index statistics
    """
    load_dotenv()
    
    try:
        stats = get_index_stats()
        return {
            "status": "connected",
            "total_vectors": stats.get("total_vector_count", 0),
            "index_fullness": stats.get("index_fullness", 0),
            "dimensions": stats.get("dimension", 0),
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def ensure_indexed(pdf_directory: str | Path | None = None) -> None:
    """
    Ensure PDFs are indexed. Run indexing only if index is empty.
    
    This is the main function to call from the Streamlit app.
    
    Args:
        pdf_directory: Optional path to PDF directory
    """
    load_dotenv()
    
    if is_index_empty():
        print("Index is empty. Starting automatic indexing...")
        run_indexing(pdf_directory)
    else:
        stats = get_index_stats()
        vector_count = stats.get("total_vector_count", 0)
        print(f"Index contains {vector_count:,} vectors. Skipping indexing.")


if __name__ == "__main__":
    # Run indexing when executed directly
    import argparse
    
    parser = argparse.ArgumentParser(description="Index PDF documents into Pinecone")
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default=None,
        help="Path to directory containing PDF files"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-indexing even if vectors exist"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check index status only"
    )
    
    args = parser.parse_args()
    
    if args.status:
        status = check_index_status()
        print("\nIndex Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
    else:
        run_indexing(args.pdf_dir, args.force)
