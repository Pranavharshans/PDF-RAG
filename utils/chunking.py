"""
Chunking Module
Splits text into chunks with overlap for RAG retrieval.
Uses tiktoken for accurate token counting.
"""

import uuid
from dataclasses import dataclass

import tiktoken

from utils.pdf_loader import PageContent, PDFDocument


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata for indexing."""
    id: str
    text: str
    source_pdf: str
    page: int
    token_count: int


# Target chunk size in tokens (600-900 range, targeting 800)
CHUNK_SIZE = 800
# Overlap between chunks for context continuity
CHUNK_OVERLAP = 100
# Encoding for token counting (cl100k_base is used by text-embedding-3-small)
ENCODING_NAME = "cl100k_base"


def get_tokenizer() -> tiktoken.Encoding:
    """Get the tiktoken encoding for token counting."""
    return tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text: str, tokenizer: tiktoken.Encoding | None = None) -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text: The text to count tokens for
        tokenizer: Optional pre-initialized tokenizer
        
    Returns:
        Number of tokens in the text
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()
    return len(tokenizer.encode(text))


def split_text_into_chunks(
    text: str,
    source_pdf: str,
    page: int,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[TextChunk]:
    """
    Split text into overlapping chunks based on token count.
    
    Args:
        text: The text content to split
        source_pdf: Name of the source PDF file
        page: Page number in the PDF
        chunk_size: Target size of each chunk in tokens
        chunk_overlap: Number of overlapping tokens between chunks
        
    Returns:
        List of TextChunk objects
    """
    if not text.strip():
        return []
    
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text)
    total_tokens = len(tokens)
    
    # If text fits in one chunk, return it as-is
    if total_tokens <= chunk_size:
        return [TextChunk(
            id=generate_chunk_id(source_pdf, page, 0),
            text=text.strip(),
            source_pdf=source_pdf,
            page=page,
            token_count=total_tokens
        )]
    
    chunks: list[TextChunk] = []
    start_idx = 0
    chunk_index = 0
    
    while start_idx < total_tokens:
        # Calculate end index for this chunk
        end_idx = min(start_idx + chunk_size, total_tokens)
        
        # Extract chunk tokens and decode back to text
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = tokenizer.decode(chunk_tokens).strip()
        
        if chunk_text:
            chunks.append(TextChunk(
                id=generate_chunk_id(source_pdf, page, chunk_index),
                text=chunk_text,
                source_pdf=source_pdf,
                page=page,
                token_count=len(chunk_tokens)
            ))
            chunk_index += 1
        
        # Move start index forward, accounting for overlap
        start_idx = end_idx - chunk_overlap
        
        # Prevent infinite loop if overlap is too large
        if start_idx >= end_idx:
            break
    
    return chunks


def generate_chunk_id(source_pdf: str, page: int, chunk_index: int) -> str:
    """
    Generate a unique ID for a chunk.
    
    Args:
        source_pdf: Name of the source PDF
        page: Page number
        chunk_index: Index of chunk within the page
        
    Returns:
        Unique chunk ID string
    """
    # Create a deterministic but unique ID
    base = f"{source_pdf}_{page}_{chunk_index}"
    # Add a short UUID suffix for extra uniqueness
    suffix = uuid.uuid5(uuid.NAMESPACE_DNS, base).hex[:8]
    return f"{source_pdf.replace('.pdf', '')}__p{page}__c{chunk_index}__{suffix}"


def chunk_page(page: PageContent) -> list[TextChunk]:
    """
    Chunk a single page's content.
    
    Args:
        page: PageContent object containing text and metadata
        
    Returns:
        List of TextChunk objects
    """
    return split_text_into_chunks(
        text=page.text,
        source_pdf=page.filename,
        page=page.page_number
    )


def chunk_document(document: PDFDocument) -> list[TextChunk]:
    """
    Chunk all pages of a PDF document.
    
    Args:
        document: PDFDocument object
        
    Returns:
        List of all TextChunk objects from the document
    """
    chunks: list[TextChunk] = []
    for page in document.pages:
        page_chunks = chunk_page(page)
        chunks.extend(page_chunks)
    return chunks


def chunk_documents(documents: list[PDFDocument]) -> list[TextChunk]:
    """
    Chunk all documents.
    
    Args:
        documents: List of PDFDocument objects
        
    Returns:
        List of all TextChunk objects from all documents
    """
    all_chunks: list[TextChunk] = []
    
    for doc in documents:
        doc_chunks = chunk_document(doc)
        all_chunks.extend(doc_chunks)
        print(f"Chunked {doc.filename}: {len(doc_chunks)} chunks")
    
    total_tokens = sum(chunk.token_count for chunk in all_chunks)
    print(f"Total: {len(all_chunks)} chunks, {total_tokens:,} tokens")
    
    return all_chunks
