"""
PDF Loader Module
Extracts text from PDF files with page number tracking for source attribution.
Uses pdfplumber for better handling of tables and complex layouts.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import pdfplumber


@dataclass
class PageContent:
    """Represents extracted content from a single PDF page."""
    filename: str
    page_number: int
    text: str


@dataclass
class PDFDocument:
    """Represents a complete PDF document with all its pages."""
    filename: str
    pages: list[PageContent]


def extract_text_from_pdf(pdf_path: str | Path) -> PDFDocument:
    """
    Extract text from a PDF file with page-level tracking.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        PDFDocument containing all pages with their text content
    """
    pdf_path = Path(pdf_path)
    filename = pdf_path.name
    
    pages: list[PageContent] = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            
            if text:
                pages.append(PageContent(
                    filename=filename,
                    page_number=page_num,
                    text=text
                ))
    
    return PDFDocument(filename=filename, pages=pages)


def load_all_pdfs(pdf_directory: str | Path) -> list[PDFDocument]:
    """
    Load all PDF files from a directory.
    
    Args:
        pdf_directory: Path to directory containing PDF files
        
    Returns:
        List of PDFDocument objects
    """
    pdf_directory = Path(pdf_directory)
    
    if not pdf_directory.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_directory}")
    
    documents: list[PDFDocument] = []
    pdf_files = sorted(pdf_directory.glob("*.pdf"))
    
    if not pdf_files:
        print(f"Warning: No PDF files found in {pdf_directory}")
        return documents
    
    for pdf_path in pdf_files:
        try:
            doc = extract_text_from_pdf(pdf_path)
            documents.append(doc)
            print(f"Loaded: {doc.filename} ({len(doc.pages)} pages)")
        except Exception as e:
            print(f"Error loading {pdf_path.name}: {e}")
            continue
    
    return documents


def iter_all_pages(documents: list[PDFDocument]) -> Generator[PageContent, None, None]:
    """
    Iterate over all pages from all documents.
    
    Args:
        documents: List of PDFDocument objects
        
    Yields:
        PageContent objects from all documents
    """
    for doc in documents:
        yield from doc.pages


def get_total_pages(documents: list[PDFDocument]) -> int:
    """Get total number of pages across all documents."""
    return sum(len(doc.pages) for doc in documents)
