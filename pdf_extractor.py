"""
PDF Extraction Module for LiveKit Voice Agent

This module handles:
1. Extracting text from PDF files
2. Splitting text into chunks suitable for embedding
3. Basic text cleaning and processing
"""

import os
import logging
from typing import List, Optional

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("voice-agent.pdf")

def extract_pdf_text(pdf_path: str) -> str:
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    logger.info(f"Extracting text from PDF: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            total_pages = len(pdf.pages)
            logger.info(f"PDF has {total_pages} pages")
            
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                text += page_text + "\n\n"
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{total_pages} pages")
            
            logger.info(f"Successfully extracted {len(text)} characters from PDF")
            return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise

def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text.
    
    Args:
        text: Raw text from PDF
        
    Returns:
        Cleaned text
    """
    # Remove excessive newlines
    text = "\n".join([line for line in text.split("\n") if line.strip()])
    
    # Additional cleaning can be added here
    
    return text

def split_text(text: str, 
               chunk_size: int = 1000, 
               chunk_overlap: int = 200) -> List[str]:
    """
    Split text into chunks suitable for embedding.
    
    Args:
        text: Text to split
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    logger.info(f"Splitting text into chunks (size={chunk_size}, overlap={chunk_overlap})")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    logger.info(f"Created {len(chunks)} text chunks")
    
    return chunks

def process_pdf(pdf_path: str, 
                chunk_size: int = 1000, 
                chunk_overlap: int = 200) -> List[str]:
    """
    Process a PDF file: extract text, clean it, and split into chunks.
    
    Args:
        pdf_path: Path to the PDF file
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of text chunks ready for embedding
    """
    # Extract text from PDF
    raw_text = extract_pdf_text(pdf_path)
    
    # Clean the text
    cleaned_text = clean_text(raw_text)
    
    # Split into chunks
    chunks = split_text(cleaned_text, chunk_size, chunk_overlap)
    
    return chunks

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_extractor.py <path_to_pdf>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    chunks = process_pdf(pdf_path)
    
    print(f"Extracted {len(chunks)} chunks from {pdf_path}")
    print("\nSample chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
