"""
Test script for PDF extraction

Usage:
    python test_pdf_extraction.py <path_to_pdf>
"""

import os
import sys
import logging
from pdf_extractor import process_pdf

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_pdf_extraction.py <path_to_pdf>")
        return
    
    pdf_path = sys.argv[1]
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        return
    
    print(f"Processing PDF: {pdf_path}")
    chunks = process_pdf(pdf_path)
    
    print(f"\nExtracted {len(chunks)} chunks from the PDF")
    
    # Display sample chunks
    print("\n=== SAMPLE CHUNKS ===")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ({len(chunk)} characters) ---")
        # Print first 200 characters of each chunk
        preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
        print(preview)
    
    print("\nPDF extraction successful!")
    print("These chunks are now ready to be embedded and stored in a vector database.")

if __name__ == "__main__":
    main()
