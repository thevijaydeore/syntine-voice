"""
Knowledge Base Module for LiveKit Voice Agent

This module handles:
1. Creating vector embeddings from PDF chunks
2. Storing embeddings in a vector database
3. Retrieving relevant chunks based on queries
"""

import os
import logging
import pickle
from typing import List, Dict, Any, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore

# Ensure FAISS is properly imported through sentence-transformers
import sentence_transformers

from pdf_extractor import process_pdf

logger = logging.getLogger("voice-agent.knowledge")

class KnowledgeBase:
    """Knowledge base for RAG with the LiveKit Voice Agent."""
    
    def __init__(self, 
                 persist_directory: str = "data/vectorstore",
                 embedding_model: Any = None):
        """
        Initialize the knowledge base.
        
        Args:
            persist_directory: Directory to persist vector store
            embedding_model: Model to use for embeddings (default: OpenAIEmbeddings)
        """
        self.persist_directory = persist_directory
        
        # Create embedding model if not provided
        if embedding_model is None:
            # Using HuggingFace embeddings with API key
            # This uses the HuggingFace Inference API
            hf_api_key = os.environ.get("HUGGINGFACE_API_KEY")
            if not hf_api_key:
                logger.warning("HUGGINGFACE_API_KEY not found in environment variables")
            
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                api_key=hf_api_key,  # Use API key for cloud-based inference
                encode_kwargs={"normalize_embeddings": True}
            )
        else:
            self.embedding_model = embedding_model
            
        # Initialize vector store
        self.vectorstore = None
        
        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
    def add_pdf(self, 
                pdf_path: str, 
                chunk_size: int = 1000, 
                chunk_overlap: int = 200) -> None:
        """
        Add a PDF to the knowledge base.
        
        Args:
            pdf_path: Path to the PDF file
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
        """
        logger.info(f"Adding PDF to knowledge base: {pdf_path}")
        
        # Process PDF to get chunks
        chunks = process_pdf(pdf_path, chunk_size, chunk_overlap)
        logger.info(f"Created {len(chunks)} chunks from PDF")
        
        # Print first chunk for debugging
        if chunks:
            logger.info(f"Sample chunk: {chunks[0][:100]}...")
        
        # Create or update vector store
        if self.vectorstore is None:
            # Create new vector store
            try:
                self.vectorstore = FAISS.from_texts(
                    texts=chunks,
                    embedding=self.embedding_model
                )
                # Save the FAISS index
                self.vectorstore.save_local(self.persist_directory)
                logger.info(f"Created new FAISS vector store in {self.persist_directory} with {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Error creating vector store: {e}")
                raise
        else:
            # Add to existing vector store
            try:
                self.vectorstore.add_texts(chunks)
                logger.info(f"Added {len(chunks)} chunks to existing vector store")
            except Exception as e:
                logger.error(f"Error adding to vector store: {e}")
                raise
        
        # Persist vector store
        try:
            self.vectorstore.save_local(self.persist_directory)
            logger.info("FAISS vector store persisted to disk")
        except Exception as e:
            logger.error(f"Error persisting vector store: {e}")
            raise
            
        # Verify chunks were added by doing a simple search
        if chunks:
            try:
                sample_text = chunks[0][:50]  # Use first 50 chars of first chunk
                results = self.search(sample_text, k=1)
                if results:
                    logger.info("Vector store verification successful - found matching chunk")
                else:
                    logger.warning("Vector store verification failed - no chunks found for sample text")
            except Exception as e:
                logger.error(f"Error verifying vector store: {e}")
            
    def load(self) -> bool:
        """
        Load knowledge base from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        logger.info(f"Loading FAISS vector store from {self.persist_directory}")
        
        try:
            # Check if the directory exists and has content
            if not os.path.exists(self.persist_directory) or not os.listdir(self.persist_directory):
                logger.warning(f"Vector store directory {self.persist_directory} does not exist or is empty")
                return False
                
            # Load the vector store
            self.vectorstore = FAISS.load_local(
                folder_path=self.persist_directory,
                embeddings=self.embedding_model
            )
            
            logger.info("FAISS vector store loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False
            
    def search(self, query: str, k: int = 3) -> List[str]:
        """
        Search the knowledge base for relevant chunks.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of relevant text chunks
        """
        if self.vectorstore is None:
            logger.warning("Vector store not initialized, loading from disk")
            if not self.load():
                logger.error("Failed to load vector store")
                return []
        
        logger.info(f"Searching knowledge base for: {query}")
        
        try:
            # Perform similarity search with score
            documents_and_scores = self.vectorstore.similarity_search_with_score(query, k=k)
            
            # Log the scores for debugging
            for i, (doc, score) in enumerate(documents_and_scores):
                logger.info(f"Result {i+1} score: {score}, preview: {doc.page_content[:50]}...")
            
            # Extract text from documents
            results = [doc.page_content for doc, _ in documents_and_scores]
            
            logger.info(f"Found {len(results)} relevant chunks")
            
            # If no results found with similarity search, try a more basic approach
            if not results and k > 0:
                logger.warning("No results found with similarity search, trying more basic approach")
                try:
                    # For FAISS, we need to handle fallback differently since it doesn't have a .get() method
                    # We'll use the docstore that FAISS maintains internally
                    if hasattr(self.vectorstore, 'docstore') and self.vectorstore.docstore:
                        # Get all documents from the docstore
                        all_docs = [doc.page_content for doc in self.vectorstore.docstore.values()]
                        logger.info(f"FAISS vector store contains {len(all_docs)} total documents")
                        
                        # Simple keyword matching as fallback
                        query_terms = query.lower().split()
                        matched_docs = []
                        
                        for doc in all_docs:
                            doc_text = doc.lower()
                            matches = sum(1 for term in query_terms if term in doc_text)
                            if matches > 0:
                                matched_docs.append((doc, matches))
                        
                        # Sort by number of matching terms
                        matched_docs.sort(key=lambda x: x[1], reverse=True)
                        results = [doc for doc, _ in matched_docs[:k]]
                        
                        logger.info(f"Found {len(results)} chunks with keyword matching")
                except Exception as e:
                    logger.error(f"Error in fallback search: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []

def create_knowledge_base(pdf_paths: List[str]) -> KnowledgeBase:
    """
    Create a knowledge base from a list of PDF paths.
    
    Args:
        pdf_paths: List of paths to PDF files
        
    Returns:
        Initialized knowledge base
    """
    kb = KnowledgeBase()
    
    for pdf_path in pdf_paths:
        kb.add_pdf(pdf_path)
        
    return kb

if __name__ == "__main__":
    # Example usage
    import sys
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    if len(sys.argv) < 2:
        print("Usage: python knowledge_base.py <pdf_path1> [<pdf_path2> ...]")
        sys.exit(1)
    
    # Create knowledge base from PDFs
    kb = create_knowledge_base(sys.argv[1:])
    
    # Test search
    while True:
        query = input("\nEnter a query (or 'q' to quit): ")
        if query.lower() == 'q':
            break
            
        results = kb.search(query)
        
        print(f"\nFound {len(results)} relevant chunks:")
        for i, result in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            print(result)
