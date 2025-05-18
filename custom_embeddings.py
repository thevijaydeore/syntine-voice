"""
Custom embeddings implementation for Groq
"""

import os
import logging
from typing import List, Optional

import os
import requests
import json
from langchain_core.embeddings import Embeddings

logger = logging.getLogger("voice-agent.groq-embeddings")

class CustomGroqEmbeddings(Embeddings):
    """
    Custom implementation of Groq embeddings for LangChain.
    """
    
    def __init__(
        self,
        model: str = "llama3-8b-8192",  # Default model that supports embeddings
        api_key: Optional[str] = None,
        dimensions: int = 1536,  # Default dimensions
    ):
        """
        Initialize the Groq embeddings.
        
        Args:
            model: The Groq model to use for embeddings
            api_key: Groq API key (defaults to GROQ_API_KEY environment variable)
            dimensions: Dimensions of the embeddings
        """
        self.model = model
        self.dimensions = dimensions
        
        # Use provided API key or get from environment
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            logger.warning("GROQ_API_KEY not found in environment variables")
        
        # Set up API endpoint and headers
        self.api_url = "https://api.groq.com/v1/embeddings"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            texts: List of document texts
            
        Returns:
            List of embeddings, one for each document
        """
        embeddings = []
        
        try:
            for text in texts:
                # Prepare request data
                request_data = {
                    "model": self.model,
                    "input": text
                }
                
                # Call Groq API using requests
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=request_data
                )
                
                # Check if request was successful
                response.raise_for_status()
                
                # Parse response
                response_data = response.json()
                
                # Extract embedding from response
                embedding = response_data["data"][0]["embedding"]
                embeddings.append(embedding)
                
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings with Groq: {e}")
            # Return empty embeddings on error
            return [[0.0] * self.dimensions] * len(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embeddings for a query.
        
        Args:
            text: Query text
            
        Returns:
            Embedding for the query
        """
        try:
            # Prepare request data
            request_data = {
                "model": self.model,
                "input": text
            }
            
            # Call Groq API using requests
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=request_data
            )
            
            # Check if request was successful
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            
            # Extract embedding from response
            return response_data["data"][0]["embedding"]
            
        except Exception as e:
            logger.error(f"Error generating query embedding with Groq: {e}")
            # Return empty embedding on error
            return [0.0] * self.dimensions
