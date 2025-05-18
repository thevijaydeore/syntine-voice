"""
Custom LLM implementation for Groq to use with LiveKit
"""

import os
import logging
from typing import Any, Dict, List, Optional

import os
import requests
import json
from livekit.agents import Agent, ChatMessage

logger = logging.getLogger("voice-agent.groq-llm")

class GroqLLM:
    """
    Custom Groq LLM implementation for LiveKit agents.
    """

    def __init__(
        self,
        model: str = "qwen-qwq-32b",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 1024,
    ):
        """
        Initialize the Groq LLM.

        Args:
            model: The Groq model to use
            api_key: Groq API key (defaults to GROQ_API_KEY environment variable)
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum number of tokens to generate
        """
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        
        # Use provided API key or get from environment
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            logger.warning("GROQ_API_KEY not found in environment variables")
        
        # Set up API endpoint and headers
        self.api_url = "https://api.groq.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def __call__(self, messages: List[ChatMessage], instructions: Optional[str] = None) -> str:
        """
        Generate a response using Groq's API.
        
        Args:
            messages: List of chat messages
            instructions: System instructions
            
        Returns:
            Generated response content
        """
        try:
            # Prepare messages for Groq API
            groq_messages = []
            
            # Add system message if provided
            if instructions:
                groq_messages.append({
                    "role": "system",
                    "content": instructions
                })
            
            # Add chat messages
            for msg in messages:
                groq_messages.append({
                    "role": msg.role,
                    "content": msg.text_content
                })
            
            # Prepare request data
            request_data = {
                "model": self.model,
                "messages": groq_messages,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens
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
            
            # Extract content from response
            content = response_data["choices"][0]["message"]["content"]
            
            return content
            
        except Exception as e:
            logger.error(f"Error generating response with Groq: {e}")
            # Return error message on failure
            return "I'm sorry, I encountered an error processing your request."
