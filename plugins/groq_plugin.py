"""
Custom Groq plugin for LiveKit agents
"""

import os
import logging
from typing import Any, Dict, List, Optional, Union

import groq
from livekit.agents import LLMPlugin, LLMOptions, LLMResponse

logger = logging.getLogger("voice-agent.groq")

class LLM(LLMPlugin):
    """
    Groq LLM plugin for LiveKit agents.
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
        Initialize the Groq LLM plugin.

        Args:
            model: The Groq model to use
            api_key: Groq API key (defaults to GROQ_API_KEY environment variable)
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum number of tokens to generate
        """
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        
        # Use provided API key or get from environment
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            logger.warning("GROQ_API_KEY not found in environment variables")
        
        # Initialize Groq client
        self.client = groq.Client(api_key=self.api_key)
        
    async def generate(self, options: LLMOptions) -> LLMResponse:
        """
        Generate a response using Groq's API.
        
        Args:
            options: LLM options including messages and parameters
            
        Returns:
            LLMResponse with the generated content
        """
        try:
            # Prepare messages for Groq API
            messages = []
            for msg in options.messages:
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            # Add system message if provided
            if options.instructions:
                messages.insert(0, {
                    "role": "system",
                    "content": options.instructions
                })
            
            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                stream=False
            )
            
            # Extract content from response
            content = response.choices[0].message.content
            
            # Return LLMResponse
            return LLMResponse(content=content)
            
        except Exception as e:
            logger.error(f"Error generating response with Groq: {e}")
            # Return empty response on error
            return LLMResponse(content="I'm sorry, I encountered an error processing your request.")
