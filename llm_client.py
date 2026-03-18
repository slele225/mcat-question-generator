"""
OpenAI-compatible LLM client for MCAT generation.
Works with vLLM, OpenAI API, OpenRouter, and other compatible backends.
"""

import os
import logging
from typing import Dict, Any, Optional

from openai import OpenAI

import config

logger = logging.getLogger(__name__)


class LLMClient:
    """Simple OpenAI-compatible LLM client."""
    
    def __init__(
        self,
        model: str = config.MODEL_NAME,
        base_url: Optional[str] = config.BASE_URL,
        api_key_env: str = config.API_KEY_ENV_VAR,
        temperature: float = config.TEMPERATURE,
        max_tokens: int = config.MAX_TOKENS,
        top_p: float = config.TOP_P,
    ):
        """
        Initialize the LLM client.
        
        Args:
            model: Model name to use
            base_url: API base URL (None = use default)
            api_key_env: Environment variable name containing API key
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        
        # Get API key from environment
        api_key = os.getenv(api_key_env)
        if not api_key and base_url and "localhost" not in base_url:
            logger.warning(f"API key not found in environment variable {api_key_env}")
        
        # Initialize client
        client_kwargs = {}
        if base_url:
            client_kwargs["base_url"] = base_url
        if api_key:
            client_kwargs["api_key"] = api_key
        else:
            client_kwargs["api_key"] = "EMPTY"  # For local vLLM
        
        self.client = OpenAI(**client_kwargs)
    
    def generate(self, prompt: str, system_message: Optional[str] = None) -> Optional[str]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: The user prompt
            system_message: Optional system message
        
        Returns:
            Generated text or None if failed
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return None
    
    def generate_json(self, prompt: str, system_message: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Generate and parse JSON from a prompt.
        
        Args:
            prompt: The user prompt
            system_message: Optional system message
        
        Returns:
            Parsed JSON or None if failed
        """
        import json
        
        response = self.generate(prompt, system_message)
        if not response:
            return None
        
        # Try to parse JSON
        try:
            # Clean response (remove markdown fences if present)
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            
            return json.loads(cleaned.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {response}")
            return None


# Singleton instance for easy import
_default_client = None


def get_client() -> LLMClient:
    """Get or create the default LLM client."""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


def set_client(client: LLMClient):
    """Set the default client (useful for testing)."""
    global _default_client
    _default_client = client