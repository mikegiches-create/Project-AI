"""
Mistral AI Integration for Market Lens
Using Hugging Face InferenceClient (FREE - Recommended 2024 Method)

This module provides a drop-in replacement for OpenAI's API
using Mistral-7B-Instruct-v0.2 via Hugging Face's official client.
"""

import os
from typing import Dict, List

# Using Hugging Face's official client library
try:
    from huggingface_hub import InferenceClient
except ImportError:
    raise ImportError(
        "Please install huggingface_hub: pip install huggingface_hub"
    )

class MistralClient:
    """Free Mistral AI client using Hugging Face InferenceClient"""
    
    def __init__(self, api_token: str = None, model: str = None):
        self.api_token = api_token or os.getenv("HUGGINGFACE_API_TOKEN")
        self.model = model or os.getenv("MISTRAL_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
        
        if not self.api_token:
            raise ValueError(
                "Hugging Face API token not found! "
                "Get your FREE token from https://huggingface.co/settings/tokens "
                "and add it to your .env file as HUGGINGFACE_API_TOKEN"
            )
        
        # Initialize the official Hugging Face Inference Client
        self.client = InferenceClient(token=self.api_token)
    
    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.45, max_tokens: int = 1000) -> str:
        """Generate chat completion using Mistral AI"""
        
        try:
            # Convert messages to the proper format
            formatted_messages = self._format_messages(messages)
            
            # Use the new chat_completion method
            response = self.client.chat_completion(
                messages=formatted_messages,
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9
            )
            
            # Extract the generated text
            if response and hasattr(response, 'choices') and len(response.choices) > 0:
                return response.choices[0].message.content.strip()
            else:
                return "⚠️ No response generated. Try again."
                
        except Exception as e:
            error_msg = str(e).lower()
            
            if "rate limit" in error_msg:
                return "⏱️ Rate limit reached. Please wait a moment and try again."
            elif "model is currently loading" in error_msg or "503" in error_msg:
                return "⏳ The AI model is loading. Please try again in 20-30 seconds!"
            elif "unauthorized" in error_msg or "401" in error_msg:
                return "❌ Invalid Hugging Face API token. Please check your .env file."
            elif "forbidden" in error_msg or "403" in error_msg:
                return "❌ Access forbidden. Check if your token has the right permissions."
            elif "not found" in error_msg or "404" in error_msg:
                return f"❌ Model not found: {self.model}. Check the model name in .env"
            elif "timeout" in error_msg:
                return "⏱️ Request timed out. Please try again!"
            else:
                return f"❌ Error: {str(e)}"
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Format messages for Hugging Face chat_completion"""
        formatted = []
        for msg in messages:
            formatted.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
        return formatted


class ChatCompletion:
    """OpenAI-compatible wrapper for Mistral AI"""
    
    def __init__(self, client: MistralClient):
        self.client = client
    
    def create(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.45, **kwargs):
        """OpenAI-compatible create method"""
        response_text = self.client.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=kwargs.get('max_tokens', 1000)
        )
        return MistralResponse(response_text)


class MistralResponse:
    """OpenAI-compatible response object"""
    def __init__(self, text: str):
        self.choices = [MistralChoice(text)]


class MistralChoice:
    """OpenAI-compatible choice object"""
    def __init__(self, text: str):
        self.message = MistralMessage(text)


class MistralMessage:
    """OpenAI-compatible message object"""
    def __init__(self, text: str):
        self.content = text


class MistralClientWrapper:
    """Main client wrapper that mimics OpenAI's interface"""
    
    def __init__(self, api_token: str = None):
        mistral_client = MistralClient(api_token=api_token)
        self.chat = type('Chat', (), {
            'completions': ChatCompletion(mistral_client)
        })()
