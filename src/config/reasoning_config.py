"""
Configuration for ReasoningLayer API providers.
"""
import os
from typing import Dict, Any

def get_reasoning_config() -> Dict[str, Any]:
    """Get ReasoningLayer configuration from environment variables."""
    
    # Default to Perplexity if no preference specified
    provider = os.getenv("REASONING_PROVIDER", "perplexity").lower()
    
    # Provider-specific configurations
    if provider == "perplexity":
        config = {
            "provider": "perplexity",
            "model_name": os.getenv("PERPLEXITY_MODEL", "sonar-pro"),
            "api_key_env": "PERPLEXITY_API_KEY"
        }
    elif provider == "deepseek":
        config = {
            "provider": "deepseek", 
            "model_name": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            "api_key_env": "DEEPSEEK_API_KEY"
        }
    else:
        # Fallback to Perplexity
        config = {
            "provider": "perplexity",
            "model_name": "sonar-pro",
            "api_key_env": "PERPLEXITY_API_KEY"
        }
    
    return config

def create_reasoning_layer():
    """Factory function to create ReasoningLayer with proper configuration."""
    from src.modules.ReasoningLayer import ReasoningLayer
    
    config = get_reasoning_config()
    return ReasoningLayer(
        model_name=config["model_name"],
        provider=config["provider"]
    )
