#!/usr/bin/env python3
"""
Test script for ReasoningLayer dual API support.
Tests both Perplexity and DeepSeek providers.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.modules.ReasoningLayer import ReasoningLayer

def test_provider(provider_name, model_name):
    """Test a specific provider."""
    print(f"\n🧪 Testing {provider_name} provider...")
    
    try:
        # Create ReasoningLayer with specific provider
        reasoning_layer = ReasoningLayer(model_name=model_name, provider=provider_name)
        print(f"✅ {provider_name} ReasoningLayer initialized successfully")
        
        # Test combined processing
        test_text = "How r u doing today? wat r u up 2?"
        cleaned, reasoned = reasoning_layer._process_input_combined(test_text, inner_speech_present=False)
        
        print(f"📝 Test input: '{test_text}'")
        print(f"🧽 Cleaned: '{cleaned}'")
        print(f"🧠 Reasoned: '{reasoned}'")
        
        return True
        
    except Exception as e:
        print(f"❌ {provider_name} test failed: {e}")
        return False

def test_factory_function():
    """Test the factory function with environment variables."""
    print("\n🏭 Testing factory function...")
    
    try:
        from src.config.reasoning_config import create_reasoning_layer, get_reasoning_config
        
        config = get_reasoning_config()
        print(f"📋 Configuration: {config}")
        
        reasoning_layer = create_reasoning_layer()
        print(f"✅ Factory function created {reasoning_layer.provider} ReasoningLayer")
        
        return True
        
    except Exception as e:
        print(f"❌ Factory function test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 ReasoningLayer Dual API Test")
    print("=" * 50)
    
    results = []
    
    # Test factory function
    results.append(test_factory_function())
    
    # Test Perplexity if API key available
    if os.getenv("PERPLEXITY_API_KEY"):
        results.append(test_provider("perplexity", "sonar-pro"))
    else:
        print("\n⚠️  Skipping Perplexity test - PERPLEXITY_API_KEY not set")
        
    # Test DeepSeek if API key available
    if os.getenv("DEEPSEEK_API_KEY"):
        results.append(test_provider("deepseek", "deepseek-chat"))
    else:
        print("\n⚠️  Skipping DeepSeek test - DEEPSEEK_API_KEY not set")
    
    # Summary
    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed - check API keys and network connectivity")

if __name__ == "__main__":
    main()
