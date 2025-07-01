#!/usr/bin/env python3
"""
Discover available GitHub Models

This script helps discover what models are available in GitHub Models.
"""

import os
import sys
import requests
import json
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

def test_common_models():
    """Test common model names to see what's available"""
    print("🔍 Testing Common GitHub Models")
    print("=" * 40)
    
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        print("❌ GITHUB_TOKEN not set")
        return
    
    endpoint = "https://models.github.ai/inference"
    
    # Common model names to test
    models_to_test = [
        "gpt-4",
        "gpt-3.5-turbo", 
        "claude-3.5-sonnet",
        "claude-3-haiku",
        "deepseek/deepseek-coder-33b-instruct",
        "deepseek/deepseek-coder-6.7b-instruct",
        "microsoft/phi-3.5",
        "microsoft/phi-3.5-mini",
        "microsoft/phi-2",
        "codellama/codellama-34b-instruct",
        "meta-llama/llama-3.1-8b-instruct",
        "meta-llama/llama-3.1-70b-instruct",
        "anthropic/claude-3.5-sonnet",
        "anthropic/claude-3-haiku",
        "openai/gpt-4",
        "openai/gpt-3.5-turbo"
    ]
    
    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(token),
    )
    
    available_models = []
    
    for model in models_to_test:
        print(f"🔄 Testing: {model}")
        try:
            response = client.complete(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say 'Hello' and nothing else."}
                ],
                temperature=0.1,
                max_tokens=10,
                model=model
            )
            print(f"✅ {model} - Available")
            available_models.append(model)
        except Exception as e:
            error_msg = str(e)
            if "unknown_model" in error_msg.lower():
                print(f"❌ {model} - Not available")
            else:
                print(f"⚠️  {model} - Error: {error_msg}")
    
    print(f"\n📊 Summary: {len(available_models)} models available")
    if available_models:
        print("✅ Available models:")
        for model in available_models:
            print(f"   - {model}")

def test_github_api():
    """Test GitHub API to see if we can get model information"""
    print("\n🔗 Testing GitHub API for Models")
    print("-" * 40)
    
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        print("❌ GITHUB_TOKEN not set")
        return
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/json',
        'X-GitHub-Api-Version': '2022-11-28'
    }
    
    # Try different GitHub API endpoints
    endpoints = [
        'https://api.github.com/models',
        'https://api.github.com/copilot/v1/models',
        'https://api.github.com/v1/models',
        'https://api.github.com/marketplace/models'
    ]
    
    for endpoint in endpoints:
        print(f"🔄 Testing: {endpoint}")
        try:
            response = requests.get(endpoint, headers=headers, timeout=10)
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                print(f"   ✅ Success: {len(response.text)} characters")
                try:
                    data = response.json()
                    if isinstance(data, list):
                        print(f"   📊 Found {len(data)} items")
                    elif isinstance(data, dict):
                        print(f"   📊 Keys: {list(data.keys())}")
                except:
                    print(f"   📄 Response is not JSON")
            else:
                print(f"   ❌ Failed: {response.text[:100]}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

def fetch_model_catalog():
    """Fetch the model catalog from GitHub Models API"""
    print("\n📋 Fetching Model Catalog")
    print("-" * 40)
    
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        print("❌ GITHUB_TOKEN not set")
        return
    
    endpoint = "https://models.github.ai/inference"
    
    try:
        # Try to get the model catalog
        import requests
        
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        
        catalog_url = f"{endpoint}/catalog/models"
        print(f"🔄 Fetching catalog from: {catalog_url}")
        
        response = requests.get(catalog_url, headers=headers, timeout=30)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                catalog = response.json()
                print(f"   ✅ Success: Found {len(catalog)} models")
                
                # Display available models
                print("\n📊 Available Models:")
                for i, model in enumerate(catalog[:20], 1):  # Show first 20
                    if isinstance(model, dict):
                        name = model.get('name', 'Unknown')
                        publisher = model.get('publisher', 'Unknown')
                        full_name = f"{publisher}/{name}"
                        print(f"   {i:2d}. {full_name}")
                    else:
                        print(f"   {i:2d}. {model}")
                
                if len(catalog) > 20:
                    print(f"   ... and {len(catalog) - 20} more models")
                
                return catalog
                
            except json.JSONDecodeError:
                print(f"   ❌ Response is not JSON: {response.text[:200]}")
                return None
        else:
            print(f"   ❌ Failed: {response.text}")
            return None
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

def test_catalog_models(catalog):
    """Test models from the catalog"""
    if not catalog:
        return
    
    print("\n🧪 Testing Catalog Models")
    print("-" * 40)
    
    token = os.getenv('GITHUB_TOKEN')
    endpoint = "https://models.github.ai/inference"
    
    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(token),
    )
    
    # Test first few models from catalog
    test_count = min(5, len(catalog))
    available_models = []
    
    for i in range(test_count):
        model_info = catalog[i]
        if isinstance(model_info, dict):
            model_name = f"{model_info.get('publisher', 'unknown')}/{model_info.get('name', 'unknown')}"
        else:
            model_name = str(model_info)
        
        print(f"🔄 Testing: {model_name}")
        try:
            response = client.complete(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say 'Hello' and nothing else."}
                ],
                temperature=0.1,
                max_tokens=10,
                model=model_name
            )
            print(f"✅ {model_name} - Available")
            available_models.append(model_name)
        except Exception as e:
            error_msg = str(e)
            if "unknown_model" in error_msg.lower():
                print(f"❌ {model_name} - Not available")
            elif "rate limit" in error_msg.lower() or "too many requests" in error_msg.lower():
                print(f"⚠️  {model_name} - Rate limited")
                break
            else:
                print(f"⚠️  {model_name} - Error: {error_msg}")
    
    print(f"\n📊 Summary: {len(available_models)} models tested successfully")
    if available_models:
        print("✅ Working models:")
        for model in available_models:
            print(f"   - {model}")

def main():
    """Main function"""
    print("🚀 GitHub Models Discovery Tool")
    print("=" * 40)
    
    # Use environment variable instead of hardcoded token
    if not os.getenv('GITHUB_TOKEN'):
        print("❌ GITHUB_TOKEN environment variable not set")
        print("💡 Set it with: export GITHUB_TOKEN='your-token-here'")
        return
    
    # Fetch model catalog first
    catalog = fetch_model_catalog()
    
    # Test catalog models
    if catalog:
        test_catalog_models(catalog)
    
    # Test common models
    test_common_models()
    test_github_api()
    
    print("\n💡 Next steps:")
    print("   1. Check the GitHub Models marketplace at github.com/marketplace/models")
    print("   2. Use the Azure AI Inference SDK documentation")
    print("   3. Try different model naming conventions")

if __name__ == '__main__':
    main() 
