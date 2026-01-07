#!/usr/bin/env python3
"""
Test script for Ollama Cloud API with logprobs support.

Ollama Cloud supports OpenAI-compatible API at /v1/chat/completions endpoint.
Logprobs are supported with the `logprobs` and `top_logprobs` parameters.

API Documentation: https://docs.ollama.com/cloud
Cloud Models: https://ollama.com/search?c=cloud
"""

import os
import math
from openai import OpenAI


# =============================================================================
# Ollama Cloud Configuration
# =============================================================================

# Ollama Cloud base URL for OpenAI-compatible API
OLLAMA_CLOUD_BASE_URL = "https://ollama.com/v1"

# Available DeepSeek cloud models:
# - deepseek-v3.1 (671B parameters, hybrid thinking/non-thinking mode)
# - deepseek-v3.2 (high efficiency with superior reasoning)
DEFAULT_MODEL = "deepseek-v3.2"


def test_ollama_cloud_logprobs():
    """
    Test Ollama Cloud API with DeepSeek V3.2 model and logprobs output.

    Requires OLLAMA_API_KEY environment variable.
    Get your API key at: https://ollama.com/settings/keys
    """

    # Configuration
    base_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_CLOUD_BASE_URL)
    api_key = os.getenv("OLLAMA_API_KEY")
    model_name = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)

    if not api_key:
        print("ERROR: OLLAMA_API_KEY environment variable is required!")
        print("Get your API key at: https://ollama.com/settings/keys")
        print()
        print("Example:")
        print('  export OLLAMA_API_KEY="your-api-key-here"')
        print("  python test_ollama_logprobs.py")
        return None

    # Initialize OpenAI-compatible client
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    print(f"Testing Ollama Cloud API")
    print(f"Base URL: {base_url}")
    print(f"Model: {model_name}")
    print("-" * 50)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer concisely."
                },
                {
                    "role": "user",
                    "content": "What is 2 + 2? Answer with just the number."
                }
            ],
            logprobs=True,
            top_logprobs=5,  # Return top 5 token probabilities
            max_tokens=50,
            temperature=0.7,
        )

        print("\n=== Response ===")
        print(f"Content: {response.choices[0].message.content}")
        print(f"Finish reason: {response.choices[0].finish_reason}")

        print("\n=== Logprobs ===")
        if response.choices[0].logprobs and response.choices[0].logprobs.content:
            for i, token_info in enumerate(response.choices[0].logprobs.content):
                print(f"\nToken {i + 1}: '{token_info.token}'")
                print(f"  Logprob: {token_info.logprob:.4f}")
                print(f"  Probability: {math.exp(token_info.logprob):.4f}")

                if token_info.top_logprobs:
                    print(f"  Top alternatives:")
                    for alt in token_info.top_logprobs:
                        prob = math.exp(alt.logprob)
                        print(f"    '{alt.token}': logprob={alt.logprob:.4f} (prob={prob:.4f})")
        else:
            print("No logprobs returned. Model may not support logprobs.")

        print("\n=== Usage ===")
        if response.usage:
            print(f"Prompt tokens: {response.usage.prompt_tokens}")
            print(f"Completion tokens: {response.usage.completion_tokens}")
            print(f"Total tokens: {response.usage.total_tokens}")

        return response

    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        raise


def test_ollama_cloud_logprobs_raw():
    """
    Alternative: Direct HTTP request to Ollama Cloud API.

    Uses the OpenAI-compatible /v1/chat/completions endpoint.
    """
    import requests
    import json

    base_url = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
    api_key = os.getenv("OLLAMA_API_KEY")
    model_name = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)

    if not api_key:
        print("ERROR: OLLAMA_API_KEY environment variable is required!")
        print("Get your API key at: https://ollama.com/settings/keys")
        return None

    # OpenAI-compatible endpoint
    url = f"{base_url}/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": "What is 2 + 2? Answer with just the number."}
        ],
        "logprobs": True,
        "top_logprobs": 5,
        "max_tokens": 50,
        "stream": False,
    }

    print(f"Making request to: {url}")
    print(f"Model: {model_name}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print("-" * 50)

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()

    data = response.json()
    print(f"\nResponse:\n{json.dumps(data, indent=2)}")

    # Parse and display logprobs nicely
    if "choices" in data and data["choices"]:
        choice = data["choices"][0]
        if "logprobs" in choice and choice["logprobs"]:
            print("\n=== Parsed Logprobs ===")
            content_logprobs = choice["logprobs"].get("content", [])
            for i, token_info in enumerate(content_logprobs):
                token = token_info.get("token", "")
                logprob = token_info.get("logprob", 0)
                print(f"Token {i + 1}: '{token}' (logprob={logprob:.4f}, prob={math.exp(logprob):.4f})")

    return data


def test_ollama_native_api():
    """
    Test using Ollama's native Python client (ollama package).

    Note: Native Ollama API may have different logprobs support.
    """
    try:
        from ollama import Client
    except ImportError:
        print("ERROR: ollama package not installed.")
        print("Install with: pip install ollama")
        return None

    api_key = os.getenv("OLLAMA_API_KEY")
    model_name = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)

    if not api_key:
        print("ERROR: OLLAMA_API_KEY environment variable is required!")
        return None

    client = Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {api_key}"}
    )

    print(f"Testing Ollama Native API")
    print(f"Model: {model_name}")
    print("-" * 50)

    # Note: Native API uses different parameters
    response = client.chat(
        model=model_name,
        messages=[
            {"role": "user", "content": "What is 2 + 2? Answer with just the number."}
        ],
        options={
            "num_predict": 50,
            "temperature": 0.7,
        }
    )

    print(f"\nResponse: {response}")
    return response


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Ollama Cloud Logprobs Test")
    print("=" * 60)
    print()
    print("Configuration:")
    print(f"  Default Base URL: {OLLAMA_CLOUD_BASE_URL}")
    print(f"  Default Model: {DEFAULT_MODEL}")
    print()
    print("Environment variables:")
    print("  OLLAMA_API_KEY  - API key (required, get at https://ollama.com/settings/keys)")
    print("  OLLAMA_BASE_URL - API endpoint (optional)")
    print("  OLLAMA_MODEL    - Model name (optional)")
    print()
    print("Usage:")
    print("  python test_ollama_logprobs.py          # OpenAI-compatible client")
    print("  python test_ollama_logprobs.py --raw    # Raw HTTP request")
    print("  python test_ollama_logprobs.py --native # Ollama native client")
    print()

    if "--raw" in sys.argv:
        test_ollama_cloud_logprobs_raw()
    elif "--native" in sys.argv:
        test_ollama_native_api()
    else:
        test_ollama_cloud_logprobs()
