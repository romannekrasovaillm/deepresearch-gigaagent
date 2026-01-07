#!/usr/bin/env python3
"""
Test script for Ollama API with logprobs support using OpenAI-compatible endpoint.

Ollama supports OpenAI-compatible API at /v1/chat/completions endpoint.
Logprobs are supported with the `logprobs` and `top_logprobs` parameters.
"""

import os
from openai import OpenAI


def test_ollama_logprobs():
    """Test Ollama Cloud API with DeepSeek model and logprobs output."""

    # Ollama Cloud configuration
    # For local Ollama: base_url="http://localhost:11434/v1"
    # For Ollama Cloud: use the cloud endpoint
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    api_key = os.getenv("OLLAMA_API_KEY", "ollama")  # Ollama doesn't require real key for local

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    # DeepSeek model name in Ollama
    # Common names: deepseek-r1, deepseek-coder, deepseek-v3
    # For DeepSeek 3.2B: deepseek-r1:1.5b, deepseek-r1:7b, etc.
    model_name = os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b")

    print(f"Testing Ollama API")
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
                print(f"  Probability: {2.718281828 ** token_info.logprob:.4f}")

                if token_info.top_logprobs:
                    print(f"  Top alternatives:")
                    for alt in token_info.top_logprobs:
                        prob = 2.718281828 ** alt.logprob
                        print(f"    '{alt.token}': logprob={alt.logprob:.4f} (prob={prob:.4f})")
        else:
            print("No logprobs returned. Model may not support logprobs.")

        print("\n=== Usage ===")
        print(f"Prompt tokens: {response.usage.prompt_tokens}")
        print(f"Completion tokens: {response.usage.completion_tokens}")
        print(f"Total tokens: {response.usage.total_tokens}")

        return response

    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        raise


def test_ollama_logprobs_raw():
    """Alternative: Direct HTTP request to Ollama API."""
    import requests
    import json

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    api_key = os.getenv("OLLAMA_API_KEY", "")
    model_name = os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b")

    # OpenAI-compatible endpoint
    url = f"{base_url}/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": "What is 2 + 2?"}
        ],
        "logprobs": True,
        "top_logprobs": 5,
        "max_tokens": 50,
        "stream": False,
    }

    print(f"Making request to: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print("-" * 50)

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    data = response.json()
    print(f"\nResponse:\n{json.dumps(data, indent=2)}")

    return data


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Ollama Logprobs Test - OpenAI-compatible API")
    print("=" * 60)
    print()
    print("Environment variables:")
    print("  OLLAMA_BASE_URL - API endpoint (default: http://localhost:11434/v1)")
    print("  OLLAMA_API_KEY  - API key for Ollama Cloud")
    print("  OLLAMA_MODEL    - Model name (default: deepseek-r1:1.5b)")
    print()

    if "--raw" in sys.argv:
        test_ollama_logprobs_raw()
    else:
        test_ollama_logprobs()
