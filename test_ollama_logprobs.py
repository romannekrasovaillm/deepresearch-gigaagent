#!/usr/bin/env python3
"""
Test script for Ollama Cloud API with logprobs support.

IMPORTANT: Logprobs are supported via NATIVE Ollama API (/api/chat),
NOT via OpenAI-compatible API (/v1/chat/completions).

API Documentation:
- Native API: https://docs.ollama.com/api/chat
- OpenAI Compatibility: https://docs.ollama.com/api/openai-compatibility
- Cloud Models: https://ollama.com/search?c=cloud
"""

import os
import math
import json
import requests


# =============================================================================
# Ollama Cloud Configuration
# =============================================================================

# Ollama Cloud base URL
OLLAMA_CLOUD_URL = "https://ollama.com"

# Available DeepSeek cloud models:
# - deepseek-v3.1 (671B parameters, hybrid thinking/non-thinking mode)
# - deepseek-v3.2 (high efficiency with superior reasoning)
DEFAULT_MODEL = "deepseek-v3.2"


def test_ollama_native_logprobs():
    """
    Test Ollama API with logprobs using NATIVE /api/chat endpoint.

    This is the correct way to get logprobs from Ollama.
    The OpenAI-compatible endpoint (/v1/chat/completions) may not return logprobs.
    """
    base_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_CLOUD_URL)
    api_key = os.getenv("OLLAMA_API_KEY")
    model_name = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)

    # Native Ollama API endpoint
    url = f"{base_url}/api/chat"

    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Native Ollama API format with logprobs
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": "What is 2 + 2? Answer with just the number."}
        ],
        "logprobs": True,           # Enable logprobs
        "top_logprobs": 5,          # Return top 5 alternatives per token
        "stream": False,            # Disable streaming for easier parsing
        "options": {
            "temperature": 0.7,
            "num_predict": 50,
        }
    }

    print("=" * 60)
    print("Ollama Native API - Logprobs Test")
    print("=" * 60)
    print(f"\nEndpoint: {url}")
    print(f"Model: {model_name}")

    # Print curl command
    print("\n=== curl command ===")
    curl_headers = '-H "Content-Type: application/json"'
    if api_key:
        curl_headers += ' -H "Authorization: Bearer $OLLAMA_API_KEY"'
    print(f"""curl '{url}' \\
  {curl_headers} \\
  -d '{json.dumps(payload)}'
""")

    print("=== Request payload ===")
    print(json.dumps(payload, indent=2))
    print("-" * 50)

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)

        print(f"\n=== HTTP Response ===")
        print(f"Status: {response.status_code}")

        data = response.json()
        print(f"\n=== Response Body ===")
        print(json.dumps(data, indent=2))

        # Parse response
        if "message" in data:
            print(f"\n=== Content ===")
            print(data["message"].get("content", ""))

        # Parse logprobs
        if "logprobs" in data and data["logprobs"]:
            print(f"\n=== Logprobs ===")
            for i, token_info in enumerate(data["logprobs"]):
                token = token_info.get("token", "")
                logprob = token_info.get("logprob", 0)
                prob = math.exp(logprob) if logprob else 0
                print(f"\nToken {i + 1}: '{token}'")
                print(f"  Logprob: {logprob:.4f}")
                print(f"  Probability: {prob:.4f}")

                if "top_logprobs" in token_info and token_info["top_logprobs"]:
                    print(f"  Top alternatives:")
                    for alt in token_info["top_logprobs"]:
                        alt_token = alt.get("token", "")
                        alt_logprob = alt.get("logprob", 0)
                        alt_prob = math.exp(alt_logprob) if alt_logprob else 0
                        print(f"    '{alt_token}': logprob={alt_logprob:.4f} (prob={alt_prob:.4f})")
        else:
            print(f"\n=== Logprobs ===")
            print("No logprobs in response.")
            print("'logprobs' field:", data.get("logprobs"))

        response.raise_for_status()
        return data

    except requests.exceptions.RequestException as e:
        print(f"\nError: {type(e).__name__}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response text: {e.response.text}")
        raise


def test_openai_compatible():
    """
    Test OpenAI-compatible endpoint (may NOT return logprobs on Ollama Cloud).
    """
    base_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_CLOUD_URL)
    api_key = os.getenv("OLLAMA_API_KEY")
    model_name = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)

    url = f"{base_url}/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": "What is 2 + 2? Answer with just the number."}
        ],
        "logprobs": True,
        "top_logprobs": 5,
        "max_tokens": 50,
        "temperature": 0.7,
        "stream": False,
    }

    print("=" * 60)
    print("OpenAI-Compatible API - Logprobs Test")
    print("=" * 60)
    print(f"\nEndpoint: {url}")
    print(f"Model: {model_name}")
    print("\nNOTE: OpenAI-compatible endpoint may NOT return logprobs on Ollama Cloud.")
    print("Use --native for native API which supports logprobs.")

    print("\n=== Request payload ===")
    print(json.dumps(payload, indent=2))
    print("-" * 50)

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        data = response.json()

        print(f"\n=== Response ===")
        print(json.dumps(data, indent=2))

        if "choices" in data and data["choices"]:
            choice = data["choices"][0]
            print(f"\n=== Content ===")
            print(choice.get("message", {}).get("content", ""))

            logprobs = choice.get("logprobs")
            print(f"\n=== Logprobs ===")
            if logprobs:
                print(json.dumps(logprobs, indent=2))
            else:
                print(f"logprobs: {logprobs}")
                print("\n⚠️  Logprobs not returned. Try using --native flag instead.")

        response.raise_for_status()
        return data

    except requests.exceptions.RequestException as e:
        print(f"\nError: {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Ollama Logprobs Test")
    print("=" * 60)
    print()
    print("Configuration:")
    print(f"  Default URL: {OLLAMA_CLOUD_URL}")
    print(f"  Default Model: {DEFAULT_MODEL}")
    print()
    print("Environment variables:")
    print("  OLLAMA_API_KEY  - API key (get at https://ollama.com/settings/keys)")
    print("  OLLAMA_BASE_URL - API endpoint (optional)")
    print("  OLLAMA_MODEL    - Model name (optional)")
    print()
    print("Usage:")
    print("  python test_ollama_logprobs.py           # Native API (recommended)")
    print("  python test_ollama_logprobs.py --native  # Native API /api/chat")
    print("  python test_ollama_logprobs.py --openai  # OpenAI-compatible /v1/...")
    print()
    print("For logprobs, use NATIVE API (default or --native flag).")
    print("OpenAI-compatible API may not return logprobs on Ollama Cloud.")
    print()

    if "--openai" in sys.argv:
        test_openai_compatible()
    else:
        test_ollama_native_logprobs()
