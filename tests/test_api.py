"""Test script for RAG API."""

import json
from pathlib import Path

import requests

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:", response.json())
    assert response.status_code == 200


def test_upload():
    """Test document upload."""
    # Create a sample text file
    sample_file = Path("tests/test_sample.txt")
    sample_file.write_text("""
    Machine Learning Overview
    
    Machine learning is a subset of artificial intelligence that enables computers 
    to learn from data without being explicitly programmed. There are three main 
    types of machine learning:
    
    1. Supervised Learning: The algorithm learns from labeled training data
    2. Unsupervised Learning: The algorithm finds patterns in unlabeled data
    3. Reinforcement Learning: The algorithm learns through trial and error
    
    Deep learning is a specialized form of machine learning that uses neural 
    networks with multiple layers. It has revolutionized computer vision, 
    natural language processing, and speech recognition.
    """)

    with open(sample_file, "rb") as f:
        response = requests.post(
            f"{BASE_URL}/upload",
            files={"file": ("test_sample.txt", f, "text/plain")},
        )

    print("Upload Response:", json.dumps(response.json(), indent=2))
    sample_file.unlink()  # Clean up
    assert response.status_code == 200


def test_query_baseline():
    """Test baseline query."""
    payload = {
        "question": "What is supervised learning?",
        "mode": "baseline",
        "top_k": 3,
    }

    response = requests.post(f"{BASE_URL}/query", json=payload)
    result = response.json()

    print("\n=== BASELINE MODE ===")
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Tokens: {result['metrics']['original_tokens']}")
    print(f"Cost: ${result['metrics']['estimated_cost_usd']:.6f}")
    print(f"Time: {result['metrics']['total_time_ms']:.1f}ms")

    assert response.status_code == 200


def test_query_semantic():
    """Test semantic highlighting query."""
    payload = {
        "question": "What is supervised learning?",
        "mode": "semantic",
        "top_k": 3,
    }

    response = requests.post(f"{BASE_URL}/query", json=payload)
    result = response.json()

    print("\n=== SEMANTIC MODE ===")
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Original Tokens: {result['metrics']['original_tokens']}")
    print(f"Pruned Tokens: {result['metrics']['pruned_tokens']}")
    print(f"Savings: {result['metrics']['token_savings_pct']:.1f}%")
    print(f"Cost: ${result['metrics']['estimated_cost_usd']:.6f}")
    print(f"Cost Savings: ${result['metrics']['cost_savings_usd']:.6f}")
    print(f"Compression: {result['metrics']['compression_rate']:.2%}")

    assert response.status_code == 200


def test_query_full():
    """Test full mode with HHEM."""
    payload = {
        "question": "What is supervised learning?",
        "mode": "full",
        "top_k": 3,
    }

    response = requests.post(f"{BASE_URL}/query", json=payload)
    result = response.json()

    print("\n=== FULL MODE (Semantic + HHEM) ===")
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Token Savings: {result['metrics']['token_savings_pct']:.1f}%")
    print(f"Cost: ${result['metrics']['estimated_cost_usd']:.6f}")
    print(f"HHEM Score: {result['metrics']['hhem_score']:.3f}")
    print(f"Is Hallucinated: {result['metrics']['is_hallucinated']}")
    print(f"Warning: {result['warning']}")

    assert response.status_code == 200


def test_compare():
    """Test comparison endpoint."""
    payload = {"question": "What is deep learning?", "top_k": 3}

    response = requests.post(f"{BASE_URL}/compare", json=payload)
    result = response.json()

    print("\n=== COMPARISON ===")
    print(f"Question: {result['question']}")
    print(f"\nBaseline Cost: ${result['summary']['baseline_cost']:.6f}")
    print(f"Semantic Cost: ${result['summary']['semantic_cost']:.6f}")
    print(f"Full Cost: ${result['summary']['full_cost']:.6f}")
    print(f"\nToken Savings: {result['summary']['token_savings_pct']:.1f}%")
    print(f"Cost Savings: ${result['summary']['cost_savings_usd']:.6f}")
    print(f"HHEM Score: {result['summary']['hhem_score']:.3f}")
    print(f"\nRecommendation: {result['summary']['recommendation']}")

    assert response.status_code == 200


if __name__ == "__main__":
    print("Starting API tests...\n")

    test_health()
    print("✓ Health check passed\n")

    test_upload()
    print("✓ Upload test passed\n")

    test_query_baseline()
    test_query_semantic()
    test_query_full()
    print("✓ Query tests passed\n")

    test_compare()
    print("✓ Comparison test passed\n")

    print("All tests passed! ✅")
