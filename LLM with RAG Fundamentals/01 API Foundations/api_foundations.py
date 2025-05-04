# Setup: pip install openai requests matplotlib pandas nltk
import os
import requests
import matplotlib.pyplot as plt
from collections import Counter
import nltk

def run_api_foundations_demo():
    # Synthetic Query Data
    queries = [
        "Explain neural networks in simple terms.",
        "Write a short story about AI.",
        "Summarize the benefits of cloud computing."
    ]
    print("Synthetic Data: Queries created")
    print(f"Queries: {queries}")

    # OpenAI API Configuration
    api_key = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Track request success
    success_counts = {"Successful": 0, "Failed": 0}

    for query in queries:
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": query}],
            "max_tokens": 100
        }
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            success_counts["Successful"] += 1
            print(f"Query: {query}")
            print(f"Response: {response.json()['choices'][0]['message']['content'].strip()}")
        except requests.RequestException as e:
            success_counts["Failed"] += 1
            print(f"Error for {query}: {e}")

    # Visualization
    plt.figure(figsize=(8, 4))
    plt.bar(success_counts.keys(), success_counts.values(), color=['green', 'red'])
    plt.title("API Request Success Rates")
    plt.xlabel("Status")
    plt.ylabel("Count")
    plt.savefig("api_foundations_output.png")
    print("Visualization: Success rates saved as api_foundations_output.png")

# Execute the demo
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    run_api_foundations_demo()