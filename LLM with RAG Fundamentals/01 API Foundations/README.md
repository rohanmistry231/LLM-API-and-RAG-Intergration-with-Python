# 🛠️ API Foundations

## 📖 Introduction

**API Foundations** introduces integrating OpenAI’s API, covering authentication, request structure, and error handling. This guide provides hands-on Python examples, setting the stage for RAG applications, aligned with the AI-driven era (May 3, 2025).

## 🌟 Key Concepts

- **API Concepts**: REST APIs, endpoints, JSON payloads.
- **Authentication**: Securing OpenAI API with keys.
- **Environment Setup**: Configuring Python for API use.
- **Request Handling**: Sending and parsing responses.

## 🛠️ Practical Example

```python
%% api_foundations.py
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
```

## 📊 Visualization Output

The code generates a bar chart (`api_foundations_output.png`) showing successful and failed API requests, illustrating reliability.

## 💡 Applications

- **Chatbots**: Authenticate APIs for conversational systems.
- **Content Generation**: Set up APIs for text creation.
- **RAG Preparation**: Establish API connectivity for retrieval systems.

## 🏆 Practical Tasks

1. Authenticate and test OpenAI API connectivity.
2. Handle basic API errors (e.g., invalid key, rate limits).
3. Visualize API request success and failure rates.

## 💡 Interview Scenarios

**Question**: What are the key components of an OpenAI API request?  
**Answer**: Endpoint, headers (with API key), and JSON payload (model, messages).  
**Key**: Authentication ensures secure access.  
**Example**: `requests.post(url, headers={"Authorization": f"Bearer {api_key}"}, json=payload)`

**Coding Task**: Authenticate and send a query to OpenAI API.  
**Tip**: Use `requests.post` with headers and payload.

## 📚 Resources

- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [Python Requests Documentation](https://requests.readthedocs.io/)

## 🤝 Contributions

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/api-foundations`).
3. Commit changes (`git commit -m 'Add API foundations content'`).
4. Push to the branch (`git push origin feature/api-foundations`).
5. Open a Pull Request.